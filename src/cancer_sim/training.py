"""Batched training of the NODE / PI-NODE ensembles.

A whole noisy ensemble (E members) is trained in one call.  Each member has
independent weights; the loss is summed per member so members do not interact,
exactly reproducing E separate trainings -- only far faster.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from . import surrogates as S
from .config import TrainConfig


def _set_requires_grad(params, value: bool):
    for p in params:
        p.requires_grad_(value)


def _clip_grad_per_member(params, max_norm: float, E: int):
    """Per-member gradient-norm clipping.

    Every parameter carries a leading ensemble dimension ``E``.  We compute the
    combined gradient norm of each member across *all* its parameters and scale
    that member's gradients independently, so members never couple through the
    clip (matching the original per-model ``clip_grad_norm_``).
    """
    device = params[0].device
    sq = torch.zeros(E, device=device)
    for p in params:
        if p.grad is not None:
            sq += p.grad.reshape(E, -1).pow(2).sum(dim=1)
    norm = sq.sqrt()
    scale = (max_norm / (norm + 1e-6)).clamp(max=1.0)   # (E,)
    for p in params:
        if p.grad is not None:
            view = scale.view([E] + [1] * (p.grad.dim() - 1))
            p.grad.mul_(view)


def _phase_schedule(kind: str, cfg: TrainConfig):
    if kind == "NODE":
        return [("joint", cfg.node_epochs, cfg.lr_node)]
    if kind == "PI_NODE":
        return [
            ("theta", cfg.pinode_theta_epochs, cfg.lr_pinode),
            ("phi", cfg.pinode_phi_epochs, cfg.lr_pinode),
            ("joint", cfg.pinode_joint_epochs, cfg.lr_pinode * 0.5),
        ]
    raise ValueError(kind)


def _configure_phase(model, kind: str, phase: str):
    if kind != "PI_NODE":
        _set_requires_grad(model.parameters(), True)
        return
    if phase == "theta":
        _set_requires_grad(model.residual.parameters(), False)
        _set_requires_grad(model.physics_parameters(), True)
    elif phase == "phi":
        _set_requires_grad(model.residual.parameters(), True)
        _set_requires_grad(model.physics_parameters(), False)
    elif phase == "joint":
        _set_requires_grad(model.parameters(), True)
    else:
        raise ValueError(phase)


def fit_ensemble(
    kind: str,
    t_fit: np.ndarray,
    Y_fit: np.ndarray,        # (T, E) per-member noisy observations
    U_fit: np.ndarray,
    cfg: TrainConfig,
    init,
    omega: Optional[float] = None,
    s_r: Optional[float] = None,
    time_norm: float = 80.0,
    device: Optional[torch.device] = None,
    sample_weights: Optional[np.ndarray] = None,
    autonomous: bool = False,
    activation: str = "relu",
    theta_init: Optional[dict] = None,
    log=lambda *_: None,
) -> Tuple[torch.nn.Module, Dict]:
    """Train an ensemble of NODE or PI-NODE surrogates simultaneously.

    ``autonomous`` / ``activation`` select the Autonomous Smooth Closure PI-NODE
    variant (time-free closure, tanh field). Defaults reproduce the paper model.

    Returns the trained (best-per-member) model and an info dict.
    """
    device = device or S.get_device()
    torch.manual_seed(cfg.seed)

    T, E = Y_fit.shape
    omega = cfg.omega if omega is None else omega
    s_r = cfg.s_r if s_r is None else s_r

    if kind == "NODE":
        model = S.BatchedNODE(E, init, hidden=cfg.hidden).to(device)
    elif kind == "PI_NODE":
        model = S.BatchedPINODE(E, init, omega=omega, s_r=s_r, hidden=cfg.hidden,
                                autonomous=autonomous, activation=activation,
                                theta_init=theta_init).to(device)
    else:
        raise ValueError(kind)

    t = torch.tensor(t_fit, dtype=S.DTYPE, device=device)
    U = torch.tensor(U_fit, dtype=S.DTYPE, device=device)
    Y = torch.tensor(Y_fit, dtype=S.DTYPE, device=device)   # (T, E)
    y0 = Y[0]                                                # (E,)

    if sample_weights is None:
        w = torch.ones(T, dtype=S.DTYPE, device=device)
    else:
        w = torch.tensor(sample_weights, dtype=S.DTYPE, device=device)
        w = w / (w.mean() + 1e-12)
    w = w.unsqueeze(1)                                       # (T,1)

    best_mse = torch.full((E,), float("inf"), device=device)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    loss_history = []

    for phase, n_epochs, lr in _phase_schedule(kind, cfg):
        if n_epochs <= 0:
            continue
        _configure_phase(model, kind, phase)
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError(f"No trainable parameters in phase {phase}")
        opt = torch.optim.Adam(params, lr=lr)

        for _ in range(n_epochs):
            opt.zero_grad(set_to_none=True)
            states = S.integrate(model, t, U, y0, time_norm)     # (T,E,2)
            pred_y = states[:, :, 0]                             # (T,E)
            per_member = (w * (pred_y - Y) ** 2).mean(dim=0)     # (E,)
            mse = per_member.mean()

            reg = torch.zeros((), device=device)
            if kind == "PI_NODE":
                idx = torch.linspace(0, T - 1, min(120, T), device=device).long()
                xs = states[idx].detach()
                tn = (t[idx] / time_norm).unsqueeze(1).expand(-1, E)
                Ub = U[idx].unsqueeze(1).expand(-1, E)
                # Evaluate residual over (sampled-T, E) points at once.
                if getattr(model, "autonomous", False):
                    feats = torch.stack([xs[:, :, 0], xs[:, :, 1], Ub], dim=-1)
                else:
                    feats = torch.stack([xs[:, :, 0], xs[:, :, 1], tn, Ub], dim=-1)
                res = model.residual(feats)  # (sampled-T, E, 2)
                reg = cfg.l2_residual * (res ** 2).mean()

            (mse + reg).backward()
            _clip_grad_per_member(params, cfg.grad_clip, E)
            opt.step()

            with torch.no_grad():
                improved = per_member < best_mse
                if improved.any():
                    best_mse = torch.where(improved, per_member.detach(), best_mse)
                    for name, p in model.state_dict().items():
                        # leading dim is the ensemble dim for all params/buffers
                        best_state[name][improved] = p[improved].clone()
            loss_history.append(float(mse.detach().cpu()))

        log(f"    {kind} phase={phase} epochs={n_epochs} lr={lr:.2e} "
            f"meanMSE={float(per_member.mean().detach()):.3e}")

    model.load_state_dict(best_state)

    info = {
        "kind": kind,
        "ensemble": E,
        "best_mse_mean": float(best_mse.mean().cpu()),
        "loss_history": loss_history[:: max(1, len(loss_history) // 200)],
        "omega": omega,
        "s_r": s_r,
        "l2_residual": cfg.l2_residual,
    }
    if kind == "PI_NODE":
        rho, gamma, Keff, mu, tau = (p.detach().cpu().numpy()
                                     for p in model.positive_params())
        info["mechanistic_params_mean"] = {
            "rho": float(rho.mean()), "gamma": float(gamma.mean()),
            "Keff": float(Keff.mean()), "mu": float(mu.mean()), "tau": float(tau.mean()),
        }
    return model, info


@torch.no_grad()
def predict_ensemble(model, t_eval: np.ndarray, U_eval: np.ndarray,
                     y0: np.ndarray, time_norm: float,
                     device: Optional[torch.device] = None):
    """Forecast all members on the evaluation grid. Returns (y (T,E), z (T,E))."""
    device = device or next(model.parameters()).device
    model.eval()
    t = torch.tensor(t_eval, dtype=S.DTYPE, device=device)
    U = torch.tensor(U_eval, dtype=S.DTYPE, device=device)
    y0_t = torch.tensor(np.atleast_1d(y0), dtype=S.DTYPE, device=device)
    states = S.integrate(model, t, U, y0_t, time_norm)
    return states[:, :, 0].cpu().numpy(), states[:, :, 1].cpu().numpy()
