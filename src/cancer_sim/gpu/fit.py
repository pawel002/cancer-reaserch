"""Batched multi-restart fitting and scoring of the reduced surrogates.

Every surrogate in the study is fitted the same way -- same integrator, same
optimiser, same budget, same restarts -- so a difference in forecast accuracy is
a difference between *models*, not between fitting procedures.

The batch axis holds (patient x beam configuration x noisy realisation x random
restart).  With 117 patients, 4 configurations, 20 realisations and 3 restarts
that is ~28 000 independent reduced models trained simultaneously in one process
on one GPU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import surrogates as S
from .dataset import Bench
from .surrogates import ModelSpec, Surrogate

TIME_NORM = 80.0


def dose_rate(t, dose_times=(15.0, 45.0), amplitude: float = 1.2,
              duration: float = 0.6, edge_fraction: float = 0.25) -> np.ndarray:
    """Analytic dose-rate pulse train -- identical formula to the 3-D generator."""
    t = np.asarray(t, dtype=float)
    U = np.zeros_like(t)
    half = 0.5 * duration
    edge = max(edge_fraction * duration, 1e-9)
    for t0 in dose_times:
        loc = t - t0
        U += amplitude * (0.5 * (1.0 + np.tanh((loc + half) / edge))
                          * 0.5 * (1.0 + np.tanh((half - loc) / edge)))
    return U


# ---------------------------------------------------------------------------
# parameter initialisation
# ---------------------------------------------------------------------------
INIT_RANGES = {              # log-uniform sampling ranges for the restarts
    "rho": (0.005, 0.30), "K": (1.0, 8.0), "gamma": (0.05, 8.0),
    "mu": (1e-5, 0.05), "tau": (0.5, 10.0),
}
INIT_CENTRE = {"rho": 0.05, "K": 3.0, "gamma": 1.0, "mu": 1e-3, "tau": 3.0,
               "q": 0.85}


def build_init(CM: int, restarts: int, seed: int,
               warm: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
    """Per-batch initial mechanistic parameters, restart 0 = the canonical point."""
    rng = np.random.default_rng(seed)
    out: Dict[str, np.ndarray] = {}
    for k, (lo, hi) in INIT_RANGES.items():
        blocks = []
        for r in range(restarts):
            if r == 0:
                base = (np.full(CM, INIT_CENTRE[k]) if warm is None or k not in warm
                        else np.asarray(warm[k], dtype=float))
                blocks.append(base)
            else:
                jitter = np.exp(rng.uniform(np.log(lo), np.log(hi), size=CM))
                if warm is not None and k in warm:      # stay near the warm start
                    jitter = np.asarray(warm[k], float) * np.exp(
                        rng.normal(0.0, 0.5, size=CM))
                blocks.append(np.clip(jitter, lo, hi))
        out[k] = np.concatenate(blocks)
    q_blocks = []
    for r in range(restarts):
        if r == 0:
            q_blocks.append(np.full(CM, INIT_CENTRE["q"]))
        else:
            q_blocks.append(rng.uniform(*S.Q_RANGE, size=CM))
    out["q"] = np.concatenate(q_blocks)
    return out


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------
@dataclass
class FitResult:
    spec: ModelSpec
    y_pred: np.ndarray                  # (T_pred, CM) forecast per member
    train_mse: np.ndarray               # (CM,) best-restart training MSE
    theta: Dict[str, np.ndarray]        # (CM,) fitted mechanistic parameters
    lam_traj: Optional[np.ndarray]      # (T_pred, CM) realised gate value
    phys_share: Optional[np.ndarray]    # (T_pred, CM) realised physics fraction
    val_mse: np.ndarray                 # (CM,) held-out tail of the fit window
    fit_time_s: float
    loss_curve: List[float] = field(default_factory=list)
    extra: Dict = field(default_factory=dict)


def _resample(t_src: np.ndarray, Y: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Linear resampling of every column of ``Y`` from ``t_src`` onto ``t_dst``."""
    j = np.clip(np.searchsorted(t_src, t_dst), 1, len(t_src) - 1)
    w = ((t_dst - t_src[j - 1]) / (t_src[j] - t_src[j - 1]))[:, None]
    return Y[j - 1] * (1.0 - w) + Y[j] * w


def _clip_per_member(params, B: int, max_norm: float = 5.0):
    """Gradient clipping applied independently to each model in the batch.

    A global ``clip_grad_norm_`` would couple the ~28 000 independent models
    through one shared scale factor (and, at this batch size, throttle them all
    to nothing).  Every parameter carries a leading batch dimension, so the
    per-member norm and rescaling are exact.
    """
    device = params[0].device
    sq = torch.zeros(B, device=device)
    for p in params:
        if p.grad is not None:
            sq += p.grad.reshape(B, -1).pow(2).sum(dim=1)
    scale = (max_norm / (sq.sqrt() + 1e-9)).clamp(max=1.0)
    for p in params:
        if p.grad is not None:
            p.grad.mul_(scale.view([B] + [1] * (p.grad.dim() - 1)))


def _reg_block(states: torch.Tensor, t_fit: torch.Tensor, U_fit: torch.Tensor,
               B: int, dev, n_points: int = 48):
    """Sub-sample the trajectory into one (K, B, .) block for the regulariser."""
    stride = max(1, len(t_fit) // n_points)
    j = torch.arange(0, len(t_fit), stride, device=dev)
    st = states[j].detach()                                    # (K, B, 2)
    K = st.shape[0]
    tn = (t_fit[j] / TIME_NORM).view(K, 1).expand(K, B)
    Ub = U_fit[j].view(K, 1).expand(K, B)
    return st, tn, Ub


def _phase_groups(model: Surrogate, spec: ModelSpec):
    """(name, params, epochs, lr) schedule; a single joint phase by default."""
    sched = spec.meta.get("phases")
    if not sched:
        return [("joint", list(model.parameters()), spec.epochs, spec.lr)]
    out = []
    for name, ep, mult in sched:
        if name == "closure":
            ps = model.closure_parameters()
        elif name == "physics":
            ps = model.physics_parameters()
        else:
            ps = list(model.parameters())
        if ps and ep > 0:
            out.append((name, ps, ep, spec.lr * mult))
    return out


def fit(spec: ModelSpec, bench: Bench, device: str = "cuda",
        seed: int = 1234, warm: Optional[Dict[str, np.ndarray]] = None,
        log=lambda *_: None) -> FitResult:
    dev = torch.device(device)
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    t0 = time.time()

    CM, R = bench.CM, max(1, spec.restarts)
    B = CM * R

    dose_times = spec.meta.get("dose_times", (15.0, 45.0))
    grid_np = S.refined_grid(float(bench.t_obs[0]), float(bench.t_obs[-1]),
                             dose_times, must_include=bench.t_obs)
    obs_idx = S.obs_indices(grid_np, bench.t_obs)
    pred_np = S.refined_grid(float(bench.t_pred[0]), float(bench.t_pred[-1]),
                             dose_times, must_include=bench.t_obs)
    # Analytic dose on both grids: interpolating a 0.6-wide pulse off the 0.1
    # output grid would distort the only channel the RT physics sees.
    U_fit_np = dose_rate(grid_np, dose_times, **spec.meta.get("dose", {}))
    U_pred_np = dose_rate(pred_np, dose_times, **spec.meta.get("dose", {}))
    sig = S.sigma_reference(bench.t_obs, bench.Y_obs)

    t_fit_d = torch.tensor(grid_np, dtype=S.DTYPE, device=dev)
    U_fit = torch.tensor(U_fit_np, dtype=S.DTYPE, device=dev)
    Y = torch.tensor(np.tile(bench.Y_obs, (1, R)), dtype=S.DTYPE, device=dev)
    y0 = Y[0].clone()
    oidx = torch.tensor(obs_idx, dtype=torch.long, device=dev)

    init = build_init(CM, R, seed, warm)
    model = Surrogate(spec, B, dev, sigma_ref=np.tile(sig, R), init=init,
                      gen=gen).to(dev)
    if spec.w_theta_anchor > 0 and warm:
        model.set_anchor({k: np.tile(v, R) for k, v in warm.items()}, dev)
    if spec.gate_context:
        # One scalar per column: the mass-weighted dose coverage at t = 0, which
        # a treatment plan supplies. It tells the GATE which regime it is in; the
        # closure never sees it, so the blend stays identifiable.
        ctx = np.repeat(bench.context(), bench.n_members)
        with torch.no_grad():
            model.context.copy_(torch.tensor(np.tile(ctx, R), dtype=S.DTYPE,
                                             device=dev))

    # optional inner split: fit on the early part of the window, score the tail.
    val_end = spec.meta.get("val_end")
    if val_end:
        fit_mask = bench.t_obs <= float(val_end) + 1e-9
    else:
        fit_mask = np.ones(len(bench.t_obs), dtype=bool)
    fit_sel = torch.tensor(np.where(fit_mask)[0], dtype=torch.long, device=dev)
    val_sel = torch.tensor(np.where(~fit_mask)[0], dtype=torch.long, device=dev)

    best = torch.full((B,), float("inf"), device=dev)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    curve: List[float] = []
    orth_frac = 0.0

    for name, params, epochs, lr in _phase_groups(model, spec):
        for p in model.parameters():
            p.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)
        opt = torch.optim.Adam(params, lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                           eta_min=lr * 0.05)
        for ep in range(epochs):
            opt.zero_grad(set_to_none=True)
            states = S.integrate(model, grid_np, U_fit, y0, TIME_NORM)
            pred = states[oidx, :, 0]                          # (N_fit, B)
            resid = (pred - Y) ** 2
            per = resid[fit_sel].mean(dim=0)                    # (B,) fitted part
            loss = per.mean()

            reg = torch.zeros((), device=dev)
            need_block = model.closure is not None and (
                spec.l2_closure > 0 or spec.gate_penalty > 0 or spec.w_orth > 0)
            if need_block:
                st, tn, Ub = _reg_block(states, t_fit_d, U_fit, B, dev)
                if spec.l2_closure > 0:
                    reg = reg + spec.l2_closure * (model.closure_raw(st, tn, Ub) ** 2).mean()
                if spec.blend == "gated" and spec.gate_penalty > 0:
                    reg = reg + spec.gate_penalty * model.lam_of(
                        st[..., 0], st[..., 1], tn, Ub).mean()
                if spec.w_orth > 0:
                    o, frac = model.orthogonality_loss(st, tn, Ub, model.begin())
                    reg = reg + spec.w_orth * o
                    orth_frac = float(frac)
            if spec.w_theta_anchor > 0:
                reg = reg + spec.w_theta_anchor * model.anchor_loss(model.begin())

            (loss + reg).backward()
            _clip_per_member(params, B, max_norm=5.0)

            # Snapshot the parameters that PRODUCED this loss, before stepping.
            with torch.no_grad():
                imp = per < best
                if bool(imp.any()):
                    best = torch.where(imp, per.detach(), best)
                    for k, v in model.state_dict().items():
                        best_state[k][imp] = v[imp].clone()
            opt.step()
            sched.step()
            if ep % 25 == 0:
                curve.append(float(loss.detach()))
        log(f"    {spec.name} phase={name} ep={epochs} lr={lr:.1e} "
            f"mse={float(per.mean()):.3e}")

    model.load_state_dict(best_state)

    # ---- pick the best restart per (case, member) --------------------------
    # With an inner split, restarts are selected on the HELD-OUT tail of the
    # assimilation window rather than on the data they were fitted to.  A
    # flexible closure can always drive the fitted residual down; only the
    # held-out tail says whether that helped or was memorisation.
    val_all = None
    if val_sel.numel():
        with torch.no_grad():
            st = S.integrate(model, grid_np, U_fit, y0, TIME_NORM)
            val_all = ((st[oidx, :, 0] - Y) ** 2)[val_sel].mean(dim=0)   # (B,)
    best_cm = best.view(R, CM)
    criterion = val_all.view(R, CM) if val_all is not None else best_cm
    pick = torch.argmin(criterion, dim=0)                   # (CM,)
    sel = pick * CM + torch.arange(CM, device=dev)
    val_mse = (val_all[sel].cpu().numpy() if val_all is not None
               else np.full(CM, np.nan))

    # ---- forecast ----------------------------------------------------------
    with torch.no_grad():
        t_pred_d = torch.tensor(pred_np, dtype=S.DTYPE, device=dev)
        U_pred = torch.tensor(U_pred_np, dtype=S.DTYPE, device=dev)
        states = S.integrate(model, pred_np, U_pred, y0, TIME_NORM)
        ctx = model.begin()

        Tg = len(pred_np)
        lam_g = np.zeros((Tg, CM), dtype=np.float64)
        share_g = np.zeros((Tg, CM), dtype=np.float64)
        for a0 in range(0, Tg, 64):
            a1 = min(a0 + 64, Tg)
            k = a1 - a0
            tn = (t_pred_d[a0:a1] / TIME_NORM).view(k, 1).expand(k, B)
            Ub = U_pred[a0:a1].view(k, 1).expand(k, B)
            dp, dm, _, lam = model.parts(states[a0:a1], tn, Ub, ctx)
            pa, pb = dp.abs()[:, sel], dm.abs()[:, sel]
            share_g[a0:a1] = (pa / (pa + pb + 1e-12)).cpu().numpy()
            lam_g[a0:a1] = lam[:, sel].cpu().numpy()

        # resample onto the dense scoring grid shared with the ground truth
        yg = states[:, sel, 0].cpu().numpy()
        y_pred = _resample(pred_np, yg, bench.t_pred)
        lam_t = _resample(pred_np, lam_g, bench.t_pred)
        share = _resample(pred_np, share_g, bench.t_pred)
        th = {k: v[sel].cpu().numpy() for k, v in model.backbone.theta().items()}

    return FitResult(spec=spec, y_pred=y_pred,
                     train_mse=best_cm.gather(0, pick.unsqueeze(0))[0].cpu().numpy(),
                     theta=th, lam_traj=lam_t, phys_share=share,
                     val_mse=val_mse,
                     fit_time_s=time.time() - t0, loss_curve=curve,
                     extra={"sigma_ref": sig, "restarts": R,
                            "orth_frac": orth_frac,
                            "grid_points_fit": int(len(grid_np)),
                            "grid_points_pred": int(len(pred_np)),
                            "clamp_fraction": float((yg <= 1e-9).mean())})


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def score(res: FitResult, bench: Bench, train_end: float = 35.0) -> Dict[str, np.ndarray]:
    """Per-(case, member) errors, and per-case medians over the members."""
    t = bench.t_pred
    tr = t <= train_end
    te = t > train_end
    y_true = bench.y_true[:, bench.case_of()]              # (T, CM)
    yp = res.y_pred
    fin = np.isfinite(yp).all(axis=0)

    def rel(mask):
        num = np.sqrt(np.mean((yp[mask] - y_true[mask]) ** 2, axis=0))
        den = np.sqrt(np.mean(y_true[mask] ** 2, axis=0)) + 1e-12
        return 100.0 * num / den

    out = {
        "train_rel_rmse": rel(tr),
        "test_rel_rmse": rel(te),
        "final_err": 100.0 * (yp[-1] - y_true[-1]) / (np.abs(y_true[-1]) + 1e-12),
        "nadir_err": 100.0 * (yp[tr | te].min(axis=0) - y_true.min(axis=0))
        / (np.abs(y_true.min(axis=0)) + 1e-12),
        "finite": fin.astype(float),
    }
    out["abs_final_err"] = np.abs(out["final_err"])
    for k in list(out):
        out[k] = np.where(fin, out[k], np.nan)
    return out


def per_case(metric: np.ndarray, bench: Bench, agg=np.nanmedian) -> np.ndarray:
    """Aggregate a per-member metric to one number per (patient, config)."""
    return agg(metric.reshape(bench.C, bench.n_members), axis=1)
