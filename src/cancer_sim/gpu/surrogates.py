"""Reduced 0-D tumour surrogates, all batched on one GPU tensor.

Every model in this module predicts the same observable -- the cumulative
(spatially integrated) tumour mass ``y(t)`` -- from sparse noisy observations in
an assimilation window, and is scored on a held-out forecast window.  All of
them share one batched RK4 integrator, one optimiser and one time grid, so the
comparison isolates the *model*, not the fitting machinery.

The batch axis ``B`` carries (patient x beam-configuration x noisy-realisation x
random restart) simultaneously: every one of those is an independent model with
its own parameters, evaluated in one set of kernels.

Model families
--------------
``mech``    purely mechanistic two-state model, paper Eq. (ode_rt_2state),
            optionally with a trainable growth exponent (see `Backbone`).
``node``    purely data-driven Neural ODE, paper Eq. (node).
``scaled``  the paper's PI-NODE, Eq. (pinode_weighted):
                dy/dt = omega * f_RT + s_r * g_psi
``convex``  the proposed replacement, a true partition of unity:
                dy/dt = (1 - lam) * f_RT + lam * sigma * tanh(g_psi)
``gated``   the same, with ``lam`` a learned function of the state:
                lam = sigmoid(h_xi(y, z, t, U))

Why ``convex``/``gated`` and not ``scaled``
-------------------------------------------
``omega`` can be absorbed *exactly* into ``(rho, gamma, mu)`` because ``f_RT`` is
homogeneous of degree one in them (``K`` enters only as ``rho/K``, so it must not
be rescaled), and ``s_r`` can be absorbed exactly into the closure's affine
output layer.  Both reachable sets are cones, so both weights are gauges, not
mixing weights -- which is exactly why the two numbers cannot be made to sum to
one.  Verified to machine precision: the pairs ``(omega, theta)`` and
``(omega/c, (c rho, c gamma, K, c mu, tau))`` give identical trajectories.

Making the weights convex is *not by itself* enough -- a cone stays a cone.
Three things are needed together, and this module implements all three:

1. **bounded closure** ``sigma_ref * tanh(g)`` -- turns the ML reachable set from
   a cone into a ball, so ``lam`` really does cap the closure's authority;
2. **anchored physics** ``w_theta * ||log(theta / theta_ref)||^2`` with
   ``theta_ref`` the assimilated mechanistic fit -- stops ``(1 - lam)`` from being
   absorbed back into ``theta``, so ``lam = 0`` is *the fitted physics* and
   intermediate ``lam`` is a genuine interpolation away from it;
3. **orthogonal closure** ``w_orth * ||P_{span(df/dtheta)} tanh(g)||^2`` -- stops
   the closure from doing anything a parameter change could have done, which is
   what makes the physics/ML *ratio* identified.

``lam`` is the control; the *realised* physics share
``Phi = |physics| / (|physics| + |ML|)`` is measured along the trajectory and
reported alongside it, so the calibration of the control can be checked rather
than assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DTYPE = torch.float32
EPS = 1e-6
Y_FLOOR = 1e-4          # keeps y^q differentiable at the origin


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _inv_softplus(x):
    x = np.maximum(np.asarray(x, dtype=np.float64), 1e-9)
    return np.log(np.expm1(np.minimum(x, 30.0)) + 1e-300)


def _param(values, B: int, device, offset: float = 0.0) -> nn.Parameter:
    """Raw parameter such that ``softplus(raw) + offset`` equals ``values``.

    The offset is the positivity floor the transform adds back, so the
    requested value is realised exactly rather than shifted by the floor.
    """
    v = np.broadcast_to(np.asarray(values, dtype=np.float64), (B,)) - offset
    return nn.Parameter(torch.tensor(_inv_softplus(v), dtype=DTYPE, device=device))


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


class BatchedMLP(nn.Module):
    """``B`` independent 2-hidden-layer MLPs evaluated in one einsum chain."""

    def __init__(self, B: int, n_in: int, n_out: int, hidden: int, device,
                 activation: str = "relu", zero_out: bool = True, gen=None):
        super().__init__()
        self.act = torch.relu if activation == "relu" else torch.tanh
        def u(*shape, bound):
            t = torch.empty(*shape, dtype=DTYPE, device=device)
            return nn.Parameter(t.uniform_(-bound, bound, generator=gen))
        b_in, b_h = 1.0 / math.sqrt(n_in), 1.0 / math.sqrt(hidden)
        self.w1 = u(B, hidden, n_in, bound=b_in)
        self.b1 = u(B, hidden, bound=b_in)
        self.w2 = u(B, hidden, hidden, bound=b_h)
        self.b2 = u(B, hidden, bound=b_h)
        if zero_out:
            self.w3 = nn.Parameter(torch.zeros(B, n_out, hidden, dtype=DTYPE, device=device))
            self.b3 = nn.Parameter(torch.zeros(B, n_out, dtype=DTYPE, device=device))
        else:
            self.w3 = u(B, n_out, hidden, bound=b_h)
            self.b3 = u(B, n_out, bound=b_h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., B, n_in) -> (..., B, n_out); the leading dims let a whole
        # block of time points be evaluated in one call (used by the regulariser)
        h = self.act(torch.einsum("boi,...bi->...bo", self.w1, x) + self.b1)
        h = self.act(torch.einsum("boi,...bi->...bo", self.w2, h) + self.b2)
        return torch.einsum("boi,...bi->...bo", self.w3, h) + self.b3


# ---------------------------------------------------------------------------
# mechanistic backbone
# ---------------------------------------------------------------------------
BACKBONES = ("logistic", "volumetric", "power")
Q_RANGE = (0.30, 1.20)

# Positivity floors of the softplus transforms.  `K_FLOOR` keeps the carrying
# capacity above the observed burden so the logistic term cannot invert; the
# rest are numerical guards only.  The upper end of K is deliberately left
# unbounded: the earlier code capped it at 5 while the least-squares optimum on
# this data is far higher, which manufactured "model inadequacy".
RHO_FLOOR = 1e-6
K_FLOOR = 0.20
GAMMA_FLOOR = 1e-6
MU_FLOOR = 1e-9
TAU_FLOOR = 0.05


class Backbone(nn.Module):
    """Interpretable two-state radiotherapy core.

        dy/dt = rho * y^q * (1 - y/K) - gamma * z * y - mu * y
        dz/dt = -z / tau + U(t)

    ``q = 1`` is the paper's logistic law (Eq. ode_rt_2state).  ``q = 2/3`` is
    the law obtained by spatially integrating a constant-speed Fisher--KPP
    invasion front (mass ~ (a + bt)^3 => dm/dt ~ m^(2/3)), which is what the 3-D
    ground truth actually does.  ``kind="power"`` makes ``q`` trainable so the
    data can choose, and the fitted value is itself a reportable result.
    """

    def __init__(self, B: int, device, kind: str = "logistic",
                 init: Optional[Dict[str, np.ndarray]] = None, gen=None):
        super().__init__()
        if kind not in BACKBONES:
            raise ValueError(kind)
        self.kind = kind
        self.B = B
        d = init or {}
        self.raw_rho = _param(d.get("rho", 0.05), B, device, RHO_FLOOR)
        self.raw_K = _param(d.get("K", 3.0), B, device, K_FLOOR)
        self.raw_gamma = _param(d.get("gamma", 1.0), B, device, GAMMA_FLOOR)
        self.raw_mu = _param(d.get("mu", 1e-3), B, device, MU_FLOOR)
        self.raw_tau = _param(d.get("tau", 3.0), B, device, TAU_FLOOR)
        if kind == "power":
            q0 = np.broadcast_to(np.asarray(d.get("q", 0.8), dtype=np.float64), (B,))
            p = (np.clip(q0, *Q_RANGE) - Q_RANGE[0]) / (Q_RANGE[1] - Q_RANGE[0])
            self.raw_q = nn.Parameter(torch.tensor(
                np.log(p / (1 - p + 1e-12) + 1e-12), dtype=DTYPE, device=device))
        else:
            self.register_buffer("_q_fixed", torch.full(
                (B,), 1.0 if kind == "logistic" else 2.0 / 3.0,
                dtype=DTYPE, device=device))

    def q(self) -> torch.Tensor:
        if self.kind == "power":
            lo, hi = Q_RANGE
            return lo + (hi - lo) * torch.sigmoid(self.raw_q)
        return self._q_fixed

    def theta(self) -> Dict[str, torch.Tensor]:
        return {
            "rho": F.softplus(self.raw_rho) + RHO_FLOOR,
            "K": F.softplus(self.raw_K) + K_FLOOR,
            "gamma": F.softplus(self.raw_gamma) + GAMMA_FLOOR,
            "mu": F.softplus(self.raw_mu) + MU_FLOOR,
            "tau": F.softplus(self.raw_tau) + TAU_FLOOR,
            "q": self.q(),
        }

    def f_y(self, y, z, U, th=None):
        th = th or self.theta()
        yy = torch.clamp(y, min=Y_FLOOR)
        growth = th["rho"] * yy.pow(th["q"]) * (1.0 - y / th["K"])
        return growth - th["gamma"] * z * y - th["mu"] * y

    def f_z(self, y, z, U, th=None):
        th = th or self.theta()
        return -z / th["tau"] + U

    def sensitivities(self, y, z, th) -> torch.Tensor:
        """``d f_y / d theta`` along the trajectory -- stacked as ``(..., n)``.

        Spanning directions of "what a change of mechanistic parameters could
        have done".  A closure with a component in this span is not modelling
        unresolved physics, it is silently re-estimating theta -- which is the
        very failure the paper attributes to reduced ODE surrogates ("unresolved
        spatial effects are absorbed into distorted effective parameters").
        """
        yy = torch.clamp(y, min=Y_FLOOR)
        yq = yy.pow(th["q"])
        sat = 1.0 - y / th["K"]
        cols = [yq * sat,                              # d/d rho
                th["rho"] * yq * y / th["K"] ** 2,     # d/d K
                -z * y,                                # d/d gamma
                -y]                                    # d/d mu
        if self.kind == "power":
            cols.append(th["rho"] * yq * torch.log(yy) * sat)   # d/d q
        return torch.stack(cols, dim=-1)


# ---------------------------------------------------------------------------
# the surrogate: backbone + optional closure, with the blending rule
# ---------------------------------------------------------------------------
BLENDS = ("none", "scaled", "convex", "gated")


@dataclass
class ModelSpec:
    """Everything that defines one surrogate variant."""

    name: str
    family: str = "phys"          # "phys" | "node"
    backbone: str = "logistic"
    blend: str = "none"
    omega: float = 0.05           # `scaled` only (paper PI-NODE)
    s_r: float = 0.10             # `scaled` only
    lam: float = 0.30             # `convex` only
    lam_init: float = 0.20        # `gated` only: prior/initial gate level
    gate_penalty: float = 0.0     # `gated` only: eta * mean(lam)
    hidden: int = 32
    gate_hidden: int = 16
    activation: str = "relu"
    # Drop the explicit time input from the closure.  A closure that sees ``t``
    # can fit a time trend inside the assimilation window and then continue it
    # linearly outside -- the classic neural-ODE drift, and the most likely
    # reason a *small* correction extrapolates worse than none at all.
    autonomous: bool = False
    # Give the gate a per-column regime signal the reduced state cannot carry
    # (the mass-weighted dose coverage at t=0, one scalar a treatment plan
    # supplies for free).  The closure itself never sees it, so the physics/ML
    # blend stays identifiable -- only the *schedule* becomes observable.
    gate_context: bool = False
    # `scaled` only: multiply BOTH weights by the patient's own rate scale
    # sigma_ref, so (omega, s_r) are dimensionless ratios rather than absolute
    # rates.  Neither reachable set changes -- both weights stay exact gauges --
    # but the branches start commensurate with THIS patient's dynamics instead
    # of the cohort median, which is what the fixed global pair silently assumes.
    scale_by_sigma: bool = False
    l2_closure: float = 1e-6
    # Identifiability of the blend (see Surrogate docstring):
    w_theta_anchor: float = 0.0   # w * ||log(theta / theta_ref)||^2
    w_orth: float = 0.0           # penalty on the closure's component in span(df/dtheta)
    restarts: int = 3
    epochs: int = 1200
    lr: float = 1.5e-2            # cosine-annealed to 5 % of this
    warm_start: bool = False      # initialise theta from a previous mech fit
    meta: Dict = field(default_factory=dict)


class Surrogate(nn.Module):
    """Batched reduced surrogate; see the module docstring for the families."""

    def __init__(self, spec: ModelSpec, B: int, device,
                 sigma_ref: Optional[np.ndarray] = None,
                 init: Optional[Dict[str, np.ndarray]] = None, gen=None):
        super().__init__()
        self.spec = spec
        self.B = B
        self.backbone = Backbone(B, device, spec.backbone, init, gen=gen)
        sig = np.broadcast_to(np.asarray(
            sigma_ref if sigma_ref is not None else 0.05, dtype=np.float64), (B,))
        self.register_buffer("sigma", torch.tensor(sig, dtype=DTYPE, device=device))

        self.anchor: Optional[Dict[str, torch.Tensor]] = None
        self.register_buffer("context", torch.zeros(B, dtype=DTYPE, device=device))
        self.closure = None
        self.gate = None
        self.n_in = 3 if spec.autonomous else 4
        if spec.family == "node":
            self.closure = BatchedMLP(B, self.n_in, 2, spec.hidden, device,
                                      spec.activation, zero_out=True, gen=gen)
            self.raw_alpha = _param(1.0, B, device)
        elif spec.blend != "none":
            self.closure = BatchedMLP(B, self.n_in, 1, spec.hidden, device,
                                      spec.activation, zero_out=True, gen=gen)
            if spec.blend == "gated":
                self.gate = BatchedMLP(B, self.n_in + int(spec.gate_context), 1,
                                       spec.gate_hidden, device, "tanh",
                                       zero_out=True, gen=gen)
                self.gate_bias = nn.Parameter(torch.full(
                    (B,), _logit(spec.lam_init), dtype=DTYPE, device=device))

    # -- pieces ------------------------------------------------------------
    def _feat(self, y, z, t_norm, U):
        Ue = U.expand_as(y)
        if self.spec.autonomous:
            return torch.stack([y, z, Ue], dim=-1)
        tn = (t_norm.expand_as(y) if torch.is_tensor(t_norm)
              else torch.full_like(y, float(t_norm)))
        return torch.stack([y, z, tn, Ue], dim=-1)

    def lam_of(self, y, z, t_norm, U) -> torch.Tensor:
        if self.spec.blend == "gated":
            f = self._feat(y, z, t_norm, U)
            if self.spec.gate_context:
                f = torch.cat([f, self.context.expand_as(y).unsqueeze(-1)], -1)
            return torch.sigmoid(self.gate(f)[..., 0] + self.gate_bias)
        if self.spec.blend == "convex":
            return torch.full_like(y, float(self.spec.lam))
        return torch.zeros_like(y)

    def begin(self) -> Dict[str, torch.Tensor]:
        """Parameter transforms hoisted out of the RK4 loop (state-independent)."""
        ctx = dict(self.backbone.theta())
        if self.spec.family == "node":
            ctx["alpha"] = F.softplus(self.raw_alpha) + EPS
        return ctx

    def parts(self, s: torch.Tensor, t_norm, U: torch.Tensor, ctx=None):
        """Return ``(dy_physics, dy_ml, dz, lam)`` -- the blend diagnostics."""
        y, z = s[..., 0], s[..., 1]
        sp = self.spec
        th = ctx if ctx is not None else self.begin()
        if sp.family == "node":
            out = self.closure(self._feat(y, z, t_norm, U))
            kill = F.softplus(out[..., 1])
            dy_ml = out[..., 0] - kill * z * y
            dz = th["alpha"] * U - z / th["tau"]
            return torch.zeros_like(y), dy_ml, dz, torch.ones_like(y)

        f_y = self.backbone.f_y(y, z, U, th)
        dz = self.backbone.f_z(y, z, U, th)
        if sp.blend == "none":
            return f_y, torch.zeros_like(y), dz, torch.zeros_like(y)

        g = self.closure(self._feat(y, z, t_norm, U))[..., 0]
        if sp.blend == "scaled":                      # paper Eq. pinode_weighted
            sc = self.sigma if sp.scale_by_sigma else 1.0
            return sp.omega * sc * f_y, sp.s_r * sc * g, dz, torch.zeros_like(y)

        lam = self.lam_of(y, z, t_norm, U)            # convex / gated
        return (1.0 - lam) * f_y, lam * self.sigma * torch.tanh(g), dz, lam

    def rhs(self, s: torch.Tensor, t_norm, U: torch.Tensor, ctx=None) -> torch.Tensor:
        dy_p, dy_m, dz, _ = self.parts(s, t_norm, U, ctx)
        return torch.stack([dy_p + dy_m, dz], dim=-1)

    def closure_raw(self, s, t_norm, U) -> Optional[torch.Tensor]:
        if self.closure is None:
            return None
        return self.closure(self._feat(s[..., 0], s[..., 1], t_norm, U))

    # -- identifiability terms ---------------------------------------------
    def set_anchor(self, ref: Dict[str, np.ndarray], device):
        """Store ``theta_ref`` for the log-prior that pins the physics amplitude."""
        self.anchor = {k: torch.tensor(np.asarray(v, dtype=np.float64),
                                       dtype=DTYPE, device=device)
                       for k, v in ref.items() if k in ("rho", "K", "gamma", "mu", "tau")}

    def anchor_loss(self, th: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.anchor:
            return torch.zeros((), device=self.sigma.device)
        acc = 0.0
        for k, ref in self.anchor.items():
            acc = acc + (torch.log(th[k] / (ref + EPS)) ** 2).mean()
        return acc / max(len(self.anchor), 1)

    def orthogonality_loss(self, states: torch.Tensor, tn, U: torch.Tensor,
                           th: Dict[str, torch.Tensor], delta: float = 1e-6):
        """Squared norm of the closure's projection onto ``span(df/dtheta)``.

        ``states`` is a ``(K, B, 2)`` block sampled along the trajectory.  The
        least-squares projection is solved per batch member on an ``(n, n)``
        normal-equation system, ``n <= 5``.
        """
        if self.closure is None:
            return torch.zeros((), device=states.device), torch.zeros((), device=states.device)
        y, z = states[..., 0], states[..., 1]
        c = torch.tanh(self.closure(self._feat(y, z, tn, U))[..., 0])   # (K,B)
        J = self.backbone.sensitivities(y, z, th)                        # (K,B,n)
        # normalise columns so delta is scale-free
        J = J / (J.pow(2).mean(dim=0, keepdim=True).sqrt() + 1e-9)
        Jb = J.permute(1, 0, 2)                                          # (B,K,n)
        cb = c.permute(1, 0).unsqueeze(-1)                               # (B,K,1)
        JtJ = Jb.transpose(1, 2) @ Jb
        eye = torch.eye(JtJ.shape[-1], device=states.device).expand_as(JtJ)
        coef = torch.linalg.solve(JtJ + delta * JtJ.shape[1] * eye,
                                  Jb.transpose(1, 2) @ cb)               # (B,n,1)
        proj = (Jb @ coef).squeeze(-1)                                   # (B,K)
        num = proj.pow(2).mean()
        frac = num / (c.pow(2).mean() + 1e-12)
        return num, frac.detach()

    # -- parameter groups --------------------------------------------------
    def physics_parameters(self):
        return [p for p in self.backbone.parameters()]

    def closure_parameters(self):
        ps = list(self.closure.parameters()) if self.closure is not None else []
        if self.gate is not None:
            ps += list(self.gate.parameters()) + [self.gate_bias]
        if hasattr(self, "raw_alpha"):
            ps.append(self.raw_alpha)
        return ps


# ---------------------------------------------------------------------------
# batched RK4 on a fixed grid (decoupled from the observation times)
# ---------------------------------------------------------------------------
def integrate(model: Surrogate, t_host: np.ndarray, U: torch.Tensor,
              y0: torch.Tensor, time_norm: float,
              z0: Optional[torch.Tensor] = None) -> torch.Tensor:
    """RK4 over a shared grid for all ``B`` models. Returns ``(T, B, 2)``.

    ``t_host`` is a NumPy array so the step size and the normalised time enter
    as Python floats: reading them off a CUDA tensor inside the loop would force
    a device synchronisation on every one of the ~230 steps and dominates the
    run time.  ``U`` stays on the device.

    The grid is a *fixed integration* grid, never the sparse observation grid,
    so the fitted parameters do not depend on how densely the tumour happens to
    have been measured.
    """
    z = torch.zeros_like(y0) if z0 is None else z0
    s = torch.stack([y0, z], dim=-1)
    ctx = model.begin()
    Umid = 0.5 * (U[:-1] + U[1:])
    inv = 1.0 / time_norm
    out = [s]
    for i in range(1, len(t_host)):
        dt = float(t_host[i] - t_host[i - 1])
        tn0 = float(t_host[i - 1]) * inv
        tn2 = float(t_host[i]) * inv
        tn1 = 0.5 * (tn0 + tn2)
        U0, U1, U2 = U[i - 1], Umid[i - 1], U[i]
        k1 = model.rhs(s, tn0, U0, ctx)
        k2 = model.rhs(torch.clamp(s + (0.5 * dt) * k1, min=0.0), tn1, U1, ctx)
        k3 = model.rhs(torch.clamp(s + (0.5 * dt) * k2, min=0.0), tn1, U1, ctx)
        k4 = model.rhs(torch.clamp(s + dt * k3, min=0.0), tn2, U2, ctx)
        s = torch.clamp(s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), min=0.0)
        out.append(s)
    return torch.stack(out)


def refined_grid(t0: float, t1: float, dose_times: Sequence[float],
                 must_include: Sequence[float] = (),
                 dt_pulse: float = 0.05, pulse_halfwidth: float = 0.6,
                 dt_response: float = 0.10, response_length: float = 12.0,
                 dt_coarse: float = 0.50) -> np.ndarray:
    """Integration grid refined where the dynamics are fast.

    Three resolutions: inside a dose pulse, during the damage-decay response
    that follows it, and everywhere else (where the reduced dynamics are smooth
    and RK4 at ``dt_coarse`` is accurate to ~1e-8 per step).  The grid is a pure
    function of the dose schedule and the observation times, so it is byte-for-
    byte identical for every model in the comparison.
    """
    pts = [np.arange(t0, t1 + dt_coarse, dt_coarse), np.array([t0, t1]),
           np.asarray(must_include, dtype=float)]
    for td in dose_times:
        pts.append(np.arange(td - pulse_halfwidth, td + pulse_halfwidth + dt_pulse,
                             dt_pulse))
        pts.append(np.arange(td, td + response_length + dt_response, dt_response))
    g = np.concatenate([p for p in pts if p.size])
    g = g[(g >= t0 - 1e-9) & (g <= t1 + 1e-9)]
    g = np.unique(np.round(g, 6))
    return np.clip(g, t0, t1)


def obs_indices(grid: np.ndarray, t_obs: np.ndarray) -> np.ndarray:
    """Exact index of every observation time on ``grid`` (asserts alignment)."""
    idx = np.array([int(np.argmin(np.abs(grid - tt))) for tt in t_obs])
    assert np.max(np.abs(grid[idx] - np.asarray(t_obs))) < 1e-5, \
        "observation times are not on the integration grid"
    return idx


def sigma_reference(t_obs: np.ndarray, Y: np.ndarray, factor: float = 1.0,
                    floor: float = 1e-3, window: int = 7) -> np.ndarray:
    """Characteristic rate scale of each observed trajectory.

    Sets the magnitude of the bounded ML branch so ``lam`` interpolates between
    two vector fields of comparable size, and so ``lam`` means the same thing
    from patient to patient.

    The estimator is the RMS derivative of a Savitzky--Golay smooth of the noisy
    observations.  Differencing the raw samples would be noise-dominated here
    (2 % noise over a 1.1-unit spacing gives a spurious rate comparable to the
    true one), and a plain range-over-span estimate is measurably less
    consistent across patients: measured against the true derivative RMS on the
    cohort, this estimator lands at 1.45x with a p10--p90 spread of 1.05--1.55,
    versus 1.91x and 1.36--2.09 for range-over-span.

    Derived only from the assimilation data, so it is available at fit time and
    identical for every model that uses it.
    """
    t = np.asarray(t_obs, dtype=float)
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    if Y.shape[0] != len(t):
        Y = Y.T
    if len(t) >= window + 2:
        from scipy.signal import savgol_filter
        Ys = savgol_filter(Y, window, 2, axis=0)
    else:
        Ys = Y
    d = np.diff(Ys, axis=0) / np.diff(t)[:, None]
    rms = np.sqrt(np.mean(d ** 2, axis=0))
    return np.maximum(factor * rms, floor)
