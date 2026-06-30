"""Neural surrogates (NODE / PI-NODE) with a batched RK4 integrator.

Performance note
----------------
The original code trained each of the ``E`` noisy ensemble members in a
separate Python loop, and inside each it integrated the ODE one scalar RK4
step at a time.  Here every ensemble member is an *independent* model whose
parameters carry a leading ensemble dimension ``E``; a single vectorised RK4
integrates all members simultaneously.  This collapses ``E`` outer iterations
into one and turns the per-step scalar ops into ``(E, ...)`` tensor ops, which
is the main training speed-up (and runs on MPS/CUDA when available).

Mathematics is unchanged from the paper:

PI-NODE (Eq. ``pinode_weighted``):
    dy/dt = omega * [rho*y*(1-y/Keff) - gamma*z*y - mu*y] + s_r * g_psi(y,z,t,U)
    dz/dt = -z/tau + U                       (latent damage kept mechanistic)

NODE (Eq. ``node``): both derivatives produced by a neural field, with a
sign-constrained kill term so radiation can never increase tumour mass.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DTYPE = torch.float32


def get_device(prefer: str = "auto") -> torch.device:
    if prefer not in ("auto", "cpu"):
        return torch.device(prefer)
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def inv_softplus(x: float) -> float:
    x = max(float(x), 1e-6)
    return math.log(math.expm1(x))


class BatchedMLP(nn.Module):
    """``E`` independent 2-hidden-layer ReLU MLPs evaluated in parallel.

    Input  (E, n_in) -> output (E, n_out).  The final layer is zero-initialised
    so the residual starts at zero (paper: "residual output initialised close
    to zero").
    """

    def __init__(self, ensemble: int, n_in: int, n_out: int, hidden: int = 32,
                 activation: str = "relu"):
        super().__init__()
        self.E = ensemble
        self.act = torch.relu if activation == "relu" else torch.tanh
        self.w1 = nn.Parameter(torch.empty(ensemble, hidden, n_in, dtype=DTYPE))
        self.b1 = nn.Parameter(torch.empty(ensemble, hidden, dtype=DTYPE))
        self.w2 = nn.Parameter(torch.empty(ensemble, hidden, hidden, dtype=DTYPE))
        self.b2 = nn.Parameter(torch.empty(ensemble, hidden, dtype=DTYPE))
        self.w3 = nn.Parameter(torch.zeros(ensemble, n_out, hidden, dtype=DTYPE))
        self.b3 = nn.Parameter(torch.zeros(ensemble, n_out, dtype=DTYPE))
        self._reset(n_in, hidden)

    def _reset(self, n_in: int, hidden: int):
        # Default nn.Linear-style uniform init for the hidden layers.
        b_in = 1.0 / math.sqrt(n_in)
        b_h = 1.0 / math.sqrt(hidden)
        nn.init.uniform_(self.w1, -b_in, b_in)
        nn.init.uniform_(self.b1, -b_in, b_in)
        nn.init.uniform_(self.w2, -b_h, b_h)
        nn.init.uniform_(self.b2, -b_h, b_h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (E, n_in) or (..., E, n_in) -> matching (..., E, n_out)
        h = self.act(torch.einsum("eoi,...ei->...eo", self.w1, x) + self.b1)
        h = self.act(torch.einsum("eoi,...ei->...eo", self.w2, h) + self.b2)
        return torch.einsum("eoi,...ei->...eo", self.w3, h) + self.b3


def _raw_param(ensemble: int, value: float) -> nn.Parameter:
    return nn.Parameter(torch.full((ensemble,), inv_softplus(value), dtype=DTYPE))


def _raw_from_array(values, offset: float = 0.0) -> nn.Parameter:
    """Per-member raw parameter so that softplus(raw)+offset == values."""
    arr = np.maximum(np.asarray(values, dtype=float) - offset, 1e-6)
    raw = np.log(np.expm1(arr))
    return nn.Parameter(torch.tensor(raw, dtype=DTYPE))


class BatchedPINODE(nn.Module):
    """Physics-informed residual Neural ODE, batched over the ensemble.

    Two extra flags enable the "Autonomous Smooth Closure" variant (§ summary):
      * ``autonomous=True``  -> closure is g_psi(y, z, U) (no explicit time input),
        so the ODE is autonomous and cannot extrapolate a spurious time-trend
        outside the assimilation window.
      * ``activation="tanh"`` -> smooth, bounded vector field that saturates
        rather than ramping linearly (the ReLU drift failure mode).
    Defaults reproduce the paper PI-NODE exactly.
    """

    def __init__(self, ensemble: int, init, omega: float, s_r: float,
                 hidden: int = 32, autonomous: bool = False,
                 activation: str = "relu", theta_init: Optional[dict] = None):
        super().__init__()
        self.E = ensemble
        self.omega = float(omega)
        self.s_r = float(s_r)
        self.autonomous = bool(autonomous)
        n_in = 3 if self.autonomous else 4
        self.residual = BatchedMLP(ensemble, n_in=n_in, n_out=2, hidden=hidden,
                                   activation=activation)
        if theta_init is None:
            self.raw_rho = _raw_param(ensemble, init.rho)
            self.raw_gamma = _raw_param(ensemble, init.beta)
            self.raw_Keff = _raw_param(ensemble, max(init.K - 0.2, 1e-5))
            self.raw_mu = _raw_param(ensemble, init.mu)
            self.raw_tau = _raw_param(ensemble, init.tau)
        else:
            # Per-member warm start from the assimilated ODE fit (UDE backbone).
            self.raw_rho = _raw_from_array(theta_init["rho"])
            self.raw_gamma = _raw_from_array(theta_init["gamma"])
            self.raw_Keff = _raw_from_array(theta_init["Keff"], offset=0.2)
            self.raw_mu = _raw_from_array(theta_init["mu"], offset=1e-8)
            self.raw_tau = _raw_from_array(theta_init["tau"])

    def physics_parameters(self):
        return [self.raw_rho, self.raw_gamma, self.raw_Keff, self.raw_mu, self.raw_tau]

    def positive_params(self):
        rho = F.softplus(self.raw_rho) + 1e-6
        gamma = F.softplus(self.raw_gamma) + 1e-6
        Keff = F.softplus(self.raw_Keff) + 0.2
        mu = F.softplus(self.raw_mu) + 1e-8
        tau = F.softplus(self.raw_tau) + 1e-6
        return rho, gamma, Keff, mu, tau

    def _residual_input(self, y, z, t_norm, U):
        if self.autonomous:
            return torch.stack([y, z, U], dim=-1)
        tn = torch.full_like(y, float(t_norm))
        return torch.stack([y, z, tn, U], dim=-1)

    def rhs(self, s: torch.Tensor, t_norm: float, U: torch.Tensor) -> torch.Tensor:
        y, z = s[:, 0], s[:, 1]
        rho, gamma, Keff, mu, tau = self.positive_params()
        mech_y = rho * y * (1.0 - y / Keff) - gamma * z * y - mu * y
        mech_z = -z / tau + U
        res = self.residual(self._residual_input(y, z, t_norm, U))
        dy = self.omega * mech_y + self.s_r * res[:, 0]
        return torch.stack([dy, mech_z], dim=-1)


class BatchedNODE(nn.Module):
    """Dose-aware Neural ODE on latent state (y, z), batched over the ensemble.

    Sign-constrained so radiation cannot increase mass:
        dy/dt = growth_nn - softplus(kill_nn) * z * y
        dz/dt = alpha * U - z / tau
    """

    def __init__(self, ensemble: int, init, hidden: int = 32):
        super().__init__()
        self.E = ensemble
        self.nn = BatchedMLP(ensemble, n_in=4, n_out=2, hidden=hidden)
        self.raw_alpha = _raw_param(ensemble, 1.0)
        self.raw_tau = _raw_param(ensemble, init.tau)

    def positive_params(self):
        return F.softplus(self.raw_alpha) + 1e-6, F.softplus(self.raw_tau) + 1e-6

    def rhs(self, s: torch.Tensor, t_norm: float, U: torch.Tensor) -> torch.Tensor:
        y, z = s[:, 0], s[:, 1]
        tn = torch.full_like(y, float(t_norm))
        out = self.nn(torch.stack([y, z, tn, U], dim=-1))
        growth = out[:, 0]
        kill = F.softplus(out[:, 1])
        alpha, tau = self.positive_params()
        dy = growth - kill * z * y
        dz = alpha * U - z / tau
        return torch.stack([dy, dz], dim=-1)


def integrate(model, t: torch.Tensor, U: torch.Tensor, y0: torch.Tensor,
              time_norm: float) -> torch.Tensor:
    """Vectorised RK4 over a shared time grid for all ensemble members.

    Args:
        t: (T,) shared time grid.
        U: (T,) shared dose input.
        y0: (E,) per-member initial tumour mass (z0 = 0).
        time_norm: denominator used to normalise time inside the networks.
    Returns:
        states (T, E, 2).
    """
    E = y0.shape[0]
    z0 = torch.zeros_like(y0)
    s = torch.stack([y0, z0], dim=-1)  # (E, 2)
    states = [s]
    ones = torch.ones(E, dtype=y0.dtype, device=y0.device)

    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        tn0 = float(t[i - 1] / time_norm)
        tn1 = float((t[i - 1] + 0.5 * dt) / time_norm)
        tn2 = float(t[i] / time_norm)
        U0 = U[i - 1] * ones
        U1 = 0.5 * (U[i - 1] + U[i]) * ones
        U2 = U[i] * ones

        k1 = model.rhs(s, tn0, U0)
        k2 = model.rhs(torch.clamp(s + 0.5 * dt * k1, min=0.0), tn1, U1)
        k3 = model.rhs(torch.clamp(s + 0.5 * dt * k2, min=0.0), tn1, U1)
        k4 = model.rhs(torch.clamp(s + dt * k3, min=0.0), tn2, U2)
        s = torch.clamp(s + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0, min=0.0)
        states.append(s)

    return torch.stack(states)  # (T, E, 2)
