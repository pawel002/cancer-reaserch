"""Volumetric Fisher-front regression (FFR) -- a reduced surrogate for the
*unconstrained expansion phase* of tumour regrowth.

Motivation
----------
The paper's reduced models integrate a state whose tumour equation is a
**logistic / exponential** growth law:

    dy/dt = rho * y (1 - y/Keff) - mu*y        (mechanistic 2-state, z=0 with no RT)
    dy/dt = growth_nn(...)                      (NODE)
    dy/dt = omega*mech_y + s_r*g_psi(...)       (PI-NODE)

But the ground truth is the *spatially integrated mass* of a 3-D
Fisher--Kolmogorov (FK) invasion.  For a focal seed spreading through tissue,
the reaction-diffusion front advances at a (locally) constant speed -- the
Fisher--KPP wave speed ``v = 2*sqrt(D*f)`` (Swanson et al. 2003, "velocity of
radial expansion" of gliomas).  With the interior saturated at carrying
capacity, the invaded volume grows like a sphere of radius ``R(t) = R0 + v t``:

    M(t)  ~  (4/3) pi R(t)^3   =>   M(t) / M(0)  ~  (1 + b t)^3 ,   b = v / R0 .

So the *cube root* of normalised mass is (to leading order) **linear in time**
during the unconstrained expansion phase -- not exponential.  A logistic/NODE
fit to a short assimilation window therefore extrapolates with the wrong
curvature (it must choose between exponential blow-up and premature saturation),
while a model with the correct volumetric curvature extrapolates cleanly.

Method
------
Given sparse noisy observations ``(t_i, m_i)`` in an assimilation window, FFR
fits the straight line

    u(t) := m(t)^(1/p)  ~=  alpha + beta * t          (p = 3, volumetric)

by (optionally weighted) ordinary least squares -- a closed-form solve, no
iterative optimiser, no neural network.  The forecast is ``m(t) = max(alpha +
beta t, 0)^p``.  The single physical output ``beta`` is the cube-root-mass
growth rate: an estimate of the tumour's *radial regrowth speed* (proportional
to the Fisher wave speed), which the logistic ``rho`` cannot report because it
conflates proliferation and diffusion.

Everything is vectorised over the noisy ensemble (E members) so the whole
ensemble is fit in one closed-form linear solve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FFRFit:
    p: float
    alpha: np.ndarray     # (E,) intercept of u = m^(1/p)
    beta: np.ndarray      # (E,) slope  (cube-root-mass growth rate)
    r0_frac: np.ndarray   # (E,) implied normalised seed radius = alpha + beta*t0
    speed_index: np.ndarray  # (E,) beta / (alpha at window start) ~ front speed / R0


def _ols_lines(t: np.ndarray, U: np.ndarray, w: Optional[np.ndarray] = None):
    """Weighted OLS of each column of U (T,E) on t (T,). Returns (alpha, beta)."""
    t = np.asarray(t, dtype=float)
    U = np.asarray(U, dtype=float)
    if U.ndim == 1:
        U = U[:, None]
    if w is None:
        w = np.ones_like(t)
    w = np.asarray(w, dtype=float)
    W = w.sum()
    tbar = (w * t).sum() / W
    Ubar = (w[:, None] * U).sum(axis=0) / W
    dt = t - tbar
    Stt = (w * dt * dt).sum()
    StU = (w[:, None] * dt[:, None] * (U - Ubar)).sum(axis=0)
    beta = StU / (Stt + 1e-12)
    alpha = Ubar - beta * tbar
    return alpha, beta


def fit_ffr(t_fit: np.ndarray, Y_fit: np.ndarray, p: float = 3.0,
            sample_weights: Optional[np.ndarray] = None) -> FFRFit:
    """Fit the volumetric Fisher-front model to an ensemble of observations.

    Args:
        t_fit: (T,) assimilation times.
        Y_fit: (T, E) per-member noisy normalised-mass observations (or (T,)).
        p:     volumetric exponent (3 for a spherical front).
        sample_weights: optional (T,) weights.
    """
    Y = np.asarray(Y_fit, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    U = np.clip(Y, 1e-9, None) ** (1.0 / p)     # (T, E)
    alpha, beta = _ols_lines(np.asarray(t_fit, float), U, sample_weights)
    t0 = float(np.min(t_fit))
    r0 = alpha + beta * t0
    speed_index = beta / (np.abs(r0) + 1e-9)
    return FFRFit(p=p, alpha=alpha, beta=beta, r0_frac=r0, speed_index=speed_index)


def predict_ffr(fit: FFRFit, t_eval: np.ndarray) -> np.ndarray:
    """Forecast normalised mass on ``t_eval``. Returns (T_eval, E)."""
    t = np.asarray(t_eval, dtype=float)[:, None]         # (T,1)
    u = fit.alpha[None, :] + fit.beta[None, :] * t       # (T,E)
    return np.clip(u, 0.0, None) ** fit.p


def fit_ffr_auto_p(t_fit: np.ndarray, Y_fit: np.ndarray,
                   p_grid=(2.0, 2.5, 3.0, 3.5),
                   sample_weights: Optional[np.ndarray] = None):
    """Variant that also selects the exponent ``p`` by in-window fit quality.

    Chooses the single ``p`` (shared across the ensemble) minimising the median
    in-window residual of ``m^(1/p)``'s linear fit.  Returned for the robustness
    study; the headline method fixes ``p=3`` on mechanistic grounds.
    """
    t = np.asarray(t_fit, float)
    best = None
    for p in p_grid:
        fit = fit_ffr(t, Y_fit, p=p, sample_weights=sample_weights)
        pred = predict_ffr(fit, t)                       # (T,E)
        Y = np.asarray(Y_fit, float)
        if Y.ndim == 1:
            Y = Y[:, None]
        resid = np.median(np.sqrt(np.mean((pred - Y) ** 2, axis=0)))
        if best is None or resid < best[0]:
            best = (resid, p, fit)
    return best[2], best[1]
