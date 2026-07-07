"""Volumetric-Growth-plus-Damage (VGD) reduced RT surrogate -- the improved
method for the *with-radiation* case.

Rationale
---------
The regrowth study (``growth_surrogate``) shows that spatially integrated
Fisher--Kolmogorov mass grows *volumetrically*: the cube root of mass (the front
radius) is linear in time, not exponential.  The paper's reduced RT model instead
puts the tumour on a **logistic** mass equation:

    dy/dt = rho*y(1 - y/Keff) - gamma*z*y - mu*y ,   dz/dt = -z/tau + U(t)

so between/after dose fractions it regrows with the wrong (logistic) curvature and
systematically under-predicts recurrence -- the exact failure the original write-up
diagnosed (|e80| ~ 25 %).

VGD keeps the paper's two-state structure and its radiotherapy damage variable
``z`` unchanged, but moves the tumour equation into **front-radius space**
``u = m^(1/3)`` where invasion growth is linear:

    du/dt = beta - g * z * u          (constant-speed front, radiation shrinks it)
    dz/dt = -z / tau + U(t)
    m(t)  = u(t) ** 3

This is the minimal, physically-motivated correction: same RT machinery, correct
regrowth geometry.  Three interpretable parameters ``(beta, g, tau)`` -- ``beta``
is the radial regrowth speed the clinicians want.  Fit by differential evolution +
Powell, exactly like ``mechanistic.fit`` (log-parameterised, capped).
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import differential_evolution, minimize

_P = 3.0   # volumetric exponent (spherical front)


def integrate_vgd(t: np.ndarray, U: np.ndarray, u0: float,
                  beta: float, g: float, tau: float
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """RK4 in (u, z) space on grid ``t``; returns (mass = u**p, z)."""
    u = np.empty_like(t, dtype=float)
    z = np.empty_like(t, dtype=float)
    u[0], z[0] = u0, 0.0

    def rhs(uu, zz, UU):
        return beta - g * zz * uu, -zz / tau + UU

    for i in range(1, len(t)):
        dt = float(t[i] - t[i - 1])
        up, zp = float(u[i - 1]), float(z[i - 1])
        U0, U2 = float(U[i - 1]), float(U[i])
        U1 = 0.5 * (U0 + U2)
        k1u, k1z = rhs(up, zp, U0)
        k2u, k2z = rhs(max(0.0, up + 0.5 * dt * k1u), max(0.0, zp + 0.5 * dt * k1z), U1)
        k3u, k3z = rhs(max(0.0, up + 0.5 * dt * k2u), max(0.0, zp + 0.5 * dt * k2z), U1)
        k4u, k4z = rhs(max(0.0, up + dt * k3u), max(0.0, zp + dt * k3z), U2)
        u[i] = max(0.0, up + dt * (k1u + 2 * k2u + 2 * k3u + k4u) / 6.0)
        z[i] = max(0.0, zp + dt * (k1z + 2 * k2z + 2 * k3z + k4z) / 6.0)
    return u ** _P, z


def fit_vgd(t_train: np.ndarray, m_train: np.ndarray, U_train: np.ndarray,
            seed: int = 123, de_maxiter: int = 60,
            powell_maxiter: int = 600) -> Dict[str, float]:
    """Assimilate (beta, g, tau) by DE + Powell (log-parameterised)."""
    u0 = float(max(m_train[0], 1e-9) ** (1.0 / _P))

    def unpack(q):
        return tuple(math.exp(min(max(float(v), -30.0), 30.0)) for v in q)

    def loss(q):
        beta, g, tau = unpack(q)
        pred, _ = integrate_vgd(t_train, U_train, u0, beta, g, tau)
        return float(np.mean((pred - m_train) ** 2))

    bounds = [
        (math.log(1e-3), math.log(2.0)),    # beta (radial regrowth speed)
        (math.log(1e-3), math.log(30.0)),   # g (radiosensitivity)
        (math.log(0.2), math.log(20.0)),    # tau (damage decay)
    ]
    de = differential_evolution(loss, bounds=bounds, seed=seed, polish=False,
                                maxiter=de_maxiter, popsize=10, tol=1e-7, workers=1)
    res = minimize(loss, de.x, method="Powell",
                   options={"maxiter": powell_maxiter, "xtol": 1e-8, "ftol": 1e-10})
    beta, g, tau = unpack(res.x)
    return {"beta": beta, "g": g, "tau": tau, "u0": u0,
            "train_loss": float(res.fun), "success": bool(res.success)}
