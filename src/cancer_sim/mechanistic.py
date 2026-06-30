"""Reduced two-state mechanistic ODE surrogate (paper Eq. ``ode_rt_2state``).

    dy/dt = rho * y * (1 - y/Keff) - gamma * z * y - mu * y
    dz/dt = -z / tau + U(t)

``y`` is the normalised tumour mass, ``z`` is a latent accumulated
radiation-damage variable providing multi-fraction memory.  Parameters
theta = (rho, gamma, Keff, mu, tau) are estimated by nonlinear least squares
(global differential evolution + local Powell refinement, paper Eq.
``ode_training``).
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import differential_evolution, minimize


def rhs(y: float, z: float, U: float, rho: float, gamma: float,
        Keff: float, mu: float, tau: float) -> Tuple[float, float]:
    dy = rho * y * (1.0 - y / Keff) - gamma * z * y - mu * y
    dz = -z / tau + U
    return dy, dz


def integrate(t: np.ndarray, U: np.ndarray, y0: float, rho: float, gamma: float,
              Keff: float, mu: float, tau: float) -> Tuple[np.ndarray, np.ndarray]:
    """RK4 integration on the (possibly non-uniform) time grid ``t``.

    ``U`` is linearly interpolated to RK4 midpoints; states are clamped to be
    non-negative for biological plausibility.
    """
    y = np.empty_like(t, dtype=float)
    z = np.empty_like(t, dtype=float)
    y[0], z[0] = y0, 0.0

    for i in range(1, len(t)):
        dt = float(t[i] - t[i - 1])
        yp, zp = float(y[i - 1]), float(z[i - 1])
        U0, U2 = float(U[i - 1]), float(U[i])
        U1 = 0.5 * (U0 + U2)

        k1y, k1z = rhs(yp, zp, U0, rho, gamma, Keff, mu, tau)
        k2y, k2z = rhs(max(0.0, yp + 0.5 * dt * k1y), max(0.0, zp + 0.5 * dt * k1z),
                       U1, rho, gamma, Keff, mu, tau)
        k3y, k3z = rhs(max(0.0, yp + 0.5 * dt * k2y), max(0.0, zp + 0.5 * dt * k2z),
                       U1, rho, gamma, Keff, mu, tau)
        k4y, k4z = rhs(max(0.0, yp + dt * k3y), max(0.0, zp + dt * k3z),
                       U2, rho, gamma, Keff, mu, tau)

        y[i] = max(0.0, yp + dt * (k1y + 2 * k2y + 2 * k3y + k4y) / 6.0)
        z[i] = max(0.0, zp + dt * (k1z + 2 * k2z + 2 * k3z + k4z) / 6.0)
    return y, z


def fit(t_train: np.ndarray, y_train: np.ndarray, U_train: np.ndarray,
        seed: int = 123) -> Dict[str, float]:
    """Assimilate theta via differential evolution + Powell (log-parameterised)."""

    def unpack(q):
        # Powell is unbounded; clamp the log-parameters to keep exp() finite.
        return tuple(math.exp(min(max(float(v), -30.0), 30.0)) for v in q)

    def loss(q):
        rho, gamma, Keff, mu, tau = unpack(q)
        pred, _ = integrate(t_train, U_train, y_train[0], rho, gamma, Keff, mu, tau)
        penalty = 1e-5 * (math.log(Keff / 1.5) ** 2 + math.log(tau / 3.0) ** 2)
        return float(np.mean((pred - y_train) ** 2) + penalty)

    bounds = [
        (math.log(0.001), math.log(0.30)),   # rho
        (math.log(0.001), math.log(30.0)),   # gamma
        (math.log(0.3), math.log(5.0)),      # Keff
        (math.log(1e-6), math.log(0.08)),    # mu
        (math.log(0.2), math.log(20.0)),     # tau
    ]

    de = differential_evolution(loss, bounds=bounds, seed=seed, polish=False,
                                maxiter=80, popsize=8, tol=1e-7, workers=1)
    res = minimize(loss, de.x, method="Powell",
                   options={"maxiter": 2500, "xtol": 1e-8, "ftol": 1e-10})
    rho, gamma, Keff, mu, tau = unpack(res.x)
    return {"rho": rho, "gamma": gamma, "Keff": Keff, "mu": mu, "tau": tau,
            "train_loss": float(res.fun), "success": bool(res.success)}
