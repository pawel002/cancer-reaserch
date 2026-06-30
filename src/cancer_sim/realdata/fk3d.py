"""3-D Fisher-Kolmogorov forward solver with a grafted radiotherapy damage field.

The diffusion-reaction kernel (anisotropic WM/GM diffusion, focal Gaussian seed)
is adapted from GliODIL's ``synthetic_generator.py`` (``m_Tildas`` / ``get_D`` /
``FK_update`` / ``gauss_sol3d``):

    GliODIL/code/cases/GliODIL/synthetic_generator.py:105-258

so that the patient growth physics here is *identical* to what GliODIL infers.
On top of GliODIL's growth-only model we graft a radiotherapy term that is the
spatial analogue of this repository's reduced 2-state ODE
(``cancer_sim.mechanistic``):

    reduced 0-D (mechanistic.py):
        dy/dt = rho * y (1 - y/K) - gamma * z * y - mu * y
        dz/dt = -z / tau + U(t)

    3-D spatial analogue (this module):
        dA/dt = div(D grad A) + f * A (1 - A) - gamma * Z * A      (tumour field)
        dZ/dt = -Z / tau + U(t) * beam(x)                          (damage field)

Spatially integrating the tumour field gives the scalar ``normalized_mass(t)``
trajectory the surrogates consume; ``U(t)`` is the (scalar) dose-rate pulse
train and ``W_eff(t)`` is the tumour-mass-weighted mean damage -- a 0-D reduction
of ``Z`` that mirrors the 2-state ``z``.

Pure NumPy: runs on CPU.  Conventions follow GliODIL: spatial axes (x, y, z) map
to array axes (0, 1, 2) and the grid spacing is unity (``dx = dy = dz = 1``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
@dataclass
class FK3DConfig:
    """Forward-simulation settings.

    Growth parameters (``Dw``, ``f``, ``Dw_ratio``, seed) come from GliODIL's
    inference for a real patient.  The radiotherapy/time parameters default to
    this repository's surrogate configuration so that the emitted curve drops
    straight into the existing experiments.
    """

    # --- growth (GliODIL-inferred per patient) ---
    Dw: float = 0.20            # water (white-matter) diffusion coefficient
    f: float = 0.12             # proliferation rate (== rho)
    Dw_ratio: float = 20.0      # WM/GM diffusion ratio (white-matter preference)
    matter_threshold: float = 0.1   # WM+GM below this is "outside the brain"

    # --- focal seed (fractional grid position + Gaussian spread) ---
    seed_center: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    seed_Dt: float = 15.0       # gauss_sol3d spread (GliODIL default)
    seed_mass: float = 1500.0   # gauss_sol3d amplitude (GliODIL default)

    # --- time axis (mapped onto the surrogate's [0, T] grid) ---
    T: float = 80.0             # total horizon (surrogate units)
    dt_output: float = 0.1      # output cadence -> 801 samples on [0, 80]
    substeps: int = 12          # internal FK steps per output step (stability)

    # --- radiotherapy schedule (graft; defaults match the 2-D synthetic cases) ---
    dose_times: Tuple[float, ...] = (15.0, 45.0)
    dose_amplitude: float = 1.2     # peak dose-rate (== dose_rate_amplitude)
    dose_duration: float = 0.6      # pulse width in surrogate-time units
    dose_edge_fraction: float = 0.25
    dose_scale: float = 1.0         # per-case beam intensity multiplier
    damage_decay_time: float = 3.0  # tau (== damage_decay_time)
    gamma: float = 1.25             # radiosensitivity (== beta)

    # --- spatial beam geometry (fractional grid units) ---
    beam_center: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    beam_radius_frac: float = 0.35  # sphere radius as a fraction of the grid
    beam_edge_smoothing: float = 0.04

    # --- optional spatial snapshots (for inspection / plotting) ---
    snapshot_times: Tuple[float, ...] = field(default_factory=tuple)


# ----------------------------------------------------------------------------
# Anisotropic diffusion (lifted from GliODIL synthetic_generator.py)
# ----------------------------------------------------------------------------
def _m_tildas(WM: np.ndarray, GM: np.ndarray, th: float) -> Dict[str, np.ndarray]:
    out = {}
    for ax in range(3):
        cond = np.logical_and(
            np.roll(WM, -1, axis=ax) + np.roll(GM, -1, axis=ax) >= th,
            WM + GM >= th,
        )
        out[f"WM_{ax}"] = np.where(cond, (np.roll(WM, -1, axis=ax) + WM) / 2.0, 0.0)
        out[f"GM_{ax}"] = np.where(cond, (np.roll(GM, -1, axis=ax) + GM) / 2.0, 0.0)
    return out


def _get_D(WM: np.ndarray, GM: np.ndarray, th: float, Dw: float,
           Dw_ratio: float) -> Dict[str, np.ndarray]:
    """Face-centred diffusion coefficients (white-matter-preferential)."""
    M = _m_tildas(WM, GM, th)
    D = {}
    for ax in range(3):
        D[f"minus_{ax}"] = Dw * (M[f"WM_{ax}"] + M[f"GM_{ax}"] / Dw_ratio)
        D[f"plus_{ax}"] = Dw * (np.roll(M[f"WM_{ax}"], 1, axis=ax)
                                + np.roll(M[f"GM_{ax}"], 1, axis=ax) / Dw_ratio)
    return D


def _diffusion(A: np.ndarray, D: Dict[str, np.ndarray]) -> np.ndarray:
    SP = np.zeros_like(A)
    for ax in range(3):
        SP = SP + (D[f"plus_{ax}"] * (np.roll(A, 1, axis=ax) - A)
                   - D[f"minus_{ax}"] * (A - np.roll(A, -1, axis=ax)))
    return SP


def _gauss_seed(shape, center_frac, Dt: float, mass: float) -> np.ndarray:
    """Focal Gaussian initial condition (GliODIL ``gauss_sol3d``)."""
    nx, ny, nz = shape
    cx, cy, cz = (center_frac[0] * nx, center_frac[1] * ny, center_frac[2] * nz)
    xv, yv, zv = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                             indexing="ij")
    r2 = (xv - cx) ** 2 + (yv - cy) ** 2 + (zv - cz) ** 2
    g = mass / np.power(4 * np.pi * Dt, 1.5) * np.exp(-r2 / (4 * Dt))
    g = np.where(g > 0.1, g, 0.0)
    g = np.where(g > 1.0, 1.0, g)
    return g.astype(np.float64)


def _beam_mask(shape, center_frac, radius_frac: float,
               edge_smoothing: float) -> np.ndarray:
    """Smooth spherical dose-delivery mask in [0, 1] (analogue of the 2-D beam)."""
    nx, ny, nz = shape
    cx, cy, cz = (center_frac[0] * nx, center_frac[1] * ny, center_frac[2] * nz)
    xv, yv, zv = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                             indexing="ij")
    scale = float(max(shape))
    dist = np.sqrt((xv - cx) ** 2 + (yv - cy) ** 2 + (zv - cz) ** 2) / scale
    edge = max(edge_smoothing, 1e-6)
    return 1.0 / (1.0 + np.exp((dist - radius_frac) / edge))


# ----------------------------------------------------------------------------
# Dose-rate pulse train U(t)  (matches the 2-D generator's trapezoidal pulse)
# ----------------------------------------------------------------------------
def dose_rate(t: np.ndarray, cfg: FK3DConfig) -> np.ndarray:
    """Smooth trapezoidal dose-rate U(t) summed over all fractions."""
    t = np.asarray(t, dtype=float)
    U = np.zeros_like(t)
    half = 0.5 * cfg.dose_duration
    edge = max(cfg.dose_edge_fraction * cfg.dose_duration, 1e-9)
    amp = cfg.dose_amplitude * cfg.dose_scale
    for t0 in cfg.dose_times:
        local = (t - t0) / 1.0
        rise = 0.5 * (1.0 + np.tanh((local + half) / edge))
        fall = 0.5 * (1.0 + np.tanh((half - local) / edge))
        U = U + amp * rise * fall
    return U


# ----------------------------------------------------------------------------
# Forward simulation
# ----------------------------------------------------------------------------
@dataclass
class FK3DResult:
    t: np.ndarray                 # (n_out,) surrogate-time grid
    normalized_mass: np.ndarray   # (n_out,) M(t) / M(0)
    U_t: np.ndarray               # (n_out,) scalar dose rate
    W_eff: np.ndarray             # (n_out,) tumour-mass-weighted mean damage
    mass_abs: np.ndarray          # (n_out,) raw integrated mass (cells)
    snapshots: Dict[float, np.ndarray]  # {time: 3-D tumour field}
    info: Dict[str, float]


def simulate(WM: np.ndarray, GM: np.ndarray, cfg: FK3DConfig) -> FK3DResult:
    """Run the 3-D FK-with-radiotherapy forward model and reduce to a curve.

    Parameters
    ----------
    WM, GM : 3-D arrays in [0, 1]
        White- and grey-matter probability maps (patient anatomy).
    cfg : FK3DConfig
        Growth + radiotherapy + time settings.

    Returns
    -------
    FK3DResult with the surrogate-ready ``normalized_mass``/``U_t``/``W_eff``
    series on the ``[0, T]`` output grid, plus optional spatial snapshots.
    """
    WM = np.asarray(WM, dtype=np.float64)
    GM = np.asarray(GM, dtype=np.float64)
    if WM.shape != GM.shape or WM.ndim != 3:
        raise ValueError("WM and GM must be matching 3-D arrays")

    shape = WM.shape
    brain = (WM + GM) > cfg.matter_threshold

    D = _get_D(WM, GM, cfg.matter_threshold, cfg.Dw, cfg.Dw_ratio)
    beam = _beam_mask(shape, cfg.beam_center, cfg.beam_radius_frac,
                      cfg.beam_edge_smoothing) * brain

    A = _gauss_seed(shape, cfg.seed_center, cfg.seed_Dt, cfg.seed_mass) * brain
    Z = np.zeros_like(A)

    n_out = int(round(cfg.T / cfg.dt_output)) + 1
    t_out = np.linspace(0.0, cfg.T, n_out)
    U_out = dose_rate(t_out, cfg)

    dt = cfg.dt_output / cfg.substeps
    # Explicit-diffusion stability margin (dx = 1, 3-D): dt < 1 / (6 * Dmax).
    d_max = float(max(np.max(D[f"plus_{ax}"]) for ax in range(3)) or cfg.Dw)
    dt_stable = 1.0 / (6.0 * d_max + 1e-12)
    if dt > dt_stable:
        raise ValueError(
            f"Unstable: dt_internal={dt:.4g} > stability limit {dt_stable:.4g}. "
            f"Increase substeps (>= {int(np.ceil(cfg.dt_output / dt_stable))}) "
            f"or lower Dw.")

    mass_abs = np.empty(n_out)
    W_eff = np.empty(n_out)
    snapshots: Dict[float, np.ndarray] = {}
    snap_targets = sorted(set(cfg.snapshot_times))

    def _record(i: int):
        m = float(A.sum())
        mass_abs[i] = m
        W_eff[i] = float((Z * A).sum() / m) if m > 0 else 0.0

    _record(0)
    for tt in snap_targets:
        if tt <= t_out[0]:
            snapshots[float(tt)] = A.copy()

    for i in range(1, n_out):
        for s in range(cfg.substeps):
            # dose rate is piecewise across the output step; midpoint sampling
            frac = (s + 0.5) / cfg.substeps
            U_now = float(np.interp(t_out[i - 1] + frac * cfg.dt_output,
                                    t_out, U_out))
            SP = _diffusion(A, D)
            reaction = cfg.f * A * (1.0 - A) - cfg.gamma * Z * A
            A = A + (SP + reaction) * dt
            np.clip(A, 0.0, 1.0, out=A)
            A *= brain
            Z = Z + (-Z / cfg.damage_decay_time + U_now * beam) * dt
            np.clip(Z, 0.0, None, out=Z)
        _record(i)
        for tt in snap_targets:
            if abs(t_out[i] - tt) < 0.5 * cfg.dt_output:
                snapshots[float(tt)] = A.copy()

    m0 = mass_abs[0] if mass_abs[0] > 0 else 1.0
    normalized = mass_abs / m0

    # Per-dose knockdown depth: min mass in a window after each fraction,
    # relative to the mass just before that fraction (the RT-response signature).
    dips = []
    for t0 in cfg.dose_times:
        pre = float(np.interp(t0, t_out, normalized))
        win = (t_out >= t0) & (t_out <= min(t0 + 15.0, cfg.T))
        if win.any() and pre > 0:
            dips.append(1.0 - float(normalized[win].min()) / pre)

    info = {
        "Dw": cfg.Dw, "f": cfg.f, "Dw_ratio": cfg.Dw_ratio,
        "grid": int(shape[0]), "n_steps": int((n_out - 1) * cfg.substeps),
        "dt_internal": dt, "dt_stable": dt_stable,
        "mass0": float(m0), "mass_final": float(mass_abs[-1]),
        "peak_mass_ratio": float(normalized.max()),
        "dose_knockdowns": [round(d, 3) for d in dips],
    }
    return FK3DResult(t_out, normalized, U_out, W_eff, mass_abs, snapshots, info)
