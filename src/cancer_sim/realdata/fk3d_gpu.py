"""GPU (PyTorch) Fisher-Kolmogorov forward solver -- regrowth only, no radiotherapy.

This is a CUDA port of :mod:`cancer_sim.realdata.fk3d` restricted to the
*unconstrained expansion phase*: the radiotherapy machinery of the reference
solver (dose pulse ``U(t)``, damage field ``Z``, kill term ``-gamma*Z*A``) is
removed, leaving the pure anisotropic reaction--diffusion growth GliODIL infers:

    dA/dt = div(D grad A) + f * A (1 - A)

Motivation
----------
The reduced 0-D surrogates (``mechanistic`` / ``surrogates``) integrate a tiny
``(ensemble, 2)`` state with a sequential RK4 loop, so a GPU gives no speed-up
there (kernel-launch overhead dominates -- see ``summary.md`` s2).  The genuinely
GPU-bound cost in this project is *generating the ground-truth curves*: a 3-D
stencil over ~1-7 M voxels for thousands of steps per patient.  That is what this
module accelerates; the diffusion stencil, D-coefficient construction, and
Gaussian seed are numerically identical to the NumPy reference (verified in
``experiments/validate_gpu_fk.py``), so the emitted ``normalized_mass(t)`` curve
matches the CPU solver to solver tolerance.

Conventions follow GliODIL / ``fk3d``: spatial axes map to array axes (0,1,2),
unit grid spacing (dx=dy=dz=1), explicit forward Euler in time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

# reuse the CPU config so the two solvers share one parameter object
from .fk3d import FK3DConfig


def _roll(x: torch.Tensor, shift: int, ax: int) -> torch.Tensor:
    return torch.roll(x, shifts=shift, dims=ax)


def _m_tildas(WM: torch.Tensor, GM: torch.Tensor, th: float) -> Dict[str, torch.Tensor]:
    """Face-centred tissue averages (mirror of fk3d._m_tildas)."""
    out = {}
    for ax in range(3):
        cond = (_roll(WM, -1, ax) + _roll(GM, -1, ax) >= th) & (WM + GM >= th)
        condf = cond.to(WM.dtype)
        out[f"WM_{ax}"] = condf * (_roll(WM, -1, ax) + WM) / 2.0
        out[f"GM_{ax}"] = condf * (_roll(GM, -1, ax) + GM) / 2.0
    return out


def _get_D(WM: torch.Tensor, GM: torch.Tensor, th: float, Dw: float,
           Dw_ratio: float) -> Dict[str, torch.Tensor]:
    """White-matter-preferential face-centred diffusion coefficients."""
    M = _m_tildas(WM, GM, th)
    D = {}
    for ax in range(3):
        D[f"minus_{ax}"] = Dw * (M[f"WM_{ax}"] + M[f"GM_{ax}"] / Dw_ratio)
        D[f"plus_{ax}"] = Dw * (_roll(M[f"WM_{ax}"], 1, ax)
                                + _roll(M[f"GM_{ax}"], 1, ax) / Dw_ratio)
    return D


def _diffusion(A: torch.Tensor, D: Dict[str, torch.Tensor]) -> torch.Tensor:
    SP = torch.zeros_like(A)
    for ax in range(3):
        SP = SP + (D[f"plus_{ax}"] * (_roll(A, 1, ax) - A)
                   - D[f"minus_{ax}"] * (A - _roll(A, -1, ax)))
    return SP


def _gauss_seed(shape, center_frac, Dt: float, mass: float,
                device, dtype) -> torch.Tensor:
    nx, ny, nz = shape
    cx, cy, cz = center_frac[0] * nx, center_frac[1] * ny, center_frac[2] * nz
    xv, yv, zv = torch.meshgrid(
        torch.arange(nx, device=device, dtype=dtype),
        torch.arange(ny, device=device, dtype=dtype),
        torch.arange(nz, device=device, dtype=dtype),
        indexing="ij")
    r2 = (xv - cx) ** 2 + (yv - cy) ** 2 + (zv - cz) ** 2
    g = mass / (4 * np.pi * Dt) ** 1.5 * torch.exp(-r2 / (4 * Dt))
    g = torch.where(g > 0.1, g, torch.zeros_like(g))
    g = torch.clamp(g, max=1.0)
    return g


def _beam_mask(shape, center_frac, radius_frac: float, edge_smoothing: float,
               device, dtype) -> torch.Tensor:
    """Smooth spherical dose-delivery mask in [0, 1] (mirror of fk3d._beam_mask)."""
    nx, ny, nz = shape
    cx, cy, cz = center_frac[0] * nx, center_frac[1] * ny, center_frac[2] * nz
    xv, yv, zv = torch.meshgrid(
        torch.arange(nx, device=device, dtype=dtype),
        torch.arange(ny, device=device, dtype=dtype),
        torch.arange(nz, device=device, dtype=dtype),
        indexing="ij")
    scale = float(max(shape))
    dist = torch.sqrt((xv - cx) ** 2 + (yv - cy) ** 2 + (zv - cz) ** 2) / scale
    edge = max(edge_smoothing, 1e-6)
    return 1.0 / (1.0 + torch.exp((dist - radius_frac) / edge))


def _dose_rate(t: np.ndarray, cfg: FK3DConfig) -> np.ndarray:
    """Smooth trapezoidal dose-rate U(t) summed over fractions (mirror of CPU)."""
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


@dataclass
class FK3DGrowthResult:
    t: np.ndarray                 # (n_out,) surrogate-time grid
    normalized_mass: np.ndarray   # (n_out,) M(t) / M(0)
    mass_abs: np.ndarray          # (n_out,) raw integrated tumour mass
    front_radius: np.ndarray      # (n_out,) effective radius = (3 V / 4pi)^(1/3)
    info: Dict[str, float]
    U_t: np.ndarray = None        # (n_out,) scalar dose rate (RT runs only)
    W_eff: np.ndarray = None      # (n_out,) tumour-mass-weighted mean damage


def simulate_growth(WM: np.ndarray, GM: np.ndarray, cfg: FK3DConfig,
                    device: str = "cuda", dtype: torch.dtype = torch.float32
                    ) -> FK3DGrowthResult:
    """Run the regrowth-only 3-D FK forward model on ``device``.

    Parameters
    ----------
    WM, GM : 3-D arrays in [0,1] (patient white / grey matter probability maps).
    cfg    : FK3DConfig (only the growth + time fields are used; RT fields ignored).
    device : torch device string, e.g. ``"cuda:0"``.
    dtype  : ``torch.float32`` for speed, ``torch.float64`` for validation.

    Returns
    -------
    FK3DGrowthResult with ``normalized_mass(t)`` and an effective front radius.
    """
    dev = torch.device(device)
    WM_t = torch.as_tensor(np.asarray(WM), dtype=dtype, device=dev)
    GM_t = torch.as_tensor(np.asarray(GM), dtype=dtype, device=dev)
    if WM_t.shape != GM_t.shape or WM_t.ndim != 3:
        raise ValueError("WM and GM must be matching 3-D arrays")

    shape = tuple(WM_t.shape)
    brain = (WM_t + GM_t) > cfg.matter_threshold
    brain_f = brain.to(dtype)

    D = _get_D(WM_t, GM_t, cfg.matter_threshold, cfg.Dw, cfg.Dw_ratio)
    A = _gauss_seed(shape, cfg.seed_center, cfg.seed_Dt, cfg.seed_mass,
                    dev, dtype) * brain_f

    n_out = int(round(cfg.T / cfg.dt_output)) + 1
    t_out = np.linspace(0.0, cfg.T, n_out)

    dt = cfg.dt_output / cfg.substeps
    d_max = float(max(D[f"plus_{ax}"].max().item() for ax in range(3)) or cfg.Dw)
    dt_stable = 1.0 / (6.0 * d_max + 1e-12)
    if dt > dt_stable:
        raise ValueError(
            f"Unstable: dt_internal={dt:.4g} > stability limit {dt_stable:.4g}. "
            f"Increase substeps (>= {int(np.ceil(cfg.dt_output / dt_stable))}) "
            f"or lower Dw.")

    mass_abs = np.empty(n_out)
    front_radius = np.empty(n_out)
    voxel_thresh = 0.5   # a voxel counts as "invaded" above this concentration

    def _record(i: int):
        m = float(A.sum().item())
        mass_abs[i] = m
        invaded = float((A > voxel_thresh).sum().item())
        front_radius[i] = (3.0 * invaded / (4.0 * np.pi)) ** (1.0 / 3.0)

    _record(0)
    f = cfg.f
    for i in range(1, n_out):
        for _ in range(cfg.substeps):
            SP = _diffusion(A, D)
            A = A + (SP + f * A * (1.0 - A)) * dt
            A = torch.clamp(A, 0.0, 1.0) * brain_f
        _record(i)

    m0 = mass_abs[0] if mass_abs[0] > 0 else 1.0
    normalized = mass_abs / m0
    info = {
        "Dw": cfg.Dw, "f": cfg.f, "Dw_ratio": cfg.Dw_ratio,
        "grid": int(shape[0]), "n_steps": int((n_out - 1) * cfg.substeps),
        "dt_internal": dt, "dt_stable": dt_stable,
        "mass0": float(m0), "mass_final": float(mass_abs[-1]),
        "peak_mass_ratio": float(normalized.max()),
        "brain_voxels": float(brain_f.sum().item()),
        "fisher_speed": float(2.0 * np.sqrt(max(cfg.Dw * cfg.f, 0.0))),
        "device": str(dev), "dtype": str(dtype),
    }
    return FK3DGrowthResult(t_out, normalized, mass_abs, front_radius, info)


def simulate_rt(WM: np.ndarray, GM: np.ndarray, cfg: FK3DConfig,
                device: str = "cuda", dtype: torch.dtype = torch.float32
                ) -> FK3DGrowthResult:
    """GPU forward model WITH the grafted radiotherapy damage field (RT on).

    Adds GliODIL's growth kernel + the spatial analogue of this repo's 2-state
    RT model (mirror of the CPU ``fk3d.simulate``):

        dA/dt = div(D grad A) + f A (1 - A) - gamma Z A
        dZ/dt = -Z / tau      + U(t) * beam(x)

    Returns the same ``normalized_mass(t)`` plus the scalar dose rate ``U_t`` and
    the tumour-mass-weighted mean damage ``W_eff`` (the 0-D reduction of Z).
    """
    dev = torch.device(device)
    WM_t = torch.as_tensor(np.asarray(WM), dtype=dtype, device=dev)
    GM_t = torch.as_tensor(np.asarray(GM), dtype=dtype, device=dev)
    if WM_t.shape != GM_t.shape or WM_t.ndim != 3:
        raise ValueError("WM and GM must be matching 3-D arrays")

    shape = tuple(WM_t.shape)
    brain = (WM_t + GM_t) > cfg.matter_threshold
    brain_f = brain.to(dtype)

    D = _get_D(WM_t, GM_t, cfg.matter_threshold, cfg.Dw, cfg.Dw_ratio)
    beam = _beam_mask(shape, cfg.beam_center, cfg.beam_radius_frac,
                      cfg.beam_edge_smoothing, dev, dtype) * brain_f
    A = _gauss_seed(shape, cfg.seed_center, cfg.seed_Dt, cfg.seed_mass,
                    dev, dtype) * brain_f
    Z = torch.zeros_like(A)

    n_out = int(round(cfg.T / cfg.dt_output)) + 1
    t_out = np.linspace(0.0, cfg.T, n_out)
    U_out = _dose_rate(t_out, cfg)

    dt = cfg.dt_output / cfg.substeps
    d_max = float(max(D[f"plus_{ax}"].max().item() for ax in range(3)) or cfg.Dw)
    dt_stable = 1.0 / (6.0 * d_max + 1e-12)
    if dt > dt_stable:
        raise ValueError(
            f"Unstable: dt_internal={dt:.4g} > stability limit {dt_stable:.4g}. "
            f"Increase substeps (>= {int(np.ceil(cfg.dt_output / dt_stable))}).")

    mass_abs = np.empty(n_out)
    W_eff = np.empty(n_out)
    front_radius = np.empty(n_out)

    def _record(i: int):
        m = float(A.sum().item())
        mass_abs[i] = m
        W_eff[i] = float((Z * A).sum().item() / m) if m > 0 else 0.0
        front_radius[i] = (3.0 * float((A > 0.5).sum().item())
                           / (4.0 * np.pi)) ** (1.0 / 3.0)

    _record(0)
    f, gamma, tau = cfg.f, cfg.gamma, cfg.damage_decay_time
    for i in range(1, n_out):
        for s in range(cfg.substeps):
            frac = (s + 0.5) / cfg.substeps
            U_now = float(np.interp(t_out[i - 1] + frac * cfg.dt_output,
                                    t_out, U_out))
            SP = _diffusion(A, D)
            A = A + (SP + f * A * (1.0 - A) - gamma * Z * A) * dt
            A = torch.clamp(A, 0.0, 1.0) * brain_f
            Z = Z + (-Z / tau + U_now * beam) * dt
            Z = torch.clamp(Z, min=0.0)
        _record(i)

    m0 = mass_abs[0] if mass_abs[0] > 0 else 1.0
    normalized = mass_abs / m0
    dips = []
    for t0 in cfg.dose_times:
        pre = float(np.interp(t0, t_out, normalized))
        win = (t_out >= t0) & (t_out <= min(t0 + 15.0, cfg.T))
        if win.any() and pre > 0:
            dips.append(round(1.0 - float(normalized[win].min()) / pre, 3))
    info = {
        "Dw": cfg.Dw, "f": cfg.f, "Dw_ratio": cfg.Dw_ratio,
        "gamma": cfg.gamma, "tau": cfg.damage_decay_time,
        "dose_times": list(cfg.dose_times), "dose_amplitude": cfg.dose_amplitude,
        "grid": int(shape[0]), "n_steps": int((n_out - 1) * cfg.substeps),
        "dt_internal": dt, "dt_stable": dt_stable,
        "mass0": float(m0), "mass_final": float(mass_abs[-1]),
        "peak_mass_ratio": float(normalized.max()),
        "dose_knockdowns": dips,
        "brain_voxels": float(brain_f.sum().item()),
        "fisher_speed": float(2.0 * np.sqrt(max(cfg.Dw * cfg.f, 0.0))),
        "device": str(dev), "dtype": str(dtype),
    }
    return FK3DGrowthResult(t_out, normalized, mass_abs, front_radius, info,
                            U_t=U_out, W_eff=W_eff)
