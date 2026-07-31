"""GPU forward model: GliODIL's growth kernel + a radiotherapy damage field.

Ground-truth generator for the surrogate study.  The tumour field obeys the
anisotropic Fisher--Kolmogorov equation GliODIL infers, augmented with the
spatial analogue of the paper's reduced two-state radiotherapy model::

    dA/dt = div(D(x) grad A) + f A (1 - A) - gamma_eff(A) Z A
    dZ/dt = -Z / tau + U(t) * beam(x)
    gamma_eff(A) = gamma * (1 - h A)        (hypoxic-core radioresistance)

``D(x)`` is GliODIL's white-matter-preferential face-centred coefficient
(``m_Tildas`` / ``get_D`` in ``synthetic_generator.py``), so the growth physics
is identical to what GliODIL infers on these patients.

The ``h`` term is the one deliberate addition: it makes the radiation response
*spatially heterogeneous in a way the 0-D surrogate cannot represent*, which is
exactly the ``M(u, x, t)`` / ``h(x, t)`` microenvironment channel of the paper's
general model (Eq. pde_general) and the reason a closure term is needed at all.
Set ``hypoxia=0`` to recover the plain graft.

Everything runs on the GPU and only the reduced scalar series are copied back,
so a full 80-unit, 801-sample trajectory on a 160^3 grid costs ~1-3 s.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from .patient import TH_UP


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class RTConfig:
    """Forward-simulation settings (solver/voxel units; time in model units)."""

    # growth (per patient, from realdata.patient.derive_growth)
    Dw: float = 0.30
    f: float = 0.05
    Dw_ratio: float = 20.0
    matter_threshold: float = 0.1

    # time axis -- the paper's [0, 80] horizon at dt = 0.1
    T: float = 80.0
    dt_output: float = 0.1
    substeps: int = 3

    # radiotherapy schedule (the paper's two irradiation events)
    dose_times: Tuple[float, ...] = (15.0, 45.0)
    dose_amplitude: float = 1.2
    dose_duration: float = 0.6
    dose_edge_fraction: float = 0.25
    gamma: float = 1.25          # radiosensitivity coefficient (paper: beta)
    tau: float = 3.0             # damage relaxation time
    hypoxia: float = 0.5         # h: density-dependent radioresistance

    # diagnostics / figures
    snapshot_times: Tuple[float, ...] = ()
    keep_volumes: bool = False   # also return the full 3-D field at snapshots


@dataclass
class RTResult:
    t: np.ndarray                 # (n_out,)
    U_t: np.ndarray               # (n_out,) scalar dose rate
    mass_total: np.ndarray        # (n_out,) integral of A  (the paper's y)
    mass_visible: np.ndarray      # (n_out,) volume with A >= TH_UP (MRI-visible)
    W_eff: np.ndarray             # (n_out,) mass-weighted mean damage (0-D z)
    R_eff: np.ndarray             # (n_out,) equivalent radius of the visible bulk
    beam_coverage: np.ndarray     # (n_out,) fraction of tumour mass inside the beam
    planes: Dict[float, Dict[str, np.ndarray]]   # cross-sections at snapshots
    volumes: Dict[float, np.ndarray]             # full fields (if requested)
    info: Dict[str, object] = field(default_factory=dict)

    @property
    def normalized_mass(self) -> np.ndarray:
        m0 = self.mass_total[0] if self.mass_total[0] > 0 else 1.0
        return self.mass_total / m0


# ---------------------------------------------------------------------------
# GliODIL diffusion kernel (torch port; identical stencil to fk3d.py)
# ---------------------------------------------------------------------------
def _roll(x: torch.Tensor, s: int, ax: int) -> torch.Tensor:
    return torch.roll(x, shifts=s, dims=ax)


def _get_D(WM: torch.Tensor, GM: torch.Tensor, th: float, Dw: float,
           ratio: float) -> Dict[str, torch.Tensor]:
    D = {}
    for ax in range(3):
        cond = ((_roll(WM, -1, ax) + _roll(GM, -1, ax) >= th) &
                (WM + GM >= th)).to(WM.dtype)
        wm_t = cond * (_roll(WM, -1, ax) + WM) / 2.0
        gm_t = cond * (_roll(GM, -1, ax) + GM) / 2.0
        D[f"minus_{ax}"] = Dw * (wm_t + gm_t / ratio)
        D[f"plus_{ax}"] = Dw * (_roll(wm_t, 1, ax) + _roll(gm_t, 1, ax) / ratio)
    return D


def _diffusion(A: torch.Tensor, D: Dict[str, torch.Tensor]) -> torch.Tensor:
    out = torch.zeros_like(A)
    for ax in range(3):
        out = out + (D[f"plus_{ax}"] * (_roll(A, 1, ax) - A)
                     - D[f"minus_{ax}"] * (A - _roll(A, -1, ax)))
    return out


def dose_rate(t: np.ndarray, cfg: RTConfig) -> np.ndarray:
    """Smooth trapezoidal dose-rate pulse train U(t)."""
    t = np.asarray(t, dtype=float)
    U = np.zeros_like(t)
    half = 0.5 * cfg.dose_duration
    edge = max(cfg.dose_edge_fraction * cfg.dose_duration, 1e-9)
    for t0 in cfg.dose_times:
        local = t - t0
        U = U + cfg.dose_amplitude * (
            0.5 * (1.0 + np.tanh((local + half) / edge)) *
            0.5 * (1.0 + np.tanh((half - local) / edge)))
    return U


# ---------------------------------------------------------------------------
# Forward simulation
# ---------------------------------------------------------------------------
def simulate(WM, GM, brain, A0, beam, cfg: RTConfig, device: str = "cuda",
             dtype: torch.dtype = torch.float32,
             centroid: Optional[Sequence[float]] = None) -> RTResult:
    """Integrate the tumour + damage fields and reduce them to scalar series.

    ``beam`` may be ``None`` (radiation off).  All spatial arrays share the same
    3-D shape; ``A0`` is the initial cell density.
    """
    dev = torch.device(device)
    to = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=dev)
    WM_t, GM_t = to(WM), to(GM)
    brain_f = to(np.asarray(brain, dtype=np.float32))
    A = to(A0) * brain_f
    rt_on = beam is not None
    beam_t = (to(beam) * brain_f) if rt_on else None
    Z = torch.zeros_like(A)

    D = _get_D(WM_t, GM_t, cfg.matter_threshold, cfg.Dw, cfg.Dw_ratio)

    n_out = int(round(cfg.T / cfg.dt_output)) + 1
    t_out = np.linspace(0.0, cfg.T, n_out)
    U_out = dose_rate(t_out, cfg) if rt_on else np.zeros(n_out)

    dt = cfg.dt_output / cfg.substeps
    d_max = max(float(D[f"plus_{ax}"].max()) for ax in range(3))
    dt_stable = 1.0 / (6.0 * max(d_max, 1e-12))
    if dt > dt_stable:
        raise ValueError(
            f"unstable: dt={dt:.4g} > {dt_stable:.4g}; need substeps >= "
            f"{int(np.ceil(cfg.dt_output / dt_stable))} (Dw={cfg.Dw:.3g})")

    # dose rate sampled at every internal substep (midpoint rule)
    n_int = (n_out - 1) * cfg.substeps
    t_int = (np.arange(n_int, dtype=float) + 0.5) * dt
    U_int = torch.as_tensor(dose_rate(t_int, cfg) if rt_on else np.zeros(n_int),
                            dtype=dtype, device=dev)

    # scalar diagnostics accumulate on-device; one host transfer at the end
    stats = torch.zeros((n_out, 5), dtype=torch.float64, device=dev)
    snap_times = sorted(set(float(s) for s in cfg.snapshot_times))
    snap_idx = {int(round(s / cfg.dt_output)): s for s in snap_times}
    planes: Dict[float, Dict[str, np.ndarray]] = {}
    volumes: Dict[float, np.ndarray] = {}

    if centroid is None:
        centroid = [s / 2 for s in A.shape]
    ci = [int(np.clip(round(float(c)), 0, s - 1)) for c, s in zip(centroid, A.shape)]

    def _record(i: int):
        m = A.sum()
        stats[i, 0] = m
        stats[i, 1] = (A >= TH_UP).sum()
        stats[i, 2] = torch.where(m > 0, (Z * A).sum() / m, torch.zeros_like(m))
        if rt_on:
            stats[i, 3] = torch.where(m > 0, (beam_t * A).sum() / m,
                                      torch.zeros_like(m))
        stats[i, 4] = Z.max()
        if i in snap_idx:
            tt = snap_idx[i]
            planes[tt] = {
                "A_axial": A[:, :, ci[2]].detach().to(torch.float32).cpu().numpy(),
                "A_coronal": A[:, ci[1], :].detach().to(torch.float32).cpu().numpy(),
                "A_sagittal": A[ci[0], :, :].detach().to(torch.float32).cpu().numpy(),
                "Z_axial": Z[:, :, ci[2]].detach().to(torch.float32).cpu().numpy(),
            }
            if cfg.keep_volumes:
                volumes[tt] = A.detach().to(torch.float16).cpu().numpy()

    _record(0)
    f, gamma, tau, h = cfg.f, cfg.gamma, cfg.tau, cfg.hypoxia
    k = 0
    for i in range(1, n_out):
        for _ in range(cfg.substeps):
            SP = _diffusion(A, D)
            if rt_on:
                kill = gamma * (1.0 - h * A) * Z * A
                A = torch.clamp(A + (SP + f * A * (1.0 - A) - kill) * dt,
                                0.0, 1.0) * brain_f
                Z = torch.clamp(Z + (-Z / tau + U_int[k] * beam_t) * dt, min=0.0)
            else:
                A = torch.clamp(A + (SP + f * A * (1.0 - A)) * dt,
                                0.0, 1.0) * brain_f
            k += 1
        _record(i)

    s = stats.cpu().numpy()
    mass_total, mass_vis, W_eff, cover, zmax = (s[:, j] for j in range(5))
    R_eff = np.cbrt(3.0 * np.maximum(mass_vis, 0.0) / (4.0 * np.pi))

    m0 = mass_total[0] if mass_total[0] > 0 else 1.0
    norm = mass_total / m0
    dips = []
    for t0 in cfg.dose_times if rt_on else ():
        pre = float(np.interp(t0, t_out, norm))
        win = (t_out >= t0) & (t_out <= min(t0 + 15.0, cfg.T))
        if win.any() and pre > 0:
            dips.append(round(1.0 - float(norm[win].min()) / pre, 4))

    info = {
        "Dw": cfg.Dw, "f": cfg.f, "Dw_ratio": cfg.Dw_ratio, "gamma": cfg.gamma,
        "tau": cfg.tau, "hypoxia": cfg.hypoxia, "rt": bool(rt_on),
        "grid": list(A.shape), "dt_internal": dt, "dt_stable": dt_stable,
        "mass0": float(m0), "peak_mass_ratio": float(norm.max()),
        "final_mass_ratio": float(norm[-1]), "dose_knockdowns": dips,
        "z_max": float(zmax.max()),
        "nominal_fisher_speed": float(2.0 * np.sqrt(max(cfg.Dw * cfg.f, 0.0))),
        "centroid_vox": [int(c) for c in ci],
    }
    return RTResult(t_out, U_out, mass_total, mass_vis, W_eff, R_eff, cover,
                    planes, volumes, info)


# ---------------------------------------------------------------------------
# NumPy reference (validation only -- same equations, no GPU)
# ---------------------------------------------------------------------------
def simulate_reference(WM, GM, brain, A0, beam, cfg: RTConfig) -> RTResult:
    """Pure-NumPy float64 mirror of :func:`simulate`, for correctness checks."""
    WM = np.asarray(WM, np.float64)
    GM = np.asarray(GM, np.float64)
    brain_f = np.asarray(brain, np.float64)
    A = np.asarray(A0, np.float64) * brain_f
    rt_on = beam is not None
    beam_a = np.asarray(beam, np.float64) * brain_f if rt_on else None
    Z = np.zeros_like(A)

    D = {}
    for ax in range(3):
        cond = ((np.roll(WM, -1, ax) + np.roll(GM, -1, ax) >= cfg.matter_threshold) &
                (WM + GM >= cfg.matter_threshold)).astype(np.float64)
        wm_t = cond * (np.roll(WM, -1, ax) + WM) / 2.0
        gm_t = cond * (np.roll(GM, -1, ax) + GM) / 2.0
        D[f"minus_{ax}"] = cfg.Dw * (wm_t + gm_t / cfg.Dw_ratio)
        D[f"plus_{ax}"] = cfg.Dw * (np.roll(wm_t, 1, ax) + np.roll(gm_t, 1, ax) / cfg.Dw_ratio)

    n_out = int(round(cfg.T / cfg.dt_output)) + 1
    t_out = np.linspace(0.0, cfg.T, n_out)
    dt = cfg.dt_output / cfg.substeps
    n_int = (n_out - 1) * cfg.substeps
    U_int = dose_rate((np.arange(n_int) + 0.5) * dt, cfg) if rt_on else np.zeros(n_int)
    U_out = dose_rate(t_out, cfg) if rt_on else np.zeros(n_out)

    mass = np.empty(n_out); vis = np.empty(n_out); W = np.empty(n_out)
    cov = np.zeros(n_out)

    def rec(i):
        m = A.sum(); mass[i] = m; vis[i] = float((A >= TH_UP).sum())
        W[i] = (Z * A).sum() / m if m > 0 else 0.0
        if rt_on and m > 0:
            cov[i] = (beam_a * A).sum() / m

    rec(0)
    k = 0
    for i in range(1, n_out):
        for _ in range(cfg.substeps):
            SP = np.zeros_like(A)
            for ax in range(3):
                SP += (D[f"plus_{ax}"] * (np.roll(A, 1, ax) - A)
                       - D[f"minus_{ax}"] * (A - np.roll(A, -1, ax)))
            if rt_on:
                kill = cfg.gamma * (1.0 - cfg.hypoxia * A) * Z * A
                A = np.clip(A + (SP + cfg.f * A * (1 - A) - kill) * dt, 0, 1) * brain_f
                Z = np.clip(Z + (-Z / cfg.tau + U_int[k] * beam_a) * dt, 0, None)
            else:
                A = np.clip(A + (SP + cfg.f * A * (1 - A)) * dt, 0, 1) * brain_f
            k += 1
        rec(i)

    R = np.cbrt(3.0 * vis / (4.0 * np.pi))
    return RTResult(t_out, U_out, mass, vis, W, R, cov, {}, {},
                    {"reference": True, "mass0": float(mass[0])})
