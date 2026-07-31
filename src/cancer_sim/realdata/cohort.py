"""Build the ground-truth cohort: real patients x the paper's four beam configs.

Per patient this produces, on one GPU:

  * a **no-treatment counterfactual** trajectory (radiation off),
  * four **radiotherapy** trajectories, one per tumour--beam alignment scenario,
  * cross-sections of the tumour and damage fields at snapshot times (figures),
  * a manifest with the imaging-derived parameters and QC numbers.

Growth-rate calibration
-----------------------
The Fisher--Kolmogorov equation is exactly invariant under a joint rescaling of
the rates and time: if ``A(x, t)`` solves it for ``(D, f)`` then ``A(x, c t)``
solves it for ``(cD, cf)``.  We exploit this to fix the one quantity a single
imaging timepoint cannot identify -- the absolute speed -- without extra
simulations: one reference run per patient on a long horizon gives the whole
one-parameter family, and we pick the scale ``c`` that makes the *untreated*
tumour reach a per-patient target burden at the end of the horizon.

The *shape* parameter ``L = sqrt(D/f)`` stays imaging-derived (the measured
edema-rim thickness of that patient), so infiltrative and nodular tumours remain
genuinely different; only the timescale is normalised.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import patient as P
from .fk_rt_gpu import RTConfig, RTResult, simulate

# Per-patient untreated burden at the horizon (mass ratio).  Drawn per patient
# so the cohort keeps a spread of aggressiveness after timescale normalisation.
TARGET_RATIO_RANGE = (2.0, 5.0)
F_REF = 0.05
T_REF = 400.0

SNAPSHOTS = (0.0, 15.0, 18.0, 30.0, 45.0, 48.0, 60.0, 80.0)


# ---------------------------------------------------------------------------
# Cohort screening
# ---------------------------------------------------------------------------
QC = {
    "min_core_mm3": 3000.0,     # ~9 mm equivalent radius
    "min_whole_mm3": 8000.0,
    "rim_mm": (1.5, 20.0),
    "brain_voxels": (8.0e5, 2.6e6),
    "edge_margin_frac": 0.06,   # tumour centroid must not hug the volume edge
}


def screen_patient(pid: str, root: Path) -> Dict:
    """Cheap native-resolution QC for one patient (no resampling, no GPU)."""
    import nibabel as nib
    pdir = Path(root) / pid
    seg = np.asarray(nib.load(str(pdir / "segm.nii.gz")).get_fdata())
    wm = np.asarray(nib.load(str(pdir / "t1_wm.nii.gz")).get_fdata())
    gm = np.asarray(nib.load(str(pdir / "t1_gm.nii.gz")).get_fdata())
    core = np.isin(seg, P._LABEL_CORE)
    whole = seg > 0
    Vc, Vw = float(core.sum()), float(whole.sum())
    R = lambda v: (3.0 * v / (4.0 * np.pi)) ** (1.0 / 3.0) if v > 0 else 0.0
    rim = R(Vw) - R(Vc)
    brain = float(((wm + gm) > 0.1).sum())
    cen = (np.array(np.nonzero(whole), float).mean(axis=1) / np.array(seg.shape)
           if Vw > 0 else np.array([0.5, 0.5, 0.5]))
    m = QC["edge_margin_frac"]
    reasons = []
    if Vc < QC["min_core_mm3"]:
        reasons.append("core_too_small")
    if Vw < QC["min_whole_mm3"]:
        reasons.append("whole_too_small")
    if not (QC["rim_mm"][0] <= rim <= QC["rim_mm"][1]):
        reasons.append("rim_out_of_range")
    if not (QC["brain_voxels"][0] <= brain <= QC["brain_voxels"][1]):
        reasons.append("bad_tissue_map")
    if np.any(cen < m) or np.any(cen > 1 - m):
        reasons.append("tumour_at_edge")
    return {"pid": pid, "V_core_mm3": Vc, "V_whole_mm3": Vw, "rim_mm": rim,
            "R_core_mm": R(Vc), "R_whole_mm": R(Vw), "brain_voxels": brain,
            "centroid_frac": [round(float(c), 4) for c in cen],
            "pass": not reasons, "reasons": reasons}


def screen_cohort(root: Path, workers: int = 24) -> List[Dict]:
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    pids = P.cohort(root)
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(partial(screen_patient, root=root), pids))


def _substeps_for(Dw: float, dt_output: float, safety: float = 1.6) -> int:
    """Smallest substep count keeping the explicit diffusion update stable."""
    dt_stable = 1.0 / (6.0 * max(Dw, 1e-9))
    return max(1, int(np.ceil(safety * dt_output / dt_stable)))


def calibrate_growth(geom: P.PatientGeometry, device: str,
                     target_ratio: Optional[float] = None,
                     T: float = 80.0) -> Dict[str, float]:
    """Pick ``(Dw, f)`` for one patient (see module docstring).

    Returns the calibrated coefficients plus the reference curve used, so the
    calibration is auditable.
    """
    L_mm = float(np.clip(geom.L_mm, *P.L_CLIP_MM))
    L_vox = L_mm / geom.voxel_mm
    rng = P.patient_rng(geom.pid, "growth")
    target = float(target_ratio if target_ratio is not None
                   else rng.uniform(*TARGET_RATIO_RANGE))
    ratio_wm_gm = float(rng.uniform(*P.RATIO_RANGE))

    D_ref = F_REF * L_vox * L_vox
    dt_out = 0.5
    cfg = RTConfig(Dw=D_ref, f=F_REF, Dw_ratio=ratio_wm_gm, T=T_REF,
                   dt_output=dt_out, substeps=_substeps_for(D_ref, dt_out),
                   dose_times=())
    res = simulate(geom.WM, geom.GM, geom.brain, geom.A0, None, cfg, device=device)
    nm = res.normalized_mass
    reached = bool(nm.max() >= target)
    if reached:
        t_star = float(np.interp(target, nm, res.t))
    else:                       # brain-limited: use the horizon we can reach
        t_star = float(res.t[-1])
        target = float(nm[-1])
    # A_c(x, t) = A_ref(x, c t); we want A_c at t = T to be A_ref at t_star.
    scale = t_star / T
    return {
        "Dw": D_ref * scale, "f": F_REF * scale, "Dw_ratio": ratio_wm_gm,
        "L_vox": L_vox, "L_mm": L_mm, "L_raw_mm": float(geom.L_mm),
        "target_ratio": target, "target_reached": reached,
        "time_scale": scale, "t_star_ref": t_star,
        "front_speed_vox": float(2.0 * np.sqrt(D_ref * F_REF) * scale),
        "front_speed_mm": float(2.0 * np.sqrt(D_ref * F_REF) * scale * geom.voxel_mm),
        "ref_curve_t": res.t[::10].tolist(), "ref_curve_m": nm[::10].tolist(),
    }


def _series(res: RTResult) -> Dict[str, np.ndarray]:
    return {
        "t": res.t.astype(np.float32),
        "mass": res.normalized_mass.astype(np.float32),
        "mass_visible": (res.mass_visible /
                         max(res.mass_visible[0], 1.0)).astype(np.float32),
        "U": res.U_t.astype(np.float32),
        "W": res.W_eff.astype(np.float32),
        "R_eff": res.R_eff.astype(np.float32),
        "coverage": res.beam_coverage.astype(np.float32),
    }


def generate_patient(pid: str, root: Path, out_dir: Path, device: str,
                     grid: int = 160, T: float = 80.0, dt_output: float = 0.1,
                     rt: Optional[RTConfig] = None,
                     configs: Sequence[P.BeamConfig] = P.BEAM_CONFIGS,
                     snapshots: Sequence[float] = SNAPSHOTS,
                     keep_volumes: bool = False) -> Dict:
    """Run one patient's whole scenario set and write it to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    geom = P.load_geometry(pid, root, grid=grid)
    gp = calibrate_growth(geom, device, T=T)
    base = rt or RTConfig()
    # Radiosensitivity and hypoxic radioresistance vary between patients; the
    # reduced surrogates are not told them, they must be assimilated.
    rrng = P.patient_rng(pid, "rt")
    gamma = float(rrng.uniform(*P.GAMMA_RANGE))
    hypoxia = float(rrng.uniform(*P.HYPOXIA_RANGE))
    cfg = replace(base, Dw=gp["Dw"], f=gp["f"], Dw_ratio=gp["Dw_ratio"],
                  gamma=gamma, hypoxia=hypoxia,
                  T=T, dt_output=dt_output,
                  substeps=_substeps_for(gp["Dw"], dt_output),
                  snapshot_times=tuple(snapshots), keep_volumes=keep_volumes)

    curves: Dict[str, Dict[str, np.ndarray]] = {}
    planes: Dict[str, Dict] = {}
    vols: Dict[str, Dict] = {}
    per_case: Dict[str, Dict] = {}

    # untreated counterfactual
    res0 = simulate(geom.WM, geom.GM, geom.brain, geom.A0, None, cfg,
                    device=device, centroid=geom.centroid_vox)
    curves["no_treatment"] = _series(res0)
    planes["no_treatment"] = res0.planes
    per_case["no_treatment"] = {"final_mass_ratio": res0.info["final_mass_ratio"],
                                "peak_mass_ratio": res0.info["peak_mass_ratio"]}

    beams: Dict[str, np.ndarray] = {}
    for bc in configs:
        beam = P.beam_field(geom, bc)
        beams[bc.name] = beam
        res = simulate(geom.WM, geom.GM, geom.brain, geom.A0, beam, cfg,
                       device=device, centroid=geom.centroid_vox)
        curves[bc.name] = _series(res)
        planes[bc.name] = res.planes
        if keep_volumes:
            vols[bc.name] = res.volumes
        per_case[bc.name] = {
            "final_mass_ratio": res.info["final_mass_ratio"],
            "peak_mass_ratio": res.info["peak_mass_ratio"],
            "dose_knockdowns": res.info["dose_knockdowns"],
            "beam_coverage0": float(res.beam_coverage[0]),
            "nadir": float(res.normalized_mass.min()),
        }

    # ---- persist -----------------------------------------------------------
    flat = {}
    for case, s in curves.items():
        for k, v in s.items():
            flat[f"{case}/{k}"] = v
    np.savez_compressed(out_dir / f"{pid}_curves.npz", **flat)

    ci = [int(np.clip(round(c), 0, s - 1))
          for c, s in zip(geom.centroid_vox, geom.shape)]
    anat = {
        "wm_axial": geom.WM[:, :, ci[2]], "gm_axial": geom.GM[:, :, ci[2]],
        "wm_coronal": geom.WM[:, ci[1], :], "gm_coronal": geom.GM[:, ci[1], :],
        "wm_sagittal": geom.WM[ci[0], :, :], "gm_sagittal": geom.GM[ci[0], :, :],
        "seg_axial": (geom.core[:, :, ci[2]].astype(np.uint8)
                      + geom.whole[:, :, ci[2]].astype(np.uint8)),
        "rec_axial": geom.recurrence[:, :, ci[2]].astype(np.uint8),
        "brain_axial": geom.brain[:, :, ci[2]].astype(np.uint8),
    }
    for name, b in beams.items():
        anat[f"beam_{name}_axial"] = b[:, :, ci[2]].astype(np.float32)
        anat[f"beam_{name}_coronal"] = b[:, ci[1], :].astype(np.float32)
    snap_flat = {}
    for case, per_t in planes.items():
        for tt, d in per_t.items():
            for k, v in d.items():
                snap_flat[f"{case}/{tt:g}/{k}"] = v.astype(np.float16)
    np.savez_compressed(out_dir / f"{pid}_fields.npz",
                        **{f"anat/{k}": v for k, v in anat.items()}, **snap_flat)
    if keep_volumes:
        vf = {f"{c}/{tt:g}": v for c, per_t in vols.items() for tt, v in per_t.items()}
        np.savez_compressed(out_dir / f"{pid}_volumes.npz", **vf)

    manifest = {
        "pid": pid, "grid": grid, "voxel_mm": geom.voxel_mm,
        "shape": list(geom.shape), "centroid_vox": ci,
        "R_core_mm": geom.R_core_mm, "R_whole_mm": geom.R_whole_mm,
        "rim_mm": geom.rim_mm, "L_raw_mm": geom.L_mm,
        "growth": {k: v for k, v in gp.items() if not k.startswith("ref_curve")},
        "rt": {"gamma": cfg.gamma, "tau": cfg.tau, "hypoxia": cfg.hypoxia,
               "dose_times": list(cfg.dose_times),
               "dose_amplitude": cfg.dose_amplitude,
               "dose_duration": cfg.dose_duration,
               "substeps": cfg.substeps, "dt_internal": cfg.dt_output / cfg.substeps},
        "anatomy": geom.info, "cases": per_case,
        "wall_time_s": round(time.time() - t_start, 2), "device": device,
    }
    (out_dir / f"{pid}_manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def load_curves(pid: str, out_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """Read back one patient's curve bundle as ``{case: {series: array}}``."""
    z = np.load(Path(out_dir) / f"{pid}_curves.npz")
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for key in z.files:
        case, series = key.split("/")
        out.setdefault(case, {})[series] = z[key]
    return out
