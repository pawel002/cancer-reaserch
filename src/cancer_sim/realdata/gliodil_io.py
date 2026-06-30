"""Readers that turn GliODIL inputs/outputs into :mod:`fk3d` solver inputs.

Two entry points:

* :func:`synthetic_patient` -- loads the WM/GM atlas shipped in the GliODIL
  repo (``precomputed/s{WM,GM}_192_192_192.npy``) and returns a fabricated but
  GliODIL-format "patient".  Needs no GPU, no download, no ``nibabel``; used by
  the end-to-end CPU demo.

* :func:`load_patient` -- loads a *real* GliODIL patient: WM/GM tissue maps
  (``.nii``/``.nii.gz`` or ``.npy``) plus the growth parameters GliODIL inferred
  (``coeffs.npy``), mapping them onto an :class:`fk3d.FK3DConfig`.

``nibabel`` is imported lazily so the demo and all NumPy paths work without it;
install it (``pip install nibabel``) only to read real ``.nii`` volumes.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .fk3d import FK3DConfig

# Atlas shipped inside the cloned GliODIL repo.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ATLAS_DIR = _REPO_ROOT / "GliODIL" / "code" / "cases" / "GliODIL" / "precomputed"


# ----------------------------------------------------------------------------
# Volume loading
# ----------------------------------------------------------------------------
def _load_volume(path: Path) -> np.ndarray:
    """Load a 3-D volume from .npy or NIfTI (.nii/.nii.gz)."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path).astype(np.float64)
    try:
        import nibabel as nib  # lazy: only needed for real .nii patients
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            f"Reading {path.name} needs nibabel. Install it with "
            "`.venv/bin/python -m pip install nibabel`, or pre-convert the "
            "volumes to .npy.") from e
    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float64)


def _resample(vol: np.ndarray, grid: Optional[int]) -> np.ndarray:
    """Downsample so the largest axis is ``grid`` voxels.

    Uses a *single* scale factor for all axes, so physically-isotropic MRI voxels
    stay isotropic (real GliODIL volumes are 240x240x155, not cubic).  The FK
    solver assumes unit grid spacing, so isotropic voxels are what it expects.
    """
    if grid is None:
        return vol
    factor = grid / max(vol.shape)
    if abs(factor - 1.0) < 1e-6:
        return vol
    from scipy.ndimage import zoom
    return zoom(vol, factor, order=1)


def _normalise_tissue(WM: np.ndarray, GM: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Clamp tissue maps to [0, 1] (probability semantics)."""
    return np.clip(WM, 0.0, 1.0), np.clip(GM, 0.0, 1.0)


# ----------------------------------------------------------------------------
# Synthetic patient (CPU demo)
# ----------------------------------------------------------------------------
def synthetic_patient(grid: Optional[int] = 96,
                      base: Optional[FK3DConfig] = None,
                      variant: int = 0) -> Tuple[np.ndarray, np.ndarray, FK3DConfig]:
    """Fabricate a GliODIL-format patient from the repo's WM/GM atlas.

    Returns ``(WM, GM, cfg)``.  ``variant`` perturbs the growth parameters
    within GliODIL's synthetic sampling ranges (run_synthetic_generator1T.py)
    so several distinct demo "patients" can be produced deterministically.
    """
    wm_path = _ATLAS_DIR / "sWM_192_192_192.npy"
    gm_path = _ATLAS_DIR / "sGM_192_192_192.npy"
    if not wm_path.exists():
        raise FileNotFoundError(
            f"GliODIL atlas not found at {wm_path}. Expected the cloned GliODIL "
            "repo under ./GliODIL.")
    WM = _resample(_load_volume(wm_path), grid)
    GM = _resample(_load_volume(gm_path), grid)
    WM, GM = _normalise_tissue(WM, GM)

    # GliODIL synthetic ranges: Dw,rho in [0.035,0.2]; ratio in [10,30].
    rng = np.random.default_rng(1000 + variant)
    Dw = float(rng.uniform(0.08, 0.20))
    f = float(rng.uniform(0.06, 0.14))
    ratio = float(rng.uniform(10.0, 30.0))
    # Seed somewhere in the white matter (brightest WM voxel near the interior).
    interior = WM.copy()
    pad = max(WM.shape[0] // 6, 1)
    interior[:pad] = interior[-pad:] = 0
    interior[:, :pad] = interior[:, -pad:] = 0
    interior[:, :, :pad] = interior[:, :, -pad:] = 0
    sx, sy, sz = np.unravel_index(int(np.argmax(interior)), WM.shape)
    seed = (float(sx) / WM.shape[0], float(sy) / WM.shape[1], float(sz) / WM.shape[2])

    cfg = base or FK3DConfig()
    cfg = replace(cfg, Dw=Dw, f=f, Dw_ratio=ratio,
                  seed_center=seed, beam_center=seed)
    return WM, GM, cfg


# ----------------------------------------------------------------------------
# Real patient
# ----------------------------------------------------------------------------
# GliODIL coeffs.npy layout (see GliODIL.py output spec):
#   [D, f, x0, y0, z0, s, th_up, th_down, log10(Dw_ratio), pet_bkg_lvl]
def params_from_coeffs(coeffs: np.ndarray, grid_shape) -> dict:
    """Map a GliODIL ``coeffs.npy`` vector to FK3DConfig overrides.

    Seed coordinates ``(x0, y0, z0)`` are interpreted as fractions of the grid
    if they lie in [0, 1], else as absolute voxel indices on ``grid_shape``.
    """
    coeffs = np.asarray(coeffs, dtype=float).ravel()
    if coeffs.size < 9:
        raise ValueError(f"coeffs.npy has {coeffs.size} entries; expected >= 9")
    x0, y0, z0 = coeffs[2], coeffs[3], coeffs[4]
    if not (0.0 <= x0 <= 1.0 and 0.0 <= y0 <= 1.0 and 0.0 <= z0 <= 1.0):
        x0, y0, z0 = x0 / grid_shape[0], y0 / grid_shape[1], z0 / grid_shape[2]
    seed = (float(np.clip(x0, 0, 1)), float(np.clip(y0, 0, 1)),
            float(np.clip(z0, 0, 1)))
    return {
        "Dw": float(coeffs[0]),
        "f": float(coeffs[1]),
        "Dw_ratio": float(10.0 ** coeffs[8]),
        "seed_center": seed,
        "beam_center": seed,
    }


def load_patient(patient_dir: Path, grid: Optional[int] = 96,
                 base: Optional[FK3DConfig] = None
                 ) -> Tuple[np.ndarray, np.ndarray, FK3DConfig]:
    """Load a real GliODIL patient directory into ``(WM, GM, cfg)``.

    Looks for white-/grey-matter maps named ``*_wm_*`` / ``*_gm_*`` (GliODIL
    convention; ``.nii``/``.nii.gz``/``.npy``) and, if present, ``coeffs.npy``
    with GliODIL's inferred growth parameters.  Without ``coeffs.npy`` the
    growth defaults from ``base``/:class:`FK3DConfig` are used (and should be set
    from GliODIL's reported values).
    """
    patient_dir = Path(patient_dir)

    def _find(include, exclude=()):
        # Matches both GliODIL's synthetic naming (*_wm_*, *_gm_*) and the real
        # dataset's (t1_wm, t1_gm, segm, segm_rec, FET). Skips macOS ._ forks.
        for p in sorted(patient_dir.iterdir()):
            name = p.name.lower()
            if name.startswith("._") or p.suffix not in (".nii", ".gz", ".npy"):
                continue
            if include in name and not any(x in name for x in exclude):
                return p
        return None

    # "segm" contains "gm", so exclude segmentation files when matching grey matter.
    wm_path = _find("_wm", exclude=())
    gm_path = _find("_gm", exclude=("segm",))
    if wm_path is None or gm_path is None:
        raise FileNotFoundError(
            f"Could not find white/grey-matter volumes (*_wm*, *_gm*) in {patient_dir}")

    WM = _resample(_load_volume(wm_path), grid)
    GM = _resample(_load_volume(gm_path), grid)
    WM, GM = _normalise_tissue(WM, GM)

    cfg = base or FK3DConfig()

    # Patient-specific growth parameters: prefer GliODIL's inferred coeffs.npy;
    # otherwise keep the (literature-range) defaults from `base`.
    coeffs_path = patient_dir / "coeffs.npy"
    if coeffs_path.exists():
        cfg = replace(cfg, **params_from_coeffs(np.load(coeffs_path), WM.shape))
    else:
        # No GliODIL run available: at least anchor the focal seed (and beam) at
        # the patient's REAL tumour location from the pre-op segmentation.
        seed = tumor_centroid(patient_dir)
        if seed is not None:
            cfg = replace(cfg, seed_center=seed, beam_center=seed)
    return WM, GM, cfg


def tumor_centroid(patient_dir: Path) -> Optional[tuple]:
    """Fractional (x, y, z) centroid of the pre-op tumour segmentation.

    Reads ``segm*`` but not the recurrence ``*segm_rec*``; returns None if no
    pre-op segmentation is present.  Resolution-independent (a fraction), so it
    is computed on the native volume before any resampling.
    """
    patient_dir = Path(patient_dir)
    seg_path = None
    for p in sorted(patient_dir.iterdir()):
        name = p.name.lower()
        if name.startswith("._") or p.suffix not in (".nii", ".gz", ".npy"):
            continue
        if "segm" in name and "rec" not in name:
            seg_path = p
            break
    if seg_path is None:
        return None
    seg = _load_volume(seg_path)
    mask = seg > 0
    if not mask.any():
        return None
    idx = np.array(np.where(mask), dtype=float)
    centroid = idx.mean(axis=1)
    return tuple(float(c) / s for c, s in zip(centroid, seg.shape))
