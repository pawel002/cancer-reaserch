"""Real GliODIL patients -> anatomy, tumour-derived initial density, and beam configs.

This module replaces the earlier "generic Gaussian seed + sampled growth rates"
bridge (``gliodil_io`` + ``fk3d``) with a construction in which *everything that
can be read off the patient's imaging is read off the patient's imaging*:

======================  ==================================================
quantity                where it comes from
======================  ==================================================
anatomy (WM/GM/CSF)     the patient's tissue probability maps
tumour location+shape   the patient's pre-operative BraTS segmentation
initial cell density    two-threshold reconstruction of that segmentation,
                        using GliODIL's own density<->label convention
invasiveness  D/f       the measured edema-rim thickness of *this* tumour
irradiation field       a conformal PTV grown from the real tumour mask
======================  ==================================================

Only one scalar per patient is *not* identifiable from a single imaging
timepoint -- the absolute speed of the invasion front (equivalently, the overall
time-scale).  It is drawn deterministically per patient from the literature
range and is the single documented modelling assumption; see
:func:`derive_growth`.

Density <-> segmentation convention (GliODIL ``synthetic_generator.py:63``)::

    label 1  enhancing core     u >= th_up      (th_up   in [0.45, 0.60])
    label 3  edema              th_down <= u < th_up  (th_down in [0.15, 0.35])
    label 4  necrotic core      u >= th_necro
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# GliODIL density thresholds (midpoints of its sampling ranges).
TH_UP = 0.50        # enhancing-core iso-level
TH_DOWN = 0.25      # edema (FLAIR) iso-level
TH_NECRO = 0.85     # necrotic iso-level

NATIVE_MM = 1.0     # the released volumes are 1 mm isotropic (240 x 240 x 155)

_LABEL_CORE = (1, 4)   # enhancing + necrotic == the T1Gd-visible bulk
                       # (labels 2, 3 are the FLAIR abnormality around it)

# Decay length used for the sub-radiological tail beyond the FLAIR contour.
# Clipped tighter than the dynamical L so a single extreme rim measurement
# cannot put most of the initial burden outside any plausible treatment field.
TAIL_L_CLIP_MM = (2.0, 8.0)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class PatientGeometry:
    """Everything the forward solver needs about one real patient."""

    pid: str
    WM: np.ndarray            # (nx, ny, nz) float32 in [0, 1]
    GM: np.ndarray
    CSF: np.ndarray
    brain: np.ndarray         # bool, WM + GM above the matter threshold
    A0: np.ndarray            # float32 initial tumour-cell density in [0, 1]
    core: np.ndarray          # bool, T1Gd-visible bulk (labels 1, 4)
    whole: np.ndarray         # bool, whole FLAIR abnormality (all labels)
    recurrence: np.ndarray    # bool, follow-up recurrence segmentation
    voxel_mm: float           # isotropic voxel size after resampling
    centroid_vox: Tuple[float, float, float]
    R_core_mm: float
    R_whole_mm: float
    rim_mm: float             # R_whole - R_core, the infiltration-rim thickness
    L_mm: float               # invisibility length sqrt(D/f) implied by the rim
    info: Dict[str, float] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self.WM.shape)         # type: ignore[return-value]

    @property
    def centroid_frac(self) -> Tuple[float, float, float]:
        return tuple(float(c) / s for c, s in zip(self.centroid_vox, self.shape))


@dataclass
class BeamConfig:
    """One tumour--beam alignment scenario (the paper's four configurations)."""

    name: str
    label: str
    target: str          # "whole" or "core": which mask the PTV conforms to
    margin_mm: float     # CTV/PTV expansion around that mask
    shift_frac: float    # displacement of the field, as a fraction of R_whole


# The paper's four beam-tumour configurations, transplanted onto real anatomy.
# `full_cover` is a clinically standard GBM plan (FLAIR abnormality + 15 mm);
# the other three are the controlled geometry mismatches the paper studies.
#
# `narrow_centered` and `strong_shift` are deliberately tuned to deliver almost
# the SAME total dose to the tumour (mass-weighted coverage ~0.45 for both) with
# completely different spatial patterns -- centred-but-too-small versus
# correctly-sized-but-displaced.  A reduced 0-D surrogate sees an identical
# U(t) and a near-identical effective coverage for the two, so any difference in
# outcome is by construction unresolvable without a closure term.
BEAM_CONFIGS: Tuple[BeamConfig, ...] = (
    BeamConfig("full_cover", "Full cover (PTV = FLAIR + 15 mm)", "whole", 15.0, 0.0),
    BeamConfig("narrow_centered", "Narrow centred (PTV = core + 5 mm)", "core", 5.0, 0.0),
    BeamConfig("slight_shift", "Slight shift (0.75 R)", "whole", 15.0, 0.75),
    BeamConfig("strong_shift", "Strong shift (1.5 R)", "whole", 15.0, 1.5),
)
BEAM_BY_NAME = {b.name: b for b in BEAM_CONFIGS}


# ---------------------------------------------------------------------------
# Deterministic per-patient randomness
# ---------------------------------------------------------------------------
def patient_rng(pid: str, salt: str = "") -> np.random.Generator:
    """Stable per-patient RNG (``hashlib``, not the salted builtin ``hash``)."""
    digest = hashlib.sha256((pid + "|" + salt).encode()).digest()[:8]
    return np.random.default_rng(int.from_bytes(digest, "big"))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load_nii(path: Path) -> np.ndarray:
    import nibabel as nib
    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)


def _resample(vol: np.ndarray, factor: float, order: int) -> np.ndarray:
    if abs(factor - 1.0) < 1e-9:
        return vol
    from scipy.ndimage import zoom
    return zoom(vol, factor, order=order)


def _equiv_radius_mm(n_vox: float, voxel_mm: float) -> float:
    """Radius of the sphere with the same volume as ``n_vox`` voxels."""
    vol = float(n_vox) * voxel_mm ** 3
    return float((3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0)) if vol > 0 else 0.0


def load_geometry(pid: str, root: Path, grid: int = 160,
                  matter_threshold: float = 0.1) -> PatientGeometry:
    """Load one patient and build its initial tumour-density field.

    ``grid`` sets the resampled size of the *largest* axis; a single isotropic
    scale factor is used so voxels stay cubic (native volumes are 240x240x155
    at 1 mm, so ``grid=160`` gives 1.5 mm voxels).
    """
    pdir = Path(root) / pid
    seg = _load_nii(pdir / "segm.nii.gz")
    factor = grid / max(seg.shape)
    voxel_mm = NATIVE_MM / factor

    core_n = np.isin(seg, _LABEL_CORE)
    whole_n = seg > 0

    WM = np.clip(_resample(_load_nii(pdir / "t1_wm.nii.gz"), factor, 1), 0.0, 1.0)
    GM = np.clip(_resample(_load_nii(pdir / "t1_gm.nii.gz"), factor, 1), 0.0, 1.0)
    csf_p = pdir / "t1_csf.nii.gz"
    CSF = (np.clip(_resample(_load_nii(csf_p), factor, 1), 0.0, 1.0)
           if csf_p.exists() else np.zeros_like(WM))
    core = _resample(core_n.astype(np.float32), factor, 1) > 0.5
    whole = _resample(whole_n.astype(np.float32), factor, 1) > 0.5
    rec_p = pdir / "segm_rec.nii.gz"
    rec = (_resample((_load_nii(rec_p) > 0).astype(np.float32), factor, 1) > 0.5
           if rec_p.exists() else np.zeros_like(core))

    brain = (WM + GM) > matter_threshold
    if not core.any():                       # tiny/absent enhancing bulk
        core = whole.copy()

    # Geometry measured on the NATIVE grid (resampling-independent).
    R_core = _equiv_radius_mm(core_n.sum(), NATIVE_MM)
    R_whole = _equiv_radius_mm(whole_n.sum(), NATIVE_MM)
    rim = max(R_whole - R_core, 0.0)
    # Fisher front decays as exp(-r / L); the two iso-levels are TH_UP and
    # TH_DOWN, so the measured rim thickness fixes L = rim / ln(th_up/th_down).
    L_mm = rim / np.log(TH_UP / TH_DOWN)

    idx = np.array(np.nonzero(whole), dtype=float)
    centroid = tuple(float(c) for c in idx.mean(axis=1)) if idx.size else \
        tuple(s / 2 for s in core.shape)

    tail_L_mm = float(np.clip(L_mm, *TAIL_L_CLIP_MM))
    A0 = build_initial_density(core, whole, brain, tail_L_mm / voxel_mm)

    info = {
        "V_core_mm3": float(core_n.sum()), "V_whole_mm3": float(whole_n.sum()),
        "V_rec_mm3": float((_load_nii(rec_p) > 0).sum()) if rec_p.exists() else 0.0,
        "brain_voxels": float(brain.sum()),
        "brain_volume_ml": float(brain.sum()) * voxel_mm ** 3 / 1000.0,
        "mass0": float(A0.sum()),
    }
    return PatientGeometry(pid=pid, WM=WM, GM=GM, CSF=CSF, brain=brain, A0=A0,
                           core=core, whole=whole, recurrence=rec,
                           voxel_mm=voxel_mm, centroid_vox=centroid,
                           R_core_mm=R_core, R_whole_mm=R_whole, rim_mm=rim,
                           L_mm=L_mm, info=info)


# ---------------------------------------------------------------------------
# Initial condition: two-threshold density reconstruction
# ---------------------------------------------------------------------------
def build_initial_density(core: np.ndarray, whole: np.ndarray,
                          brain: np.ndarray, tail_L_vox: float) -> np.ndarray:
    """Cell-density field that reproduces BOTH of the patient's real contours.

    Three regions, glued so the field is continuous and matches the measured
    iso-surfaces exactly (not just on average):

    ``inside the enhancing core``
        ``u = 1 - (1 - th_up) exp(d_c / L)`` -- saturates toward the carrying
        capacity, equals ``th_up`` on the enhancing surface.
    ``between the enhancing and FLAIR surfaces``
        log-linear in the *local* rim coordinate ``s = d_c / (d_c + |d_w|)``:
        ``u = th_up (th_down/th_up)^s``.  This is the exponential Fisher tail
        with a locally-measured decay length, so a thin rim on one side of the
        tumour and a thick rim on the other are both honoured.
    ``beyond the FLAIR surface``
        ``u = th_down exp(-d_w / L)`` with the patient's (clipped) global
        invisibility length -- the sub-radiological diffuse burden.
    """
    from scipy.ndimage import distance_transform_edt

    L = float(max(tail_L_vox, 0.6))
    core = core & whole if whole.any() else core
    d_c = distance_transform_edt(~core) - distance_transform_edt(core)
    d_w = distance_transform_edt(~whole) - distance_transform_edt(whole)

    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(d_c > 0, d_c / np.maximum(d_c - d_w, 1e-6), 0.0)
    s = np.clip(s, 0.0, 1.0)

    inner = 1.0 - (1.0 - TH_UP) * np.exp(np.minimum(d_c, 0.0) / L)
    middle = TH_UP * (TH_DOWN / TH_UP) ** s
    outer = TH_DOWN * np.exp(-np.maximum(d_w, 0.0) / L)

    A0 = np.where(d_c <= 0.0, inner, np.where(d_w <= 0.0, middle, outer))
    A0 = np.where(A0 < 1e-3, 0.0, A0)
    return np.clip(A0, 0.0, 1.0).astype(np.float32) * brain


# ---------------------------------------------------------------------------
# Irradiation field: conformal PTV grown from the real tumour mask
# ---------------------------------------------------------------------------
def beam_field(geom: PatientGeometry, cfg: BeamConfig,
               penumbra_mm: float = 4.0) -> np.ndarray:
    """Relative dose field in [0, 1] for one beam configuration.

    The field is *conformal*: it is a smoothed indicator of the real tumour mask
    expanded by ``cfg.margin_mm`` (clinical CTV/PTV practice), optionally
    displaced by ``cfg.shift_frac * R_whole`` to create the paper's controlled
    tumour--beam mismatch.  The sigmoid edge models the dose penumbra.
    """
    from scipy.ndimage import distance_transform_edt, shift as ndshift

    mask = geom.core if cfg.target == "core" else geom.whole
    if cfg.shift_frac > 0.0:
        delta = cfg.shift_frac * geom.R_whole_mm / geom.voxel_mm
        direction = _shift_direction(geom)
        mask = ndshift(mask.astype(np.float32), delta * direction,
                       order=0, mode="constant", cval=0.0) > 0.5
        if not mask.any():                     # shifted out of the volume
            mask = geom.whole

    d_out = distance_transform_edt(~mask) * geom.voxel_mm       # mm outside
    d_in = distance_transform_edt(mask) * geom.voxel_mm         # mm inside
    d = d_out - d_in                                            # signed, +outside
    w = max(penumbra_mm, 1e-3)
    return (1.0 / (1.0 + np.exp((d - cfg.margin_mm) / w))).astype(np.float32)


def _shift_direction(geom: PatientGeometry) -> np.ndarray:
    """Unit displacement direction for the shifted-beam configurations.

    Deterministic per patient and biased toward the brain interior so the field
    is displaced *within* the head rather than off the edge of the volume.
    """
    brain_idx = np.array(np.nonzero(geom.brain), dtype=float)
    brain_c = brain_idx.mean(axis=1)
    inward = brain_c - np.asarray(geom.centroid_vox, dtype=float)
    n = np.linalg.norm(inward)
    inward = inward / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
    rnd = patient_rng(geom.pid, "beamdir").normal(size=3)
    rnd /= np.linalg.norm(rnd) + 1e-12
    v = 0.65 * inward + 0.35 * rnd
    return v / (np.linalg.norm(v) + 1e-12)


# ---------------------------------------------------------------------------
# Growth parameters
# ---------------------------------------------------------------------------
# Front-speed range in voxel units per model time unit.  Together with T = 80
# this makes the invasion front advance ~8-20 voxels over the horizon, i.e. the
# tumour roughly doubles-to-quadruples in mass -- the regime in which a
# recurrence forecast is a meaningful, non-trivial task.
SPEED_RANGE = (0.10, 0.25)
# Invisibility length sqrt(D/f).  The clip only removes pathological rim
# measurements; at (3, 20) mm it leaves the cohort's p10-p90 (6-19 mm) intact,
# so this stays the genuine imaging-derived source of per-patient heterogeneity.
L_CLIP_MM = (3.0, 20.0)
RATIO_RANGE = (10.0, 30.0)      # GliODIL's WM/GM diffusion-ratio range

# Per-patient radiotherapy response (the reduced surrogates must infer these).
GAMMA_RANGE = (0.80, 1.80)      # radiosensitivity coefficient
HYPOXIA_RANGE = (0.30, 0.70)    # density-dependent radioresistance strength


def derive_growth(geom: PatientGeometry) -> Dict[str, float]:
    """Per-patient Fisher--Kolmogorov coefficients in solver (voxel) units.

    The *shape* of the growth is imaging-derived: the invisibility length
    ``L = sqrt(D/f)`` is the rim thickness measured on this patient's own
    segmentation, so an infiltrative tumour gets a diffusion-dominated kernel
    and a nodular one gets a proliferation-dominated kernel.

    The *speed* of the growth (``v = 2 sqrt(D f)``) cannot be identified from a
    single timepoint, so it is drawn deterministically per patient from
    ``SPEED_RANGE``.  Given ``(L, v)``::

        D = v * L / 2        f = v / (2 L)

    which reproduces both ``sqrt(D/f) = L`` and ``2 sqrt(D f) = v`` exactly.
    """
    L_mm = float(np.clip(geom.L_mm, *L_CLIP_MM))
    L_vox = L_mm / geom.voxel_mm
    rng = patient_rng(geom.pid, "growth")
    v = float(rng.uniform(*SPEED_RANGE))
    ratio = float(rng.uniform(*RATIO_RANGE))
    Dw = 0.5 * v * L_vox
    f = 0.5 * v / L_vox
    return {
        "Dw": Dw, "f": f, "Dw_ratio": ratio,
        "front_speed_vox": v, "front_speed_mm": v * geom.voxel_mm,
        "L_vox": L_vox, "L_mm": L_mm, "L_raw_mm": float(geom.L_mm),
    }


def cohort(root: Path, limit: Optional[int] = None,
           exclude: Sequence[str] = ()) -> list:
    """Patient ids with usable anatomy and a non-degenerate tumour."""
    root = Path(root)
    pids = sorted(p.name for p in root.iterdir()
                  if p.is_dir() and p.name.startswith("data_")
                  and (p / "segm.nii.gz").exists()
                  and (p / "t1_wm.nii.gz").exists())
    pids = [p for p in pids if p not in set(exclude)]
    return pids[:limit] if limit else pids
