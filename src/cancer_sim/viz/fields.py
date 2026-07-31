"""Cross-section figures: tumour and damage fields on real patient anatomy."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import style as st


def load_fields(pid: str, cohort_dir: Path) -> Dict:
    """Read one patient's stored anatomy and snapshot planes."""
    z = np.load(Path(cohort_dir) / f"{pid}_fields.npz")
    anat, snaps = {}, {}
    for key in z.files:
        if key.startswith("anat/"):
            anat[key[5:]] = z[key].astype(np.float32)
        else:
            case, tt, what = key.split("/")
            snaps.setdefault(case, {}).setdefault(float(tt), {})[what] = \
                z[key].astype(np.float32)
    return {"anat": anat, "snaps": snaps}


def _crop(shape, centre, half: int):
    """Square window around the tumour, clipped to the volume."""
    out = []
    for c, n in zip(centre, shape):
        lo = int(np.clip(c - half, 0, max(n - 2 * half, 0)))
        out.append((lo, min(lo + 2 * half, n)))
    return out


def draw_slice(ax, anat: Dict, A: Optional[np.ndarray] = None,
               Z: Optional[np.ndarray] = None, beam: Optional[np.ndarray] = None,
               plane: str = "axial", box=None, voxel_mm: float = 1.5,
               a_max: float = 1.0, z_max: Optional[float] = None,
               show_scale: bool = True):
    """One cross-section: anatomy + tumour density + damage + beam outline."""
    def sl(a):
        a = np.asarray(a, dtype=float)
        if box is not None:
            a = a[box[0][0]:box[0][1], box[1][0]:box[1][1]]
        return st.orient(a, plane)

    wm, gm = anat[f"wm_{plane}"], anat[f"gm_{plane}"]
    raw = lambda a: (a if box is None
                     else np.asarray(a)[box[0][0]:box[0][1], box[1][0]:box[1][1]])
    st.brain_slice(ax, raw(wm), raw(gm), raw((wm + gm) > 0.1), plane=plane)

    if A is not None:
        a = sl(A)
        ax.imshow(np.where(a > 0.02, a, np.nan), origin="lower",
                  cmap=st.CMAP_TUMOUR, vmin=0.0, vmax=a_max, alpha=0.92,
                  interpolation="bilinear", zorder=2)
        # iso-contours at the two clinically visible thresholds
        ax.contour(a, levels=[0.25], colors=["#ffffff"], linewidths=0.7,
                   alpha=0.8, zorder=3)
        ax.contour(a, levels=[0.50], colors=["#ffffff"], linewidths=1.2,
                   zorder=3)
    if Z is not None and z_max:
        zz = sl(Z)
        ax.imshow(np.where(zz > 0.03 * z_max, zz, np.nan), origin="lower",
                  cmap=st.CMAP_DAMAGE, vmin=0.0, vmax=z_max, alpha=0.5,
                  interpolation="bilinear", zorder=4)
    if beam is not None:
        ax.contour(sl(beam), levels=[0.5], colors=[st.DOSE], linewidths=1.4,
                   linestyles="--", zorder=5)
    if show_scale:
        st.orientation_marks(ax, plane)
        st.scalebar(ax, voxel_mm, 20.0)
    return ax


def figure_beam_configs(pid: str, cohort_dir: Path, out: Path,
                        cases: Sequence[str] = ("full_cover", "narrow_centered",
                                                "slight_shift", "strong_shift")):
    """The four tumour--beam configurations on one patient's real anatomy."""
    st.apply()
    d = load_fields(pid, cohort_dir)
    anat, snaps = d["anat"], d["snaps"]
    man = _manifest(pid, cohort_dir)
    A0 = snaps[cases[0]][0.0]["A_axial"]
    box = _crop(A0.shape, _tumour_centre(A0), 62)

    fig, axes = plt.subplots(1, len(cases), figsize=(2.55 * len(cases), 3.15))
    for k, (ax, case) in enumerate(zip(np.atleast_1d(axes), cases)):
        draw_slice(ax, anat, A=A0, beam=anat[f"beam_{case}_axial"], box=box,
                   voxel_mm=man["voxel_mm"], show_scale=(k == 0))
        cov = man["cases"][case]["beam_coverage0"]
        ax.set_title(f"({'abcd'[k]}) {st.CASE_LABEL[case]}\n"
                     f"dose coverage {cov:.0%}", fontsize=9, color=st.INK,
                     loc="center")

    handles = _field_legend()
    handles[0].set_label("cell density $u(x,0)$")
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    st.titles(fig, f"Beam–tumour configurations on real anatomy · patient {pid}",
              "one axial section through the tumour centroid; the initial cell "
              "density is reconstructed from this patient's own segmentation",
              rect_top=0.80)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _field_legend():
    from matplotlib.colors import to_rgba
    grad = Patch(facecolor=to_rgba("#2a78d6", 0.85), edgecolor="none")
    return [
        grad,
        Line2D([], [], color="#8c8c8c", lw=1.2, label="tumour $u\\geq0.50$ (T1Gd)"),
        Line2D([], [], color="#8c8c8c", lw=0.7, label="tumour $u\\geq0.25$ (FLAIR)"),
        Line2D([], [], color=st.DOSE, lw=1.4, ls="--",
               label="irradiation field (50 % isodose)"),
    ]


def figure_time_evolution(pid: str, case: str, cohort_dir: Path, out: Path,
                          times: Sequence[float] = (0.0, 15.0, 18.0, 30.0,
                                                    45.0, 48.0, 60.0, 80.0),
                          compare: Optional[str] = None):
    """Spatio-temporal montage: knockdown and regrowth, with the damage field.

    With ``compare`` set, a second row shows the same times for another beam
    configuration, so the geometry-dependent difference is visible directly.
    """
    st.apply()
    d = load_fields(pid, cohort_dir)
    anat, snaps = d["anat"], d["snaps"]
    man = _manifest(pid, cohort_dir)
    rows = [case] + ([compare] if compare else [])
    have = [t for t in times if t in snaps[rows[0]]]

    A0 = snaps[rows[0]][have[0]]["A_axial"]
    box = _crop(A0.shape, _tumour_centre(A0), 62)
    z_max = max(float(np.max(snaps[r][t]["Z_axial"]))
                for r in rows for t in have) or 1.0

    fig_h = 1.62 * len(rows) + 0.7
    fig, axes = plt.subplots(len(rows), len(have),
                             figsize=(1.52 * len(have), fig_h), squeeze=False)
    curves = _curves(pid, cohort_dir)
    for i, r in enumerate(rows):
        m, tgrid = curves[r]["mass"], curves[r]["t"]
        for j, tt in enumerate(have):
            ax = axes[i][j]
            s = snaps[r][tt]
            draw_slice(ax, anat, A=s["A_axial"], Z=s["Z_axial"],
                       beam=anat[f"beam_{r}_axial"], box=box,
                       voxel_mm=man["voxel_mm"], z_max=z_max,
                       show_scale=(i == 0 and j == 0))
            if i == 0:
                dose = " ↓RT" if any(abs(tt - d) < 0.6 for d in (15.0, 45.0)) else ""
                ax.set_title(f"t = {tt:g}{dose}", fontsize=8.5, pad=3,
                             color=st.DOSE if dose else st.INK, loc="center")
            ratio = float(np.interp(tt, tgrid, m))
            ax.text(0.97, 0.975, f"×{ratio:.2f}", transform=ax.transAxes,
                    ha="right", va="top", color="#ffffff", fontsize=8.5,
                    fontweight="700", zorder=9,
                    bbox=dict(facecolor="#000000", alpha=0.42, pad=1.6,
                              edgecolor="none", boxstyle="round,pad=0.22"))
        axes[i][0].set_ylabel(st.CASE_LABEL[r], fontsize=9, color=st.INK,
                              fontweight="600", labelpad=6)
        axes[i][0].yaxis.set_visible(True)
        axes[i][0].set_yticks([])

    handles = [
        Patch(facecolor="#2a78d6", alpha=0.9, label="tumour cell density $u$"),
        Patch(facecolor="#eb6834", alpha=0.55, label="radiation damage $Z$"),
        Line2D([], [], color=st.DOSE, lw=1.4, ls="--", label="irradiation field"),
        Line2D([], [], color="#8c8c8c", lw=1.2,
               label="$u\\geq0.5$ (imaging-visible bulk)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.035))
    st.titles(fig,
              f"Radiotherapy response in tissue · patient {pid}",
              "fractions delivered at t = 15 and t = 45; “×” is the tumour mass "
              "relative to baseline."
              + (" Same patient, same dose schedule — only the field geometry "
                 "differs." if len(rows) > 1 else ""),
              rect_top=1.0 - 0.62 / fig_h)
    fig.subplots_adjust(wspace=0.02, hspace=0.10)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _tumour_centre(A: np.ndarray):
    m = A > 0.2
    if not m.any():
        return [s / 2 for s in A.shape]
    idx = np.array(np.nonzero(m), dtype=float)
    return idx.mean(axis=1)


def _manifest(pid: str, cohort_dir: Path) -> Dict:
    import json
    return json.loads((Path(cohort_dir) / f"{pid}_manifest.json").read_text())


def _curves(pid: str, cohort_dir: Path):
    from ..realdata.cohort import load_curves
    return load_curves(pid, cohort_dir)
