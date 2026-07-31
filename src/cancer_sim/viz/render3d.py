"""Volume renderings of the tumour and the irradiation field.

A single depth-shaded maximum-intensity projection, computed with NumPy only --
no external renderer, so it runs anywhere the rest of the study runs.  Used for
the showcase patients whose full 3-D fields were retained.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from . import style as st


def _project(vol: np.ndarray, axis: int, shade: float = 0.55):
    """Depth-shaded maximum-intensity projection along ``axis``.

    Each ray keeps its maximum value and remembers where that maximum sat along
    the ray; the depth then darkens far material, which gives the flat MIP a
    readable sense of front-to-back without a full volumetric integrator.
    """
    v = np.moveaxis(np.asarray(vol, dtype=np.float32), axis, 0)
    n = v.shape[0]
    idx = np.argmax(v, axis=0)
    mip = np.take_along_axis(v, idx[None], axis=0)[0]
    depth = idx.astype(np.float32) / max(n - 1, 1)
    return mip, 1.0 - shade * depth


def _emit(ax, field, plane, cmap, vmax, shade=True, alpha=1.0, thresh=0.02,
          zorder=2):
    mip, dep = field
    a = st.orient(mip, plane)
    d = st.orient(dep, plane)
    rgba = cmap(np.clip(a / max(vmax, 1e-9), 0, 1))
    if shade:
        rgba[..., :3] *= d[..., None]
    rgba[..., 3] = np.where(a > thresh, alpha * np.clip(a / max(vmax, 1e-9), 0.15, 1.0), 0.0)
    ax.imshow(rgba, origin="lower", interpolation="bilinear", zorder=zorder)


def figure_volume(pid: str, cohort_dir: Path, out: Path,
                  case: str = "strong_shift",
                  times: Sequence[float] = (0.0, 18.0, 48.0, 80.0),
                  compare: Optional[str] = "full_cover"):
    """Three-dimensional view of the tumour before, during and after treatment."""
    st.apply()
    import json
    man = json.loads((Path(cohort_dir) / f"{pid}_manifest.json").read_text())
    vz = np.load(Path(cohort_dir) / f"{pid}_volumes.npz")
    fz = np.load(Path(cohort_dir) / f"{pid}_fields.npz")
    have = {}
    for key in vz.files:
        c, tt = key.split("/")
        have.setdefault(c, {})[float(tt)] = vz[key].astype(np.float32)
    rows = [r for r in ([case] + ([compare] if compare else [])) if r in have]
    if not rows:
        raise KeyError(f"no retained volumes for {pid}")
    ts = [t for t in times if t in have[rows[0]]]

    brain = fz["anat/brain_axial"] if "anat/brain_axial" in fz.files else None
    planes = [("axial", 2), ("coronal", 1)]

    fig, axes = plt.subplots(len(rows) * len(planes), len(ts),
                             figsize=(2.0 * len(ts),
                                      2.05 * len(rows) * len(planes) + 0.6),
                             squeeze=False)
    for ri, r in enumerate(rows):
        for pi, (pname, pax) in enumerate(planes):
            for j, tt in enumerate(ts):
                ax = axes[ri * len(planes) + pi][j]
                A = have[r][tt].astype(np.float32)
                ax.set_facecolor("#08080a")
                _emit(ax, _project(A, pax), pname, st.CMAP_TUMOUR, 1.0,
                      alpha=0.95)
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                ax.set_aspect("equal")
                for s in ax.spines.values():
                    s.set_visible(False)
                if ri == 0 and pi == 0:
                    ax.set_title(f"t = {tt:g}", fontsize=9, color=st.INK)
                if j == 0:
                    ax.set_ylabel(f"{st.CASE_LABEL[r]}\n{pname}", fontsize=8,
                                  color=st.INK, fontweight="600")
                    ax.yaxis.set_visible(True); ax.set_yticks([])
                    st.orientation_marks(ax, pname)
    st.titles(fig, f"Tumour burden in three dimensions · patient {pid}",
              "depth-shaded maximum-intensity projection of the cell-density "
              "field; the same dose schedule, two field geometries",
              rect_top=0.90)
    fig.subplots_adjust(wspace=0.02, hspace=0.06)
    fig.savefig(out, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    return out
