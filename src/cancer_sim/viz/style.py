"""Shared figure style: one visual system for every plot in the study.

Colour assignment follows the job each encoding does:

* **categorical** (method identity, beam configuration) -- fixed hue order,
  never cycled, validated for colour-vision deficiency;
* **sequential** (tumour cell density) -- one hue, light to dark;
* **status** (dose field, ground truth, forecast window) -- reserved roles that
  never impersonate a series.

Every figure also carries a redundant, non-colour channel (direct labels, line
style, or hatching), so nothing depends on hue alone.
"""

from __future__ import annotations

from typing import Dict, Sequence

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba

# --- categorical: validated fixed order -------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#4a3aa7", "#e34948"]

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"

TRUTH = "#0b0b0b"        # the PDE ground truth is ink, never a series colour
DOSE = "#4a3aa7"         # irradiation events
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
WINDOW = "#f0efec"       # assimilation-window wash

# --- method identity (stable across every figure) ---------------------------
METHOD_COLOR: Dict[str, str] = {
    "Ground truth": TRUTH,
    "No treatment": MUTED,
    "ODE": SERIES[1],
    "ODE-vol": SERIES[3],
    "ODE-power": SERIES[4],
    "NODE": SERIES[5],
    "PI-NODE": SERIES[7],
    "PI-NODE-tuned": SERIES[6],
    "CPI-NODE": SERIES[2],
    "GPI-NODE": SERIES[0],
}
METHOD_DASH: Dict[str, tuple] = {
    "Ground truth": (),
    "No treatment": (1, 2),
    "ODE": (5, 2),
    "ODE-vol": (4, 1, 1, 1),
    "ODE-power": (6, 2, 1, 2),
    "NODE": (2, 2),
    "PI-NODE": (3, 1, 1, 1),
    "PI-NODE-tuned": (1, 1),
    "CPI-NODE": (7, 2),
    "GPI-NODE": (),
}

CASE_LABEL = {
    "full_cover": "Full cover",
    "narrow_centered": "Narrow centred",
    "slight_shift": "Slight shift",
    "strong_shift": "Strong shift",
    "no_treatment": "No treatment",
}
CASE_COLOR = {
    "full_cover": SERIES[0], "narrow_centered": SERIES[1],
    "slight_shift": SERIES[2], "strong_shift": SERIES[3],
    "no_treatment": MUTED,
}


def method_style(name: str, idx: int = 0) -> Dict:
    """Colour + dash pattern for a method, stable across figures."""
    base = name.split("(")[0].strip()
    color = METHOD_COLOR.get(base, SERIES[idx % len(SERIES)])
    dash = METHOD_DASH.get(base, (4, 2))
    return {"color": color, "dashes": dash if dash else (None, None)}


# --- sequential ramps --------------------------------------------------------
def _ramp(name: str, stops: Sequence[str], alpha_low: float = 0.0):
    cols = [to_rgba(c) for c in stops]
    if alpha_low is not None:
        cols[0] = (*cols[0][:3], alpha_low)
    return LinearSegmentedColormap.from_list(name, cols, N=256)


# tumour cell density: transparent -> blue -> deep blue (the sequential hue)
CMAP_TUMOUR = _ramp("tumour", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
# radiation damage field: transparent -> orange (the second sequential context)
CMAP_DAMAGE = _ramp("damage", ["#fde3d5", "#f7a077", "#eb6834", "#b93f10"])
# brain anatomy: neutral greys, kept low-contrast so overlays dominate
CMAP_BRAIN = LinearSegmentedColormap.from_list(
    "brain", ["#08080a", "#3a3a3c", "#7d7d7e", "#c9c9c6", "#f2f2ee"], N=256)


def apply(font_scale: float = 1.0):
    """Install the study's matplotlib defaults."""
    mpl.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 220,
        "savefig.bbox": "tight", "savefig.facecolor": SURFACE,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9 * font_scale,
        "axes.titlesize": 10 * font_scale, "axes.labelsize": 9 * font_scale,
        "xtick.labelsize": 8 * font_scale, "ytick.labelsize": 8 * font_scale,
        "legend.fontsize": 8 * font_scale,
        "axes.titleweight": "600", "axes.titlelocation": "left",
        "axes.titlepad": 6,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": GRID, "grid.linewidth": 0.7, "grid.alpha": 1.0,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.handlelength": 2.2,
        "legend.columnspacing": 1.2, "legend.borderaxespad": 0.2,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


def mark_window(ax, t0: float, t1: float, label: str = "assimilation",
                y: float = 0.955):
    """Shade the assimilation window and label it once per axes."""
    ax.axvspan(t0, t1, color=WINDOW, zorder=0, lw=0)
    if label:
        ax.text(0.5 * (t0 + t1), y, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", color=MUTED, fontsize=7)


def mark_doses(ax, times: Sequence[float], label: bool = True):
    """Vertical rules at the irradiation events, labelled inside the axes."""
    for k, td in enumerate(times):
        ax.axvline(td, color=DOSE, lw=1.0, ls=(0, (2, 2)), alpha=0.75, zorder=1)
        if label:
            ax.text(td + 0.8, 0.018, f"RT{k+1}",
                    transform=ax.get_xaxis_transform(), ha="left", va="bottom",
                    color=DOSE, fontsize=7, fontweight="700")


def label_end(ax, x, y, text, color, dx: float = 0.8, **kw):
    """Direct label at the right end of a line (so identity is not colour-only)."""
    ax.annotate(text, xy=(x, y), xytext=(dx, 0), textcoords="offset points",
                color=color, fontsize=8, fontweight="600",
                va="center", ha="left", **kw)


def titles(fig, title: str, subtitle: str = "", rect_top: float = 1.0,
           size: float = 12.0, sub_size: float = 8.7):
    """Left-aligned title + subtitle stack that never collides.

    ``matplotlib``'s ``suptitle`` has no notion of a second line, so the two are
    placed explicitly.  The vertical offset is derived from the font size and
    the figure height rather than being a fixed fraction, so short figures do
    not overlap their own subtitle.
    """
    h_in = fig.get_size_inches()[1]
    line = (size / 72.0) * 1.30 / h_in          # title height as a figure fraction
    fig.text(0.008, 0.995, title, ha="left", va="top", fontsize=size,
             fontweight="700", color=INK)
    if subtitle:
        fig.text(0.008, 0.995 - line, subtitle, ha="left", va="top",
                 fontsize=sub_size, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, min(rect_top, 0.995 - line
                                        - (1.9 * sub_size / 72.0) / h_in)))


def panel_tag(ax, tag: str, dx: float = -0.06, dy: float = 1.06):
    ax.text(dx, dy, tag, transform=ax.transAxes, fontsize=10,
            fontweight="700", color=INK, va="top", ha="left")


# --- anatomical orientation --------------------------------------------------
# The released volumes are (L, P, S): array axis 0 increases to the patient's
# LEFT, axis 1 to the POSTERIOR, axis 2 to the SUPERIOR.  Raw ``imshow`` of a
# slice would therefore be upside-down.  ``orient`` maps a stored slice into
# radiological display convention (image-left = patient right, anterior up).
ORIENT_LABELS = {
    "axial": ("R", "L", "P", "A"),        # left, right, bottom, top
    "coronal": ("R", "L", "I", "S"),
    "sagittal": ("P", "A", "I", "S"),
}


def orient(sl, plane: str):
    """Put a stored ``(axis_i, axis_j)`` slice into display orientation."""
    a = np.asarray(sl)
    if plane == "axial":        # (L, P) -> col = L, row = A (flip P)
        return a.T[::-1, :]
    if plane == "coronal":      # (L, S) -> col = L, row = S
        return a.T
    if plane == "sagittal":     # (P, S) -> col = A (flip P), row = S
        return a.T[:, ::-1]
    raise ValueError(plane)


def orientation_marks(ax, plane: str, color: str = "#c3c2b7"):
    """Compact orientation key in a corner, so the view is unambiguous."""
    left, right, bottom, top = ORIENT_LABELS[plane]
    ax.text(0.015, 0.985, f"{top}\n{left}·{right}\n{bottom}", ha="left",
            va="top", color=color, fontsize=5.6, fontweight="700",
            linespacing=0.95, transform=ax.transAxes, zorder=8)


def brain_slice(ax, wm, gm, brain=None, plane: str = "axial"):
    """Render the anatomical backdrop of a cross-section."""
    tissue = np.asarray(wm) + 0.55 * np.asarray(gm)
    if brain is not None:
        tissue = np.where(np.asarray(brain) > 0, tissue, np.nan)
    ax.imshow(orient(tissue, plane), origin="lower", cmap=CMAP_BRAIN,
              vmin=0.0, vmax=1.4, interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


def scalebar(ax, voxel_mm: float, length_mm: float = 20.0, pad: float = 0.045):
    """Physical scale bar (the cross-sections have no axes)."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    frac = (length_mm / voxel_mm) / abs(x1 - x0)
    xr = 1.0 - pad
    xl = xr - frac
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=2.6, foreground="#000000", alpha=0.55)]
    ax.plot([xl, xr], [pad, pad], color="#ffffff", lw=2.2, solid_capstyle="butt",
            transform=ax.transAxes, zorder=8, path_effects=halo)
    ax.text(0.5 * (xl + xr), pad + 0.02, f"{length_mm:.0f} mm", color="#ffffff",
            fontsize=6.5, ha="center", va="bottom", transform=ax.transAxes,
            zorder=8, path_effects=halo)
