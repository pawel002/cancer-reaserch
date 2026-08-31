"""Animated cross-sections of irradiated tumour growth.

Renders the dense snapshot cache written by ``experiments/make_animations.py``
into GIFs.  Every artist is created once and updated per frame -- redrawing a
five-panel figure a hundred times is otherwise the dominant cost.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba

from . import style as st

CASES = ("no_treatment", "full_cover", "narrow_centered", "strong_shift")

_DOSE_RGB = to_rgba(st.DOSE)[:3]
# Beam wash: painted over the anatomy only while the source is on, so
# irradiation reads as an event rather than as decoration.
CMAP_BEAM = LinearSegmentedColormap.from_list(
    "beam", [(*_DOSE_RGB, 0.0), (*_DOSE_RGB, 0.60)], N=256)

# Cell density.  The stock ramp fades to alpha 0 at its low end, which makes the
# infiltrative margin -- the part that matters here -- nearly invisible; this
# one keeps the margin legible and still saturates in the core.
CMAP_TUM = LinearSegmentedColormap.from_list("tum", [
    (*to_rgba("#8cc0f5")[:3], 0.30), (*to_rgba("#4b90e2")[:3], 0.66),
    (*to_rgba("#2a78d6")[:3], 0.86), (*to_rgba("#184f95")[:3], 0.96),
    (*to_rgba("#0d366b")[:3], 1.00)], N=256)
CMAP_DAM = LinearSegmentedColormap.from_list("dam", [
    (*to_rgba("#f9c3a4")[:3], 0.28), (*to_rgba("#f0894f")[:3], 0.70),
    (*to_rgba("#eb6834")[:3], 0.90), (*to_rgba("#a8330a")[:3], 1.00)], N=256)

TH = 0.06          # display floor: below this the density is not drawn
FPS = 12
DPI = 96


def _load(cache: Path) -> Dict:
    z = np.load(cache)
    return {k: z[k] for k in z.files}


def _grab(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def _to_gif(frames: List[np.ndarray], dst: Path, fps: int = FPS,
            hold: int = 10) -> None:
    """Write the GIF, holding the final frame so the outcome can be read."""
    import imageio.v2 as imageio
    frames = frames + [frames[-1]] * hold
    imageio.mimsave(dst, frames, format="GIF", fps=fps, loop=0,
                    palettesize=64, subrectangles=True)
    print(f"  -> {dst.name}  ({dst.stat().st_size/1e6:.1f} MB, "
          f"{len(frames)} frames)")


def _frame_plan(t_snap: np.ndarray, dose_times, coarse: int = 4,
                near: float = 2.5, hold: int = 3) -> List[int]:
    """Snapshot indices to render: coarse overall, dense around each dose."""
    idx = []
    for k, ts in enumerate(t_snap):
        close = any(abs(float(ts) - float(td)) <= near for td in dose_times)
        if close or k % coarse == 0 or k == len(t_snap) - 1:
            idx.append(k)
            if any(abs(float(ts) - float(td)) < 0.51 for td in dose_times):
                idx += [k] * hold          # linger on the moment of exposure
    return idx


def _box(brain: np.ndarray, pad: int = 3) -> Tuple[slice, slice]:
    """Bounding box of the head, so the panels are not mostly background."""
    ii, jj = np.where(np.asarray(brain) > 0)
    if len(ii) == 0:
        return slice(None), slice(None)
    return (slice(max(int(ii.min()) - pad, 0), int(ii.max()) + pad + 1),
            slice(max(int(jj.min()) - pad, 0), int(jj.max()) + pad + 1))


def _titles(fig, title: str, subtitle: str, width: int = 118) -> None:
    """Figure titles that do not disturb an explicitly placed gridspec."""
    import textwrap
    fig.text(0.012, 0.976, title, fontsize=12.5, fontweight="700",
             color=st.INK, ha="left", va="top")
    fig.text(0.012, 0.930, "\n".join(textwrap.wrap(subtitle, width)),
             fontsize=8.3, color=st.MUTED, ha="left", va="top",
             linespacing=1.35)


def _panel(ax, d: Dict, case: str, bx) -> Dict:
    """Static backdrop plus the two updatable overlays of one cross-section."""
    st.brain_slice(ax, d["anat/wm"][bx], d["anat/gm"][bx],
                   d["anat/brain"][bx], "axial")
    beam_im = None
    if f"beam/{case}" in d:
        beam = st.orient(d[f"beam/{case}"][bx], "axial")
        beam_im = ax.imshow(beam, origin="lower", cmap=CMAP_BEAM, vmin=0.0,
                            vmax=1.0, interpolation="bilinear", zorder=2)
        beam_im.set_alpha(0.0)
        ax.contour(beam, levels=[0.5], colors=[st.DOSE], linewidths=1.0,
                   linestyles="dashed", zorder=5, alpha=0.9)
    A = st.orient(d[f"{case}/A"][0][bx].astype(np.float32), "axial")
    tum = ax.imshow(np.where(A >= TH, A, np.nan), origin="lower",
                    cmap=CMAP_TUM, vmin=TH, vmax=1.0,
                    interpolation="bilinear", zorder=4)
    return {"beam": beam_im, "tum": tum}


def _set_panel(art, d: Dict, case: str, k: int, bx, u_rel: float) -> None:
    A = st.orient(d[f"{case}/A"][k][bx].astype(np.float32), "axial")
    art["tum"].set_data(np.where(A >= TH, A, np.nan))
    if art["beam"] is not None:
        art["beam"].set_alpha(float(np.clip(u_rel, 0.0, 1.0)))


def _clock(fig, T: float):
    return fig.text(0.988, 0.976, f"t = 0.0 / {T:.0f}", ha="right", va="top",
                    fontsize=11.0, color=st.INK, fontweight="700",
                    family="monospace")


# ---------------------------------------------------------------------------
def anim_treatment(d: Dict, dst: Path, pid: str, stride: int = 2) -> None:
    """Four beam geometries on one anatomy, with the mass each produces."""
    st.apply(1.0)
    bx = _box(d["anat/brain"])
    t_snap, t = d["t_snap"], d["full_cover/t"]
    U = d["full_cover/U"]
    umax = float(U.max()) or 1.0

    fig = plt.figure(figsize=(11.0, 6.2), dpi=DPI)
    gs = fig.add_gridspec(2, 4, height_ratios=[2.5, 1.0], hspace=0.10,
                          wspace=0.03, left=0.052, right=0.988,
                          top=0.858, bottom=0.098)

    arts, badges = {}, {}
    clock = None
    for j, case in enumerate(CASES):
        ax = fig.add_subplot(gs[0, j])
        arts[case] = _panel(ax, d, case, bx)
        cov = float(d[f"{case}/cover"][0]) if f"{case}/cover" in d else 0.0
        sub = ("no irradiation" if case == "no_treatment"
               else f"{cov:.0%} of tumour mass inside the beam")
        ax.text(0.0, 1.055, st.CASE_LABEL[case], transform=ax.transAxes,
                fontsize=10.2, fontweight="700", color=st.CASE_COLOR[case],
                ha="left", va="baseline")
        ax.text(0.0, 1.012, sub, transform=ax.transAxes, fontsize=7.6,
                color=st.MUTED, ha="left", va="baseline")
        badges[case] = ax.text(
            0.5, 0.035, "IRRADIATING", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=7.6, fontweight="800", color="#ffffff",
            zorder=9, alpha=0.0,
            path_effects=[pe.withStroke(linewidth=3.4, foreground=st.DOSE)])
        if j == 0:
            st.scalebar(ax, float(d["meta/voxel_mm"]), 20.0)
            st.orientation_marks(ax, "axial")
    clock = _clock(fig, float(t[-1]))

    axc = fig.add_subplot(gs[1, :])
    st.mark_doses(axc, list(d["meta/dose_times"]))
    axc.axhline(1.0, color=st.AXIS, lw=0.9, ls=(0, (4, 3)), zorder=1)
    lines, heads = {}, {}
    for case in CASES:
        c = st.CASE_COLOR[case]
        lines[case], = axc.plot([], [], color=c, lw=2.1, zorder=4,
                                label=st.CASE_LABEL[case])
        heads[case], = axc.plot([], [], "o", color=c, ms=4.6, zorder=5,
                                mec="#ffffff", mew=1.0)
    cursor = axc.axvline(0.0, color=st.INK, lw=1.0, alpha=0.4, zorder=6)
    ymax = max(float(d[f"{c}/mass"].max()) for c in CASES)
    axc.set_xlim(0, float(t[-1])); axc.set_ylim(0, ymax * 1.06)
    axc.set_xlabel("time (model units)")
    axc.set_ylabel("tumour mass  $y(t)/y(0)$")
    axc.legend(ncol=4, loc="lower left", bbox_to_anchor=(0.0, 1.005),
               fontsize=8.4)

    _titles(fig, f"Radiotherapy on real glioma anatomy — patient {pid}",
            "One patient's white/grey matter drives a 3-D reaction-diffusion "
            "tumour; four beam geometries irradiate it. Violet wash = source "
            "on, dashed outline = planned target volume.")

    frames = []
    for k in _frame_plan(t_snap, d["meta/dose_times"]):
        i = int(np.clip(round(t_snap[k] / (t[1] - t[0])), 0, len(t) - 1))
        u_rel = float(U[i]) / umax
        for case in CASES:
            _set_panel(arts[case], d, case, k, bx,
                       u_rel if case != "no_treatment" else 0.0)
            badges[case].set_alpha(0.0 if case == "no_treatment"
                                   else float(u_rel > 0.02))
            m = d[f"{case}/mass"][: i + 1]
            lines[case].set_data(t[: i + 1], m)
            heads[case].set_data([t[i]], [m[-1]])
        cursor.set_xdata([t[i], t[i]])
        clock.set_text(f"t = {t[i]:.1f} / {t[-1]:.0f}")
        frames.append(_grab(fig))
    plt.close(fig)
    _to_gif(frames, dst)


# ---------------------------------------------------------------------------
def anim_mechanism(d: Dict, dst: Path, pid: str, case: str = "full_cover",
                   stride: int = 2) -> None:
    """Cell density and the damage field that suppresses it, side by side."""
    st.apply(1.05)
    bx = _box(d["anat/brain"])
    t_snap, t = d["t_snap"], d[f"{case}/t"]
    U = d[f"{case}/U"]
    umax = float(U.max()) or 1.0

    fig = plt.figure(figsize=(9.4, 5.9), dpi=DPI)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.45, 1.0], hspace=0.11,
                          wspace=0.035, left=0.078, right=0.978,
                          top=0.845, bottom=0.108)

    ax0 = fig.add_subplot(gs[0, 0])
    art = _panel(ax0, d, case, bx)
    ax0.text(0.0, 1.015, "Tumour cell density  $A(x,t)$", transform=ax0.transAxes,
             fontsize=9.8, fontweight="700", color=st.SERIES[0], ha="left",
             va="baseline")
    st.scalebar(ax0, float(d["meta/voxel_mm"]), 20.0)
    st.orientation_marks(ax0, "axial")
    clock = _clock(fig, float(t[-1]))

    ax1 = fig.add_subplot(gs[0, 1])
    st.brain_slice(ax1, d["anat/wm"][bx], d["anat/gm"][bx],
                   d["anat/brain"][bx], "axial")
    gam, hyp = float(d["meta/gamma"]), float(d["meta/hypoxia"])
    Kill = (gam * np.clip(1.0 - hyp * d[f"{case}/A"].astype(np.float32), 0, None)
            * d[f"{case}/Z"].astype(np.float32)
            * d[f"{case}/A"].astype(np.float32))
    kmax = float(np.percentile(Kill[Kill > 0], 99.5)) if (Kill > 0).any() else 1.0
    K0 = st.orient(Kill[0][bx], "axial")
    dam = ax1.imshow(np.where(K0 > 1e-4, K0, np.nan), origin="lower",
                     cmap=CMAP_DAM, vmin=0.0, vmax=kmax,
                     interpolation="bilinear", zorder=4)
    beam = st.orient(d[f"beam/{case}"][bx], "axial")
    ax1.contour(beam, levels=[0.5], colors=[st.DOSE], linewidths=1.0,
                linestyles="dashed", zorder=5, alpha=0.9)
    ax1.text(0.0, 1.015, r"Cell kill rate  $\gamma(1-hA)\,Z\,A$",
             transform=ax1.transAxes,
             fontsize=9.8, fontweight="700", color=st.SERIES[1], ha="left",
             va="baseline")

    axc = fig.add_subplot(gs[1, :])
    st.mark_doses(axc, list(d["meta/dose_times"]))
    axc.axhline(1.0, color=st.AXIS, lw=0.9, ls=(0, (4, 3)), zorder=1)
    axc.plot(t, d["no_treatment/mass"], color=st.MUTED, lw=1.7,
             ls=(0, (5, 3)), zorder=3, label="untreated counterfactual")
    lm, = axc.plot([], [], color=st.TRUTH, lw=2.3, zorder=5, label="irradiated")
    head, = axc.plot([], [], "o", color=st.TRUTH, ms=5.0, zorder=6,
                     mec="#ffffff", mew=1.0)
    axu = axc.twinx()
    axu.set_ylim(0, umax * 3.6); axu.set_yticks([]); axu.grid(False)
    for s in axu.spines.values():
        s.set_visible(False)
    axu.fill_between(t, 0, U, color=st.DOSE, alpha=0.18, lw=0, zorder=2)
    cursor = axc.axvline(0.0, color=st.INK, lw=1.0, alpha=0.4, zorder=7)
    axc.set_xlim(0, float(t[-1]))
    axc.set_ylim(0, max(float(d["no_treatment/mass"].max()),
                        float(d[f"{case}/mass"].max())) * 1.06)
    axc.set_xlabel("time (model units)")
    axc.set_ylabel("tumour mass  $y(t)/y(0)$")
    axc.legend(ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.005),
               fontsize=8.6)

    _titles(fig, f"How irradiation enters the model - {st.CASE_LABEL[case].lower()}"
                 f", patient {pid}",
            "Dose deposits damage Z in the beam; Z decays with time constant "
            "tau and removes cells at rate gamma(1-hA)ZA. Kill peaks where dose "
            "and cell density overlap -- what a mismatched field loses.")

    frames = []
    for k in _frame_plan(t_snap, d["meta/dose_times"]):
        i = int(np.clip(round(t_snap[k] / (t[1] - t[0])), 0, len(t) - 1))
        _set_panel(art, d, case, k, bx, float(U[i]) / umax)
        K = st.orient(Kill[k][bx], "axial")
        dam.set_data(np.where(K > 1e-4, K, np.nan))
        m = d[f"{case}/mass"][: i + 1]
        lm.set_data(t[: i + 1], m)
        head.set_data([t[i]], [m[-1]])
        cursor.set_xdata([t[i], t[i]])
        clock.set_text(f"t = {t[i]:.1f} / {t[-1]:.0f}")
        frames.append(_grab(fig))
    plt.close(fig)
    _to_gif(frames, dst)


def render_all(cache: Path, out: Path, pid: str = "") -> None:
    d = _load(Path(cache))
    pid = pid or Path(cache).name.split("_dense")[0]
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    anim_treatment(d, out / "anim1_treatment.gif", pid)
    anim_mechanism(d, out / "anim2_mechanism.gif", pid)


# ---------------------------------------------------------------------------
# The surrogate: forecast unrolling, and the balance it actually realises
# ---------------------------------------------------------------------------
MODEL_STYLE = {
    "PI-NODE (retuned)":   dict(color="#2a78d6", lw=2.4, zorder=7),
    "PI-NODE (published)": dict(color="#7fb2ea", lw=2.0, zorder=6),
    "Mechanistic ODE":     dict(color="#1baf7a", lw=1.9, zorder=5,
                                ls=(0, (5, 2))),
    "Closure only":        dict(color="#eb6834", lw=1.9, zorder=5,
                                ls=(0, (2, 2))),
}
NOMINAL_ML = {"PI-NODE (published)": 0.10 / (0.05 + 0.10),
              "PI-NODE (retuned)": 0.02 / (0.02 + 0.02)}


def anim_model(npz: Path, meta: Dict, dst: Path, cases=("full_cover",
                                                        "narrow_centered"),
               train_end: float = 35.0) -> None:
    """Forecast unrolling next to the realised physics/ML balance."""
    st.apply(1.0)
    d = {k: np.load(npz)[k] for k in np.load(npz).files}
    t, t_obs = d["t_pred"], d["t_obs"]
    labels = [tuple(l) for l in meta["labels"]]
    M = int(meta["n_members"])
    variants = [v for v in meta["variants"] if f"{v}/y_pred" in d]

    # representative patient: the one whose retuned-PI-NODE error is the median
    key = "PI-NODE (retuned)" if "PI-NODE (retuned)/test" in d else variants[0]
    err = d[f"{key}/test"]
    pick = {}
    for case in cases:
        cand = [(i, labels[i][0]) for i in range(len(labels))
                if labels[i][1] == case]
        e = np.array([np.nanmedian(err[i * M:(i + 1) * M]) for i, _ in cand])
        pick[case] = cand[int(np.argsort(e)[len(e) // 2])]

    fig = plt.figure(figsize=(11.4, 6.6), dpi=DPI)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.85, 1.0], hspace=0.30,
                          wspace=0.16, left=0.062, right=0.985,
                          top=0.800, bottom=0.095)

    art = {}
    for j, case in enumerate(cases):
        ci, pid = pick[case]
        col = ci * M
        axm = fig.add_subplot(gs[0, j])
        st.mark_window(axm, float(t_obs[0]), train_end)
        st.mark_doses(axm, [15.0, 45.0])
        axm.plot(t, d["y_true"][:, ci], color=st.TRUTH, lw=2.6, zorder=4,
                 label="ground truth (3-D PDE)")
        axm.plot(t_obs, d["Y_obs"][:, col], "o", color=st.MUTED, ms=3.4,
                 mec="none", zorder=8, label="noisy observations")
        lines = {}
        for v in variants:
            lines[v], = axm.plot([], [], label=v, **MODEL_STYLE[v])
        cur = axm.axvline(t[0], color=st.INK, lw=1.0, alpha=0.4, zorder=9)
        axm.set_xlim(float(t[0]), float(t[-1]))
        ymax = max(float(np.nanmax(d["y_true"][:, ci])),
                   *[float(np.nanmax(d[f"{v}/y_pred"][:, col])) for v in variants])
        axm.set_ylim(0, min(ymax, 4.0) * 1.08)
        axm.set_title(f"{st.CASE_LABEL[case]} — {pid}", fontsize=10, loc="left")
        axm.set_ylabel("tumour mass  $y(t)/y(0)$")
        if j == 0:
            handles = axm.get_legend_handles_labels()

        axp = fig.add_subplot(gs[1, j])
        st.mark_window(axp, float(t_obs[0]), train_end, label="")
        phis = {}
        for v in variants:
            if f"{v}/phi" not in d or v not in NOMINAL_ML:
                continue
            axp.axhline(NOMINAL_ML[v], color=MODEL_STYLE[v]["color"], lw=1.2,
                        ls=(0, (4, 3)), alpha=0.75, zorder=3)
            phis[v], = axp.plot([], [], color=MODEL_STYLE[v]["color"], lw=2.2,
                                zorder=6)
        curp = axp.axvline(t[0], color=st.INK, lw=1.0, alpha=0.4, zorder=9)
        axp.set_xlim(float(t[0]), float(t[-1])); axp.set_ylim(0, 1.02)
        axp.set_xlabel("time (model units)")
        axp.set_ylabel("ML share  $1-\\Phi$")
        if j == 0:
            axp.text(0.015, 0.10, "dashed = what the weights claim   ·   "
                     "solid = what the fit realises", transform=axp.transAxes,
                     fontsize=7.4, color=st.MUTED, ha="left", va="bottom")
        art[case] = dict(lines=lines, phis=phis, cur=cur, curp=curp, col=col)

    fig.legend(*handles, ncol=6, loc="upper left", bbox_to_anchor=(0.055, 0.890),
               fontsize=8.2, columnspacing=1.4, handlelength=1.9)
    _titles(fig, "The PI-NODE surrogate: forecast, and the balance it actually "
                 "realises",
            "Fitted on the shaded window, then forecast through a second, "
            "unseen irradiation.")

    frames = []
    steps = np.linspace(1, len(t), 84).astype(int)
    for n in steps:
        for case in cases:
            a = art[case]
            for v, ln in a["lines"].items():
                ln.set_data(t[:n], d[f"{v}/y_pred"][:n, a["col"]])
            for v, ln in a["phis"].items():
                ln.set_data(t[:n], 1.0 - d[f"{v}/phi"][:n, a["col"]])
            a["cur"].set_xdata([t[n - 1]] * 2)
            a["curp"].set_xdata([t[n - 1]] * 2)
        frames.append(_grab(fig))
    plt.close(fig)
    _to_gif(frames, dst, hold=14)
