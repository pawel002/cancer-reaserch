"""Cohort-level figures: what the 3-D ground truth does, and why it is hard.

These figures characterise the *benchmark*, before any surrogate is fitted, and
establish the premise the surrogate study rests on: two beam configurations that
deliver almost the same total dose to the tumour produce very different
outcomes, so a reduced model that sees only ``U(t)`` and an effective coverage
cannot separate them without a closure term.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from . import style as st
from ..realdata.cohort import load_curves

CASES = ("full_cover", "narrow_centered", "slight_shift", "strong_shift")


def load_manifests(cohort_dir: Path) -> List[Dict]:
    return [json.loads(p.read_text())
            for p in sorted(Path(cohort_dir).glob("*_manifest.json"))]


def figure_ground_truth(cohort_dir: Path, out: Path,
                        highlight: str = "data_001"):
    """(a) cohort mass trajectories, (b) the coverage/outcome dissociation."""
    st.apply()
    mans = load_manifests(cohort_dir)
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2),
                             gridspec_kw={"width_ratios": [1.25, 1, 1]})

    # ---- (a) every patient, every configuration ---------------------------
    ax = axes[0]
    for td in (15.0, 45.0):
        ax.axvline(td, color=st.DOSE, lw=1.0, ls=(0, (2, 2)), alpha=0.75)
        ax.text(td + 0.7, 0.03, f"RT{1 if td < 30 else 2}",
                transform=ax.get_xaxis_transform(), color=st.DOSE, fontsize=7.5,
                fontweight="600", va="bottom")
    pids = [m["pid"] for m in mans]
    for pid in pids:
        c = load_curves(pid, cohort_dir)
        for case in CASES:
            ax.plot(c[case]["t"], c[case]["mass"], color=st.CASE_COLOR[case],
                    lw=0.5, alpha=0.18, zorder=2)
        ax.plot(c["no_treatment"]["t"], c["no_treatment"]["mass"],
                color=st.MUTED, lw=0.5, alpha=0.14, zorder=1)
    ch = load_curves(highlight, cohort_dir)
    for case in CASES:
        ax.plot(ch[case]["t"], ch[case]["mass"], color=st.CASE_COLOR[case],
                lw=2.2, zorder=5)
        st.label_end(ax, ch[case]["t"][-1], ch[case]["mass"][-1],
                     f" {st.CASE_LABEL[case]}", st.CASE_COLOR[case])
    ax.plot(ch["no_treatment"]["t"], ch["no_treatment"]["mass"], color=st.MUTED,
            lw=1.8, dashes=(1, 2), zorder=4)
    st.label_end(ax, ch["no_treatment"]["t"][-1], ch["no_treatment"]["mass"][-1],
                 " untreated", st.MUTED)
    ax.set_xlabel("time (model units)")
    ax.set_ylabel("tumour mass  $y(t)/y(0)$")
    ax.set_xlim(0, 80)
    ax.set_ylim(0, min(6, ax.get_ylim()[1]))
    ax.set_title(f"(a) 3-D ground truth · {len(pids)} patients × 4 fields "
                 f"(bold: patient {highlight})")

    # ---- (b) dose delivered vs outcome ------------------------------------
    ax = axes[1]
    for case in CASES:
        x = [m["cases"][case]["beam_coverage0"] for m in mans]
        y = [m["cases"][case]["final_mass_ratio"] for m in mans]
        ax.plot(x, y, "o", ms=4.5, color=st.CASE_COLOR[case], alpha=0.7,
                mew=0, label=st.CASE_LABEL[case])
    ax.axhline(1.0, color=st.MUTED, lw=1.0, ls=(0, (3, 3)))
    ax.text(0.02, 1.06, "no net change", transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=7.5, color=st.MUTED)
    ax.set_yscale("log")
    ax.set_xlabel("dose coverage of the tumour at $t=0$")
    ax.set_ylabel("mass at $t=80$  (ratio)")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.set_title("(b) same dose, different outcome")

    # ---- (c) what the assimilation window cannot determine ----------------
    ax = axes[2]
    d_in, d_out, pair_lbl = window_ambiguity(cohort_dir, mans)
    ax.plot(d_in, d_out, "o", ms=4, color=st.SERIES[0], alpha=0.35, mew=0)
    thr = np.percentile(d_in, 5)
    hard = d_out[d_in <= thr]
    ax.axvspan(0, thr, color=st.WINDOW, zorder=0, lw=0)
    ax.plot(d_in[d_in <= thr], hard, "o", ms=4.5, color=st.CRITICAL, mew=0,
            alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("difference inside the assimilation window\n"
                  "(rel. RMS over $t\\in[14,35]$)")
    ax.set_ylabel("difference at $t=80$   $|\\log_2$ mass ratio$|$")
    ax.set_title("(c) the window does not determine the forecast")
    ax.text(0.04, 0.95,
            f"pairs that agree to <{100*thr:.1f} % in-window\n"
            f"still differ by up to {np.max(hard):.1f} doublings at $t=80$\n"
            f"(median {np.median(hard):.2f})",
            transform=ax.transAxes, va="top", fontsize=8, color=st.INK2)

    st.titles(fig,
              "The benchmark: real anatomy, real tumours, four irradiation "
              "geometries",
              "3-D anisotropic Fisher–Kolmogorov growth (GliODIL kernel) with a "
              "grafted radiotherapy damage field; two fractions at t = 15 and 45",
              rect_top=0.90)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def window_ambiguity(cohort_dir: Path, mans: List[Dict], max_pairs: int = 40000):
    """How much two trajectories can agree in-window yet diverge afterwards.

    This is the quantitative statement of the paper's premise: a reduced model
    sees only ``y(t)`` on the assimilation window and the dose schedule, so any
    two (patient, configuration) columns that are indistinguishable there are
    indistinguishable *to the surrogate* -- yet their forecasts differ.  The
    gap is what a physics prior or a learned closure has to supply.
    """
    Y, finals = [], []
    for m in mans:
        c = load_curves(m["pid"], cohort_dir)
        for case in CASES:
            t, y = c[case]["t"], c[case]["mass"]
            w = (t >= 14.0) & (t <= 35.0)
            Y.append(y[w])
            finals.append(max(float(y[-1]), 1e-6))
    Y = np.asarray(Y, dtype=float)
    finals = np.asarray(finals)
    n = len(Y)
    rng = np.random.default_rng(0)
    i = rng.integers(0, n, max_pairs)
    j = rng.integers(0, n, max_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    scale = np.sqrt((Y ** 2).mean(axis=1))
    d_in = (np.sqrt(((Y[i] - Y[j]) ** 2).mean(axis=1))
            / (0.5 * (scale[i] + scale[j]) + 1e-12))
    d_out = np.abs(np.log2(finals[i] / finals[j]))
    return d_in, d_out, (i, j)


def figure_growth_law(cohort_dir: Path, out: Path, n_show: int = 40):
    """Which reduced growth law the spatial ground truth actually follows."""
    st.apply()
    mans = load_manifests(cohort_dir)[:n_show]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.9))

    # (a) untreated mass, linear vs cube-root
    ax = axes[0]
    for m in mans:
        c = load_curves(m["pid"], cohort_dir)["no_treatment"]
        ax.plot(c["t"], c["mass"], color=st.SERIES[0], lw=0.8, alpha=0.35)
    ax.set_xlabel("time")
    ax.set_ylabel("mass  $y/y_0$")
    ax.set_title("(a) untreated growth")

    ax = axes[1]
    r2_cube, r2_lin, qs = [], [], []
    for m in mans:
        c = load_curves(m["pid"], cohort_dir)["no_treatment"]
        t, y = c["t"], np.maximum(c["mass"], 1e-9)
        ax.plot(t, y ** (1 / 3), color=st.SERIES[2], lw=0.8, alpha=0.35)
        r2_cube.append(_r2(t, y ** (1 / 3)))
        r2_lin.append(_r2(t, y))
        qs.append(_fit_exponent(t, y))
    ax.set_xlabel("time")
    ax.set_ylabel("$(y/y_0)^{1/3}$")
    ax.set_title(f"(b) cube root is linear in $t$\n"
                 f"median $R^2$ = {np.median(r2_cube):.4f} "
                 f"(vs {np.median(r2_lin):.4f} for the mass itself)")

    ax = axes[2]
    ax.hist(qs, bins=18, color=st.SERIES[3], alpha=0.85, lw=0)
    ax.axvline(2 / 3, color=st.CRITICAL, lw=1.8)
    ax.text(2 / 3, ax.get_ylim()[1] * 0.94, "  $q=2/3$\n  (surface growth)",
            color=st.CRITICAL, fontsize=8, va="top", fontweight="600")
    ax.axvline(1.0, color=st.MUTED, lw=1.4, ls=(0, (3, 3)))
    ax.text(1.0, ax.get_ylim()[1] * 0.55, "  $q=1$\n  (logistic)",
            color=st.MUTED, fontsize=8, va="top")
    ax.set_xlabel("effective growth exponent  $q$  in  $\\dot y \\propto y^q$")
    ax.set_ylabel("patients")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"(c) recovered from the 3-D truth\nmedian $q$ = "
                 f"{np.median(qs):.3f}")

    st.titles(fig, "The reduced growth law implied by 3-D invasion",
              "a constant-speed Fisher front makes mass grow as $(a+bt)^3$, "
              "i.e. $\\dot y \\propto y^{2/3}$ — not the logistic "
              "$\\dot y \\propto y$ the reduced model assumes", rect_top=0.87)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _r2(t, y):
    A = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    return float(1 - resid.var() / (y.var() + 1e-15))


def _fit_exponent(t, y, lo=1.05):
    """Slope of log(dy/dt) against log(y) -- the effective exponent q."""
    m = y > lo
    if m.sum() < 20:
        m = y > y[0] * 1.01
    dy = np.gradient(y, t)
    ok = m & (dy > 0)
    if ok.sum() < 10:
        return np.nan
    A = np.vstack([np.log(y[ok]), np.ones(ok.sum())]).T
    coef, *_ = np.linalg.lstsq(A, np.log(dy[ok]), rcond=None)
    return float(coef[0])
