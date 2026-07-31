"""Trajectory figures: observed mass, forecasts, uncertainty bands, blending."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from . import style as st

DOSES = (15.0, 45.0)


def band(ax, t, Y, color, label=None, dashes=None, lw=2.0, alpha=0.16,
         q=(5, 95), zorder=3):
    """Median line + inter-quantile band over the noisy-ensemble members."""
    med = np.nanmedian(Y, axis=1)
    lo, hi = np.nanpercentile(Y, q[0], axis=1), np.nanpercentile(Y, q[1], axis=1)
    ax.fill_between(t, lo, hi, color=color, alpha=alpha, lw=0, zorder=zorder - 1)
    line, = ax.plot(t, med, color=color, lw=lw, zorder=zorder, label=label)
    if dashes:
        line.set_dashes(dashes)
    return med


def forecast_panel(ax, bench, preds: Dict[str, np.ndarray], col: int,
                   train_end: float = 35.0, show_obs: bool = True,
                   show_no_treat: bool = True, direct_labels: bool = True):
    """One (patient, configuration): truth, observations and every forecast."""
    t = bench.t_pred
    M = bench.n_members
    sl = slice(col * M, (col + 1) * M)

    st.mark_window(ax, t[0], train_end)
    st.mark_doses(ax, DOSES)

    if show_no_treat:
        ax.plot(t, bench.y_no_treat[:, col], color=st.MUTED, lw=1.4,
                dashes=(1, 2), zorder=2)
        if direct_labels:
            st.label_end(ax, t[-1], bench.y_no_treat[-1, col], " untreated",
                         st.MUTED)

    ax.plot(t, bench.y_true[:, col], color=st.TRUTH, lw=2.6, zorder=6,
            solid_capstyle="round")
    if direct_labels:
        st.label_end(ax, t[-1], bench.y_true[-1, col], " truth", st.TRUTH)

    if show_obs:
        ax.plot(bench.t_obs, bench.Y_obs_clean[:, col], "o", ms=4.0,
                mfc="#ffffff", mec=st.TRUTH, mew=1.2, zorder=7)

    for k, (name, Y) in enumerate(preds.items()):
        s = st.method_style(name, k)
        med = band(ax, t, Y[:, sl], s["color"], dashes=s["dashes"], zorder=4 + k)
        if direct_labels:
            st.label_end(ax, t[-1], med[-1], f" {name}", s["color"])

    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(bottom=0)
    return ax


def figure_forecasts(bench, preds: Dict[str, np.ndarray], pid: str, out: Path,
                     cases: Optional[Sequence[str]] = None, train_end=35.0):
    """2x2 of the beam configurations for one patient."""
    st.apply()
    cases = list(cases or bench.cases)
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.6), sharex=True)
    for k, (ax, case) in enumerate(zip(axes.ravel(), cases)):
        col = bench.index(pid, case)
        forecast_panel(ax, bench, preds, col, train_end,
                       direct_labels=False)
        cov = bench.manifests[pid]["cases"][case]["beam_coverage0"]
        ax.set_title(f"({'abcd'[k]})  {st.CASE_LABEL[case]} · "
                     f"dose coverage {cov:.0%}")
        if k // 2 == 1:
            ax.set_xlabel("time (model units)")
        if k % 2 == 0:
            ax.set_ylabel("tumour mass  $y(t)/y(0)$")

    handles = [Line2D([], [], color=st.TRUTH, lw=2.6, label="3-D PDE ground truth"),
               Line2D([], [], color=st.TRUTH, lw=0, marker="o", ms=4,
                      mfc="#fff", mew=1.2, label="assimilation samples"),
               Line2D([], [], color=st.MUTED, lw=1.4, dashes=(1, 2),
                      label="untreated counterfactual")]
    for k, name in enumerate(preds):
        s = st.method_style(name, k)
        ln = Line2D([], [], color=s["color"], lw=2.0, label=name)
        if s["dashes"][0]:
            ln.set_dashes(s["dashes"])
        handles.append(ln)
    fig.legend(handles=handles, loc="lower center", ncol=min(5, len(handles)),
               frameon=False, bbox_to_anchor=(0.5, -0.055))
    st.titles(fig, f"Cumulative tumour mass under radiotherapy · patient {pid}",
              "median and 5–95 % band over 20 noisy assimilation realisations; "
              "the forecast window (t > 35) contains the second, unseen "
              "irradiation event", rect_top=0.90)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_weight_control(scaled: Dict[Tuple[float, float], Dict],
                          convex: Dict[float, Dict], out: Path):
    """The study's central figure: is the blending weight a real control?

    Left  -- the paper's (omega, s_r): forecast error over the grid; the
             realised physics share is printed in each cell.
    Middle-- the convex control lambda: error and realised share versus lambda.
    Right -- calibration: realised physics share against the nominal control,
             for both parameterisations, against the identity line.
    """
    st.apply()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1),
                             gridspec_kw={"width_ratios": [1.15, 1, 1]})

    # ---- (a) omega x s_r heat map -----------------------------------------
    ax = axes[0]
    oms = sorted({k[0] for k in scaled})
    srs = sorted({k[1] for k in scaled})
    grid = np.array([[scaled.get((o, s), {}).get("test", np.nan) for s in srs]
                     for o in oms])
    im = ax.imshow(grid, cmap="Blues_r", origin="lower", aspect="auto")
    ax.set_xticks(range(len(srs)), [f"{s:g}" for s in srs])
    ax.set_yticks(range(len(oms)), [f"{o:g}" for o in oms])
    ax.set_xlabel("residual scale  $s_r$")
    ax.set_ylabel("physics weight  $\\omega$")
    ax.grid(False)
    for i, o in enumerate(oms):
        for j, s in enumerate(srs):
            c = scaled.get((o, s))
            if not c:
                continue
            dark = grid[i, j] < np.nanmedian(grid)
            ax.text(j, i + 0.14, f"{c['test']:.0f}%", ha="center", va="center",
                    fontsize=8.5, fontweight="700",
                    color="#ffffff" if dark else st.INK)
            ax.text(j, i - 0.22, f"$\\Phi$={c['phi']:.2f}", ha="center",
                    va="center", fontsize=7,
                    color="#ffffff" if dark else st.INK2)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                 label="forecast rel. RMSE (%)")
    ax.set_title("(a) paper weighting: $\\dot y=\\omega f_{RT}+s_r g_\\psi$")

    # ---- (b) lambda sweep --------------------------------------------------
    ax = axes[1]
    lams = sorted(convex)
    err = [convex[l]["test"] for l in lams]
    q1 = [convex[l].get("q1", np.nan) for l in lams]
    q3 = [convex[l].get("q3", np.nan) for l in lams]
    ax.fill_between(lams, q1, q3, color=st.SERIES[0], alpha=0.15, lw=0)
    ax.plot(lams, err, color=st.SERIES[0], lw=2.2, marker="o", ms=5,
            mfc="#ffffff", mew=1.6)
    best = lams[int(np.nanargmin(err))]
    ax.axvline(best, color=st.SERIES[0], lw=1.0, ls=(0, (2, 2)), alpha=0.6)
    ax.annotate(f"$\\lambda^\\star={best:g}$", xy=(best, np.nanmin(err)),
                xytext=(0, -16), textcoords="offset points", ha="center",
                color=st.SERIES[0], fontsize=9.5, fontweight="700")
    ax.set_xlabel("closure weight  $\\lambda$")
    ax.set_ylabel("forecast rel. RMSE (%)")
    ax.set_title("(b) convex control: $\\dot y=(1{-}\\lambda)f_{RT}"
                 "+\\lambda\\,\\sigma\\tanh g_\\psi$")
    ax.text(0.02, 0.03, "pure\nphysics", transform=ax.transAxes, fontsize=7.5,
            color=st.MUTED, ha="left", va="bottom", linespacing=1.0)
    ax.text(0.98, 0.03, "pure\nclosure", transform=ax.transAxes, fontsize=7.5,
            color=st.MUTED, ha="right", va="bottom", linespacing=1.0)
    # One endpoint is bimodal across the cohort; let its band run off-axis
    # rather than compressing the curve everyone reads.
    hi = float(np.nanpercentile([convex[l].get("q3", convex[l]["test"])
                                 for l in lams], 85))
    ax.set_ylim(top=max(hi * 1.15, 1.25 * float(np.nanmax(err))))

    # ---- (c) calibration ---------------------------------------------------
    ax = axes[2]
    ax.plot([0, 1], [0, 1], color=st.MUTED, lw=1.0, ls=(0, (3, 3)), zorder=1)
    ax.text(0.70, 0.62, "the knob means\nwhat it says", color=st.MUTED,
            fontsize=7.5, rotation=38, ha="center", va="center",
            rotation_mode="anchor")
    lam_phi = [1.0 - convex[l]["phi"] for l in lams]
    ax.plot(lams, lam_phi, color=st.SERIES[0], lw=2.2, marker="o", ms=5,
            mfc="#ffffff", mew=1.6, zorder=4, label="convex  $\\lambda$")
    xs, ys = [], []
    for (o, s), c in sorted(scaled.items()):
        nominal = s / (o + s)
        xs.append(nominal)
        ys.append(1.0 - c["phi"])
    ax.plot(xs, ys, "s", ms=5, color=st.SERIES[1], mfc="#ffffff", mew=1.6,
            ls="none", zorder=3, label="paper  $s_r/(\\omega{+}s_r)$")
    ax.set_xlabel("nominal ML share")
    ax.set_ylabel("realised ML share  $1-\\Phi$")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="upper left")
    ax.set_title("(c) does the knob mean what it says?")

    st.titles(fig,
              "Controlling the physics–ML balance in a reduced radiotherapy "
              "surrogate",
              "cohort medians over all patients and beam configurations; "
              "$\\Phi$ is the realised physics share of the fitted vector field",
              rect_top=0.88)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_cohort(summary: List[Dict], out: Path, metric: str = "test",
                  label: str = "forecast rel. RMSE (%)", top: int = 14,
                  keep: Sequence[str] = ()):
    """Per-method distribution over the cohort, split by beam configuration.

    (a) is a ranked distribution -- identity comes from the axis label, so it
    scales to any number of methods.  (b) is a *sequential* heat map rather than
    grouped bars: magnitude is the encoding, so one hue light-to-dark is correct
    and, unlike a categorical bar palette, it does not run out of colours.
    """
    st.apply()
    rank = sorted(range(len(summary)),
                  key=lambda i: np.nanmedian(summary[i][metric]))
    keep = set(keep)
    sel = [i for i in rank[:top]]
    for i in rank[top:]:
        if summary[i]["method"] in keep:
            sel.append(i)
    sel = sorted(sel, key=lambda i: np.nanmedian(summary[i][metric]))
    methods = [summary[i]["method"] for i in sel]
    cases = list(summary[0]["by_case"])
    n = len(sel)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 0.34 * n + 2.3),
                             gridspec_kw={"width_ratios": [1.5, 1]})

    # ---- (a) ranked distribution ------------------------------------------
    ax = axes[0]
    best = np.nanmedian(summary[sel[0]][metric])
    for k, i in enumerate(sel):
        v = np.asarray(summary[i][metric], dtype=float)
        v = v[np.isfinite(v)]
        c = st.method_style(methods[k], k)["color"]
        ax.boxplot([v], positions=[n - 1 - k], vert=False, widths=0.6,
                   patch_artist=True, showfliers=False, whis=(10, 90),
                   medianprops=dict(color=st.INK, lw=1.5),
                   boxprops=dict(facecolor=c, alpha=0.32, lw=0),
                   whiskerprops=dict(color=c, lw=1.1),
                   capprops=dict(color=c, lw=1.1))
        m = np.nanmedian(v)
        ax.plot(m, n - 1 - k, "o", ms=5.5, color=c, mec="#ffffff", mew=1.3,
                zorder=5)
        ax.text(m, n - 1 - k + 0.36, f"{m:.1f}", fontsize=7.2, color=st.INK2,
                ha="center")
    ax.axvline(best, color=st.MUTED, lw=0.9, ls=(0, (3, 3)), zorder=0)
    ax.set_yticks(range(n), methods[::-1])
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel(label)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    # One bimodal method (a closure with no physics fails outright on the
    # covered geometries) would otherwise stretch the axis and flatten the rest.
    # Clip, and mark every box that runs past the edge so nothing is hidden.
    q90 = [np.nanpercentile(np.asarray(summary[i][metric], float), 90)
           for i in sel]
    xmax = float(np.nanpercentile(q90, 75) * 1.6)
    if max(q90) > xmax:
        ax.set_xlim(right=xmax)
        for k in range(n):
            if q90[k] > xmax:
                ax.annotate("", xy=(xmax, n - 1 - k), xytext=(-14, 0),
                            textcoords="offset points", zorder=6,
                            arrowprops=dict(arrowstyle="-|>", lw=1.4,
                                            color=st.method_style(methods[k], k)["color"]))
                ax.text(xmax, n - 1 - k - 0.42, f"90 % → {q90[k]:.0f}",
                        fontsize=6.8, color=st.INK2, ha="right", va="top")
    ax.set_title(f"(a) cohort distribution — top {n} "
                 f"(box: IQR, whiskers: 10–90 %)")

    # ---- (b) magnitude by configuration ------------------------------------
    ax = axes[1]
    grid = np.array([[np.nanmedian(summary[i]["by_case"][c]) for c in cases]
                     for i in sel])
    im = ax.imshow(grid, cmap="Blues", aspect="auto",
                   vmin=np.nanmin(grid), vmax=np.nanpercentile(grid, 97))
    hi = np.nanmedian(grid)
    for r in range(grid.shape[0]):
        for cix in range(grid.shape[1]):
            ax.text(cix, r, f"{grid[r, cix]:.0f}", ha="center", va="center",
                    fontsize=7.4, fontweight="600",
                    color="#ffffff" if grid[r, cix] > hi else st.INK)
    ax.set_xticks(range(len(cases)),
                  [st.CASE_LABEL[c].replace(" ", "\n") for c in cases],
                  fontsize=8)
    ax.set_yticks(range(n), methods, fontsize=7.6)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label=label)
    ax.set_title("(b) median by beam–tumour configuration")

    st.titles(fig, "Reduced-surrogate forecast accuracy on real glioma anatomy",
              "forecast window (35, 80]; median over 20 noisy assimilation "
              "realisations per patient-configuration")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_error_budget(summary: List[Dict], gap: Dict, out: Path,
                        show: Sequence[str] = ()):
    """Split each method's forecast error into structural and estimation parts.

    The oracle fit (whole noise-free trajectory) bounds what the model form can
    do; anything above it is what twenty noisy samples fail to determine. The
    decomposition is what makes "the closure helps" a testable claim rather than
    a hope: added flexibility must buy more than the estimation variance it costs.
    """
    st.apply()
    by_bb = {r["backbone"]: r["oracle_rel_rmse_median"] for r in gap["rows"]}
    rows = [s for s in summary if not show or s["method"] in show]
    rows = sorted(rows, key=lambda r: np.nanmedian(r["test"]))
    n = len(rows)

    fig, ax = plt.subplots(figsize=(9.2, 0.36 * n + 2.4))
    n_nophys = 0
    for k, r in enumerate(rows):
        total = float(np.nanmedian(r["test"]))
        spec = r.get("spec") or {}
        y = n - 1 - k
        # The oracle residual is a property of the *mechanistic* form, so the
        # split is only meaningful for models that have one. Purely learned
        # fields (NODE, lambda = 1) get a single undifferentiated bar.
        has_phys = (spec.get("family", "phys") != "node"
                    and not (spec.get("blend") == "convex"
                             and float(spec.get("lam", 0)) >= 1.0))
        if has_phys:
            struct = min(by_bb.get(spec.get("backbone", "logistic"),
                                   min(by_bb.values())), total)
            ax.barh(y, struct, height=0.62, color=st.SERIES[7], alpha=0.85,
                    lw=0, zorder=3)
            ax.barh(y, total - struct, left=struct, height=0.62,
                    color=st.SERIES[0], alpha=0.55, lw=0, zorder=3)
        else:
            n_nophys += 1
            ax.barh(y, total, height=0.62, color=st.MUTED, alpha=0.45, lw=0,
                    zorder=3)
        ax.text(total + 0.4, y, f"{total:.1f}", va="center", fontsize=7.6,
                color=st.INK2)
    ax.set_yticks(range(n), [r["method"] for r in rows][::-1])
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(0, 1.12 * max(float(np.nanmedian(r["test"])) for r in rows))
    ax.set_xlabel("forecast rel. RMSE (%)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=st.SERIES[7], alpha=0.85,
              label="structural: irreducible for this mechanistic form"),
        Patch(facecolor=st.SERIES[0], alpha=0.55,
              label="estimation: what 20 noisy samples cannot determine")]
    if n_nophys:
        handles.append(Patch(facecolor=st.MUTED, alpha=0.45,
                             label="no mechanistic branch — split undefined"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=8)
    st.titles(fig, "Where the forecast error comes from",
              "the structural part is the residual of an oracle fit to the "
              "entire noise-free trajectory; everything above it is estimation")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_gate(bench, lam: np.ndarray, phi: np.ndarray, preds, pid: str,
                out: Path, case: str = "narrow_centered"):
    """Where the learned closure takes over from the physics, in time."""
    st.apply()
    col = bench.index(pid, case)
    M = bench.n_members
    sl = slice(col * M, (col + 1) * M)
    t = bench.t_pred

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.4, 1]})
    ax = axes[0]
    st.mark_window(ax, t[0], 35.0)
    st.mark_doses(ax, DOSES)
    ax.plot(t, bench.y_true[:, col], color=st.TRUTH, lw=2.6, zorder=6)
    for k, (name, Y) in enumerate(preds.items()):
        s = st.method_style(name, k)
        band(ax, t, Y[:, sl], s["color"], dashes=s["dashes"], label=name)
    ax.set_ylabel("tumour mass $y(t)/y(0)$")
    ax.legend(loc="upper center", ncol=len(preds) + 1, fontsize=7.5,
              bbox_to_anchor=(0.5, -0.02))
    ax.set_title("(a) forecast")

    ax = axes[1]
    st.mark_window(ax, t[0], 35.0, label="")
    st.mark_doses(ax, DOSES, label=False)
    band(ax, t, lam[:, sl], st.SERIES[0], label="gate $\\lambda(t)$")
    band(ax, t, 1.0 - phi[:, sl], st.SERIES[1],
         dashes=(5, 2), label="realised ML share $1-\\Phi(t)$")
    ax.axhline(0.5, color=st.MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.set_ylim(0, 1)
    ax.set_ylabel("closure authority")
    ax.set_xlabel("time (model units)")
    ax.legend(loc="upper left", ncol=2)
    ax.set_title("(b) how much of the dynamics the closure supplies")

    lam_mean = float(np.nanmean(lam[:, sl]))
    verdict = ("with this gate penalty the closure is switched off almost "
               "everywhere, so the model reduces to its mechanistic backbone"
               if lam_mean < 0.05 else
               f"the closure supplies a mean {lam_mean:.0%} of the derivative")
    st.titles(fig, f"Learned physics–ML gating · patient {pid} · "
              f"{st.CASE_LABEL[case]}",
              "(b) is the share of the derivative the closure actually "
              f"supplies, per state and per time — {verdict}", rect_top=0.91)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
