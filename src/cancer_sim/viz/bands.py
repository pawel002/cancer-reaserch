"""Uncertainty bands: how each surrogate behaves as the observations get sparser.

One panel per observation count.  Within a panel every curve is the *same*
patient and the *same* sample times; what varies across the ensemble is the
noise realisation.  The band is therefore exactly the spread the noise induces
in the forecast, which a single fitted line cannot show.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import style as st

MODEL_STYLE = {
    "PI-NODE": dict(color=st.SERIES[0], label="PI-NODE"),
    "NODE":    dict(color=st.SERIES[1], label="NODE"),
    "ODE":     dict(color=st.SERIES[2], label="ODE"),
}
CASE_TITLE = {"narrow_centered": "narrow centered", "full_cover": "full cover"}
LO, HI = 25, 75          # pasmo miedzykwartylowe
TRAIN_END = 35.0


def _load(d: Path, case: str, n: int) -> Dict:
    z = np.load(d / f"band_{case}_n{n}.npz")
    return {k: z[k] for k in z.files}


def _panel(ax, d: Dict, n: int, show_ylabel: bool, ymax: float) -> None:
    t, gt = d["t_pred"], d["y_true"]
    st.mark_window(ax, float(d["t_obs"][0]), TRAIN_END, label="")
    st.mark_doses(ax, [15.0, 45.0], label=False)

    for name, sty in MODEL_STYLE.items():
        key = f"{name}/y_pred"
        if key not in d:
            continue
        Y = d[key]                                   # (T, members)
        lo, med, hi = np.nanpercentile(Y, [LO, 50, HI], axis=1)
        ax.fill_between(t, lo, hi, color=sty["color"], alpha=0.20, lw=0, zorder=3)
        ax.plot(t, med, color=sty["color"], lw=2.0, zorder=6, label=sty["label"])

    ax.plot(t, gt, color=st.TRUTH, lw=2.4, zorder=7, label="baseline")
    # the observations one ensemble member actually saw
    ax.plot(d["t_obs"], d["Y_obs"][:, 0], "o", color=st.MUTED, ms=4.0,
            mec="#ffffff", mew=0.8, zorder=8, label="zaszumione obserwacje")

    err = {k: float(np.nanmedian(d[f"{k}/test"])) for k in MODEL_STYLE
           if f"{k}/test" in d}
    txt = "   ".join(f"{MODEL_STYLE[k]['label'].split(' (')[0]} {v:.1f}%"
                     for k, v in err.items())
    ax.set_title(f"N = {n} obserwacji", fontsize=10, loc="left", pad=15)
    ax.text(0.0, 1.005, txt, transform=ax.transAxes, fontsize=7.6,
            color=st.MUTED, ha="left", va="baseline")

    ax.set_xlim(float(t[0]), float(t[-1]))
    ax.set_ylim(0, ymax)          # wspólna dla wszystkich paneli
    ax.set_xlabel("czas (jednostki modelu)")
    if show_ylabel:
        ax.set_ylabel("masa guza  $y(t)/y(0)$")


def figure(band_dir: Path, case: str, dst: Path,
           n_obs: Sequence[int] = (5, 10, 15, 20)) -> None:
    band_dir = Path(band_dir)
    data = {n: _load(band_dir, case, n) for n in n_obs
            if (band_dir / f"band_{case}_n{n}.npz").exists()}
    if not data:
        print(f"  (brak danych dla {case})")
        return
    import json
    meta = json.loads((band_dir / f"band_{case}_n{list(data)[0]}.json").read_text())

    # jedna skala osi Y dla wszystkich paneli -- inaczej pasma z różnych N
    # nie dałyby się porównać wzrokowo
    ymax = 1.08 * max(
        max(float(np.nanmax(d["y_true"])) for d in data.values()),
        max(float(np.nanpercentile(d[f"{k}/y_pred"], HI))
            for d in data.values() for k in MODEL_STYLE
            if f"{k}/y_pred" in d))
    ymax = min(ymax, 6.0)

    st.apply(1.0)
    LEFT = 0.070          # wspolna lewa krawedz: osie, tytul, legenda
    fig = plt.figure(figsize=(11.0, 6.2), dpi=150)
    gs = fig.add_gridspec(2, 2, hspace=0.33, wspace=0.17, left=LEFT,
                          right=0.988, top=0.818, bottom=0.094)
    axes = []
    for k, (n, d) in enumerate(sorted(data.items())):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        _panel(ax, d, n, show_ylabel=(k % 2 == 0), ymax=ymax)
        axes.append(ax)

    h, l = axes[0].get_legend_handles_labels()
    order = [l.index(x) for x in ["baseline", "zaszumione obserwacje",
                                  "PI-NODE", "NODE", "ODE"] if x in l]
    fig.legend([h[i] for i in order], [l[i] for i in order], ncol=5,
               loc="upper left", bbox_to_anchor=(LEFT, 0.930), fontsize=8.4,
               columnspacing=1.3, handlelength=1.8)

    fig.text(LEFT, 0.982,
             "Wpływ liczby i zaszumienia obserwacji: "
             + CASE_TITLE.get(case, case.replace("_", " ")),
             fontsize=13.5, fontweight="700", color=st.INK, ha="left", va="top")
    fig.savefig(dst, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  -> {dst.name}")


def all_figures(band_dir: Path, out: Path) -> None:
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    for case, tag in (("narrow_centered", "narrow"), ("full_cover", "full")):
        figure(Path(band_dir), case, out / f"fig_bands_{tag}.png")
