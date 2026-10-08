"""Liczby do opisu rysunków z pasmami: obciążenie vs wariancja w funkcji N."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BANDS = ROOT / "results" / "bands"
LO, HI = 25, 75
MODELS = ("PI-NODE", "NODE", "ODE")


def main() -> int:
    out = {}
    for case in ("narrow_centered", "full_cover"):
        rows = []
        for n in (5, 10, 15, 20):
            f = BANDS / f"band_{case}_n{n}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            t, gt = z["t_pred"], z["y_true"]
            r = {"n": n}
            for m in MODELS:
                Y = z[f"{m}/y_pred"]
                lo, med, hi = np.nanpercentile(Y, [LO, 50, HI], axis=1)
                # szerokosc pasma na koncu horyzontu, w jednostkach masy
                r[f"{m}_band80"] = float(hi[-1] - lo[-1])
                # blad samej MEDIANY trajektorii -> obciazenie
                r[f"{m}_bias80"] = float(abs(med[-1] - gt[-1]))
                r[f"{m}_bias_rel"] = float(100 * abs(med[-1] - gt[-1]) / gt[-1])
                r[f"{m}_err"] = float(np.nanmedian(z[f"{m}/test"]))
            rows.append(r)
        out[case] = rows

        gt80 = float(np.load(BANDS / f"band_{case}_n{rows[0]['n']}.npz")["y_true"][-1])
        print(f"\n=== {case} (prawda w t=80: {gt80:.2f}) ===")
        head = f"{'N':>3}  " + "".join(f"{m:^30}" for m in MODELS)
        sub = f"{'':>3}  " + "".join(f"{'blad%':>7}{'pasmo':>8}{'obc.':>7}{'obc.%':>8}"
                                     for _ in MODELS)
        print(head); print(sub)
        for r in rows:
            line = f"{r['n']:>3}  "
            for m in MODELS:
                line += (f"{r[m+'_err']:>7.1f}{r[m+'_band80']:>8.2f}"
                         f"{r[m+'_bias80']:>7.2f}{r[m+'_bias_rel']:>8.1f}")
            print(line)

    (BANDS / "band_summary.json").write_text(json.dumps(out, indent=1))
    print(f"\nzapisano {BANDS/'band_summary.json'}")

    # rysunek z pasmami -> results/figures oraz report/src
    from cancer_sim.viz import bands as BV
    figs = ROOT / "results" / "figures"
    BV.all_figures(BANDS, figs)
    rep = ROOT / "report" / "src"
    rep.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(figs / "fig_bands_narrow.png",
                    rep / "rys4-pasma-waska-wiazka.png")
    print(f"zapisano {rep/'rys4-pasma-waska-wiazka.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
