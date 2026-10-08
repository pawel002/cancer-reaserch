"""Czy omega i s_r da sie po prostu DOPASOWAC zamiast szukac siatka?

Teoria (rozdz. 5 raportu): kierunek (omega, theta) -> (omega/c, c theta) jest
plaski, wiec gradient nie zidentyfikuje "prawdziwej" wartosci -- optymalizator
moze sie po nim slizgac dowolnie.  Ten skrypt sprawdza trzy rzeczy naraz:

  1. czy uczenie dogania najlepsza komorke siatki (wtedy grid search zbedny),
  2. gdzie wagi laduja i czy punkt ladowania zalezy od startu
     (jesli to cechowania -- powinien zalezec),
  3. czy koncowy blad zalezy od startu.

    PYTHONPATH=src python experiments/trained_weights.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "results" / "benchmark"
OUT = ROOT / "results" / "trained_weights"

INIT = {"train_both_from_pub": (0.05, 0.10), "train_both_from_mid": (0.02, 0.02),
        "train_both_from_best": (0.005, 0.005), "train_both_from_bad": (1.0, 1.0),
        "train_sr_from_pub": (0.05, 0.10), "train_om_from_pub": (0.05, 0.10),
        "fix_pub": (0.05, 0.10), "fix_best": (0.005, 0.005)}


def err(group: str):
    d = json.loads((BENCH / group / "bench.json").read_text())
    C, M = int(d["C"]), int(d["n_members"])
    out = {}
    for f in sorted(glob.glob(str(BENCH / group / "pred_*.npz"))):
        name = os.path.basename(f)[5:-4]
        z = np.load(f)
        e = np.nanmedian(z["metric_test_rel_rmse"].reshape(C, M), axis=1)
        rec = {"med": float(np.nanmedian(e)),
               "iqr": float(np.nanpercentile(e, 75) - np.nanpercentile(e, 25))}
        for w in ("omega", "s_r"):
            k = f"theta_{w}"
            if k in z.files:
                v = np.nanmedian(z[k].reshape(C, M), axis=1)
                rec[w] = (float(np.nanmedian(v)),
                          float(np.nanpercentile(v, 10)),
                          float(np.nanpercentile(v, 90)))
        out[name] = rec
    return out


def main() -> int:
    if not (BENCH / "pinode4").exists():
        print("brak wynikow pinode4"); return 1
    r = err("pinode4")

    print("=" * 78)
    print("1. DOKLADNOSC: czy uczenie dogania najlepsza komorke siatki?")
    print(f"{'wariant':26s}{'blad %':>9}{'IQR':>8}   start (omega, s_r)")
    for k in sorted(r, key=lambda k: r[k]["med"]):
        i = INIT.get(k, ("?", "?"))
        print(f"{k:26s}{r[k]['med']:9.2f}{r[k]['iqr']:8.1f}   ({i[0]}, {i[1]})")

    base = r.get("fix_best", {}).get("med")
    pub = r.get("fix_pub", {}).get("med")
    if base is not None:
        print(f"\n  odniesienie: siatka najlepsza {base:.2f} %, publikacyjne {pub:.2f} %")
        for k in [x for x in r if x.startswith("train")]:
            d = r[k]["med"] - base
            verdict = ("dogonil" if d <= 0.5 else
                       "blisko" if d <= 2.0 else "gorszy")
            print(f"    {k:26s} {d:+6.2f} pkt wzgledem najlepszej siatki -> {verdict}")

    print("\n" + "=" * 78)
    print("2. GDZIE LADUJA WAGI (mediana kohorty, p10-p90)")
    print(f"{'wariant':26s}{'omega: start -> koniec':>34}{'s_r: start -> koniec':>34}")
    for k in sorted(r):
        if not any(w in r[k] for w in ("omega", "s_r")):
            continue
        i = INIT.get(k, (float("nan"), float("nan")))
        cols = []
        for w, i0 in (("omega", i[0]), ("s_r", i[1])):
            if w in r[k]:
                m, lo, hi = r[k][w]
                cols.append(f"{i0:g} -> {m:.4f} [{lo:.4f},{hi:.4f}]".rjust(34))
            else:
                cols.append("(stala)".rjust(34))
        print(f"{k:26s}" + "".join(cols))

    print("\n" + "=" * 78)
    print("3. CZY PUNKT LADOWANIA ZALEZY OD STARTU?  (test cechowania)")
    fam = [k for k in ("train_both_from_best", "train_both_from_mid",
                       "train_both_from_pub", "train_both_from_bad") if k in r]
    if len(fam) >= 2:
        for w in ("omega", "s_r"):
            vals = [(k, r[k][w][0]) for k in fam if w in r[k]]
            if len(vals) < 2:
                continue
            v = [x[1] for x in vals]
            print(f"  {w:6s} punkty ladowania: " +
                  ", ".join(f"{k.replace('train_both_from_','')}={x:.4f}"
                            for k, x in vals))
            print(f"         rozpietosc {max(v)/max(min(v),1e-12):.1f}x -> "
                  + ("ZALEZY od startu (zgodnie z teoria cechowania)"
                     if max(v)/max(min(v), 1e-12) > 1.5
                     else "zbiega do wspolnej wartosci"))
        errs = [r[k]["med"] for k in fam]
        print(f"\n  blad koncowy: {min(errs):.2f}-{max(errs):.2f} % "
              f"(rozpietosc {max(errs)-min(errs):.2f} pkt) -> "
              + ("zalezy od startu" if max(errs) - min(errs) > 1.0
                 else "praktycznie niezalezny od startu"))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "trained_weights.json").write_text(json.dumps(r, indent=1))
    print(f"\nzapisano {OUT/'trained_weights.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
