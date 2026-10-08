"""Aggregate benchmark shards into tables and the study's figures.

    PYTHONPATH=src python experiments/aggregate_bench.py --group main
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "benchmark"
COHORT = ROOT / "datasets" / "cohort"


def load_group(group_dir: Path, bench) -> Dict[str, Dict]:
    """Read every ``pred_*.npz`` in a group into per-method result dicts."""
    out = {}
    for p in sorted(group_dir.glob("pred_*.npz")):
        name = p.name[5:-4]
        z = np.load(p)
        spec_p = group_dir / f"spec_{name}.json"
        spec = json.loads(spec_p.read_text()) if spec_p.exists() else {}
        m = {k[7:]: z[k] for k in z.files if k.startswith("metric_")}
        out[name] = {
            "name": name, "spec": spec, "metrics": m,
            "y_pred": z["y_pred"], "lam": z["lam"], "phys_share": z["phys_share"],
            "theta": {k[6:]: z[k] for k in z.files if k.startswith("theta_")},
        }
    return out


def summarise(res: Dict, bench) -> Dict:
    """Per-method cohort summary: medians overall and by beam configuration."""
    m = res["metrics"]
    C, M = bench.C, bench.n_members
    per_case = {k: np.nanmedian(v.reshape(C, M), axis=1) for k, v in m.items()}
    cases = np.array([c for _, c in bench.labels])
    by_case = {c: per_case["test_rel_rmse"][cases == c] for c in bench.cases}
    by_case_final = {c: per_case["abs_final_err"][cases == c] for c in bench.cases}
    phi = res["phys_share"]
    t = bench.t_pred
    return {
        "method": res["name"],
        "spec": res["spec"],
        "test": per_case["test_rel_rmse"],
        "train": per_case["train_rel_rmse"],
        "final": per_case["abs_final_err"],
        "by_case": by_case,
        "by_case_final": by_case_final,
        "phi_mean": float(np.nanmean(phi)),
        "phi_in": float(np.nanmean(phi[t <= 35.0])),
        "phi_out": float(np.nanmean(phi[t > 35.0])),
        "lam_mean": float(np.nanmean(res["lam"])) if res["lam"].size > 1 else np.nan,
        "fit_time_s": res["spec"].get("fit_time_s", np.nan),
    }


def table(rows: List[Dict], sort_key="test") -> str:
    rows = sorted(rows, key=lambda r: np.nanmedian(r[sort_key]))
    hdr = (f"{'method':26s} {'test%':>7s} {'IQR':>6s} {'|e80|%':>7s} "
           f"{'train%':>7s} {'Phi_in':>7s} {'Phi_out':>7s} {'s':>6s}   "
           + "  ".join(f"{c[:9]:>9s}" for c in rows[0]["by_case"]))
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        v = np.asarray(r["test"], float)
        iqr = np.nanpercentile(v, 75) - np.nanpercentile(v, 25)
        lines.append(
            f"{r['method']:26s} {np.nanmedian(v):7.2f} {iqr:6.2f} "
            f"{np.nanmedian(r['final']):7.1f} {np.nanmedian(r['train']):7.2f} "
            f"{r['phi_in']:7.2f} {r['phi_out']:7.2f} {r['fit_time_s']:6.0f}   "
            + "  ".join(f"{np.nanmedian(r['by_case'][c]):9.2f}" for c in r["by_case"]))
    return "\n".join(lines)


def write_csv(rows: List[Dict], path: Path):
    import csv
    cases = list(rows[0]["by_case"])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "test_rel_rmse_median", "test_iqr",
                    "abs_final_err_median", "train_rel_rmse_median",
                    "phi_in", "phi_out", "lam_mean", "fit_time_s"]
                   + [f"test_{c}" for c in cases]
                   + [f"e80_{c}" for c in cases])
        for r in sorted(rows, key=lambda r: np.nanmedian(r["test"])):
            v = np.asarray(r["test"], float)
            w.writerow([r["method"], f"{np.nanmedian(v):.3f}",
                        f"{np.nanpercentile(v,75)-np.nanpercentile(v,25):.3f}",
                        f"{np.nanmedian(r['final']):.3f}",
                        f"{np.nanmedian(r['train']):.3f}",
                        f"{r['phi_in']:.4f}", f"{r['phi_out']:.4f}",
                        f"{r['lam_mean']:.4f}", f"{r['fit_time_s']:.1f}"]
                       + [f"{np.nanmedian(r['by_case'][c]):.3f}" for c in cases]
                       + [f"{np.nanmedian(r['by_case_final'][c]):.3f}" for c in cases])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="main")
    ap.add_argument("--members", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from cancer_sim.gpu import dataset as DS
    gdir = RESULTS / args.group
    meta_p = gdir / "bench.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        bench = DS.build(COHORT, pids=meta["pids"], cases=tuple(meta["cases"]),
                         n_members=meta["n_members"])
    else:
        pids = sorted(p.name[:-11] for p in COHORT.glob("*_curves.npz"))
        if args.limit:
            pids = pids[:args.limit]
        bench = DS.build(COHORT, pids=pids, n_members=args.members)

    res = load_group(gdir, bench)
    if not res:
        print(f"no results in {gdir}")
        return 1
    rows = [summarise(r, bench) for r in res.values()]
    print(table(rows))
    write_csv(rows, gdir / "summary.csv")
    print()
    print(paired_tests(rows, bench))
    print(f"\nwrote {gdir/'summary.csv'}  ({len(rows)} methods, "
          f"{bench.C} case-columns, {bench.n_members} members)")
    return 0


def paired_tests(rows: List[Dict], bench, reference: str = None) -> str:
    """Paired comparison of every method against the best one.

    The methods all see the *same* patient-configuration columns and the same
    noise realisations, so the comparison is paired: a Wilcoxon signed-rank test
    over the columns is the right check, and the win rate is more informative
    than the difference of medians.
    """
    from scipy.stats import wilcoxon
    rows = sorted(rows, key=lambda r: np.nanmedian(r["test"]))
    ref = next((r for r in rows if r["method"] == reference), rows[0])
    base = np.asarray(ref["test"], float)
    lines = [f"Paired against the best method ({ref['method']}, "
             f"median {np.nanmedian(base):.2f} %), n = {len(base)} "
             f"patient-configurations. 'beats ref' is the fraction of columns "
             f"on which the row is better than the reference:",
             f"{'method':26s} {'median Δ':>9s} {'beats ref':>10s} {'p':>10s}"]
    for r in rows:
        if r is ref:
            continue
        v = np.asarray(r["test"], float)
        ok = np.isfinite(v) & np.isfinite(base)
        if ok.sum() < 10:
            continue
        d = v[ok] - base[ok]
        try:
            p = wilcoxon(v[ok], base[ok]).pvalue
        except ValueError:
            p = float("nan")
        lines.append(f"{r['method']:26s} {np.median(d):+9.2f} "
                     f"{100*np.mean(d < 0):9.0f}% {p:10.2e}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
