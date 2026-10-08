"""Fit PI-NODE and the mechanistic ODE to N noisy samples, many times over.

Answers a different question than the cohort benchmark: for one patient, how do
the two models behave as the number of observations grows, and how much of their
error is *variance from the noise* rather than bias?  Each (case, N) cell draws
``--members`` independent noise realisations of the same N sample times, fits
both models to every realisation, and stores all the trajectories so the figure
can show a median and a percentile band instead of a single line.

    PYTHONPATH=src python experiments/band_fits.py --shard 0 --gpu 4
    PYTHONPATH=src python experiments/band_fits.py            # launches all shards
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "datasets" / "cohort"
OUT = ROOT / "results" / "bands"

CASES = ("narrow_centered", "full_cover")
# Reference model whose per-column errors define "the median patient".
PICK_FROM = ("pinode", "grid_w0.005_s0.005")


def median_patients():
    """One patient per geometry: the cohort *median* performer for it.

    Derived at run time rather than hard-coded -- the median moves whenever the
    benchmark is recomputed, and a stale hard-coded id would quietly make the
    figure unrepresentative.
    """
    from cancer_sim.gpu import dataset as DS
    b = DS.build(COHORT, n_members=20)
    e = np.load(ROOT / "results" / "benchmark" / PICK_FROM[0] /
                f"pred_{PICK_FROM[1]}.npz")["metric_test_rel_rmse"]
    e = np.nanmedian(e.reshape(b.C, b.n_members), axis=1)
    out = {}
    for case in CASES:
        idx = [i for i, l in enumerate(b.labels) if l[1] == case]
        v = e[idx]
        out[case] = b.labels[idx[int(np.argsort(v)[len(v) // 2])]][0]
    return out
N_OBS = (5, 10, 15, 20)

# sigma = 5 % of the local value, so +-2 sigma covers +-10 % -- the cap asked for.
NOISE_STD = 0.05

MODELS = [
    ("PI-NODE", dict(backbone="volumetric", blend="scaled",
                     omega=0.005, s_r=0.005)),
    ("NODE", dict(family="node")),
    ("ODE", dict(backbone="volumetric")),
]

SHARDS = list(itertools.product(CASES, N_OBS))


def run_shard(shard: int, gpu: int, members: int, epochs: int, restarts: int):
    from cancer_sim.gpu import dataset as DS, fit as FIT
    from cancer_sim.gpu.surrogates import ModelSpec

    case, n_obs = SHARDS[shard]
    pid = median_patients()[case]
    bench = DS.build(COHORT, pids=[pid], cases=[case], n_members=members,
                     n_fit=n_obs, noise_std=NOISE_STD)
    dev = f"cuda:{gpu}"
    print(f"[shard {shard}] {case} pid={pid} N={n_obs} "
          f"members={members} -> {dev}", flush=True)

    store = {
        "t_pred": bench.t_pred.astype(np.float32),
        "y_true": bench.y_true[:, 0].astype(np.float32),
        "t_obs": bench.t_obs.astype(np.float32),
        "Y_obs": bench.Y_obs.astype(np.float32),          # (N, members)
        "y_no_treat": bench.y_no_treat[:, 0].astype(np.float32),
        "U_pred": bench.U_pred.astype(np.float32),
    }
    for name, kw in MODELS:
        spec = ModelSpec(name=name, epochs=epochs, restarts=restarts, **kw)
        res = FIT.fit(spec, bench, dev)
        sc = FIT.score(res, bench)
        store[f"{name}/y_pred"] = res.y_pred.astype(np.float32)   # (T, members)
        store[f"{name}/test"] = np.asarray(sc["test_rel_rmse"], dtype=np.float32)
        print(f"  {name:8s} test rel-RMSE median "
              f"{np.nanmedian(sc['test_rel_rmse']):6.2f} %  ({res.fit_time_s:.0f} s)",
              flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / f"band_{case}_n{n_obs}.npz"
    np.savez_compressed(dst, **store)
    (OUT / f"band_{case}_n{n_obs}.json").write_text(json.dumps(
        {"case": case, "pid": pid, "n_obs": n_obs, "members": members,
         "noise_std": NOISE_STD, "epochs": epochs, "restarts": restarts}, indent=1))
    print(f"[shard {shard}] wrote {dst.name}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--gpus", default="4,5,6,7")
    ap.add_argument("--members", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=1200)
    ap.add_argument("--restarts", type=int, default=3)
    args = ap.parse_args()

    if args.shard >= 0:
        run_shard(args.shard, args.gpu, args.members, args.epochs, args.restarts)
        return 0

    gpus = [int(g) for g in args.gpus.split(",")]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "src")])
    OUT.mkdir(parents=True, exist_ok=True)
    procs = []
    for i, (case, n) in enumerate(SHARDS):
        gpu = gpus[i % len(gpus)]
        log = open(OUT / f"shard{i}.log", "w")
        procs.append(subprocess.Popen(
            [sys.executable, "-u", __file__, "--shard", str(i), "--gpu", str(gpu),
             "--members", str(args.members), "--epochs", str(args.epochs),
             "--restarts", str(args.restarts)],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
    print(f"launched {len(procs)} shards over GPUs {gpus}", flush=True)
    rc = [p.wait() for p in procs]
    print("exit codes:", rc)
    return max(rc) if rc else 0


if __name__ == "__main__":
    raise SystemExit(main())
