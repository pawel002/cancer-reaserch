"""Run the surrogate benchmark over the real-patient cohort, sharded by method.

Each GPU takes a subset of the *methods* and fits that method for the entire
cohort in a single batched tensor (patients x beam configurations x noisy
realisations x restarts), which is far more efficient than sharding patients:
the per-epoch cost of the unrolled RK4 is dominated by kernel launches, not by
the batch width.

    PYTHONPATH=src python experiments/bench.py --group main
    PYTHONPATH=src python experiments/bench.py --group weights
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "datasets" / "cohort"
RESULTS = ROOT / "results" / "benchmark"


def method_specs(group: str, epochs: int, restarts: int) -> List:
    from cancer_sim.gpu.surrogates import ModelSpec
    E, R = epochs, restarts

    def S(name, **kw):
        kw.setdefault("epochs", E)
        kw.setdefault("restarts", R)
        return ModelSpec(name=name, **kw)

    if group == "pinode":
        # Everything about the paper's PI-NODE branch, Eq. (pinode_weighted):
        #   dy/dt = omega * f_RT + s_r * g_psi
        # Both weights are exact gauges, so this grid is NOT exploring model
        # classes -- it is measuring how the two scales precondition the fit.
        specs = [
            S("pub-logistic",   backbone="logistic",   blend="scaled",
              omega=0.05, s_r=0.10),
            S("pub-volumetric", backbone="volumetric", blend="scaled",
              omega=0.05, s_r=0.10),
        ]
        # (a) does the published pair sit on a slope?  Extend the grid down.
        for om in (0.005, 0.01, 0.02, 0.05):
            for sr in (0.005, 0.01, 0.02, 0.05):
                specs.append(S(f"grid_w{om}_s{sr}", backbone="volumetric",
                               blend="scaled", omega=om, s_r=sr))
        # (b) the proposed extension: both weights measured in units of the
        # patient's own rate scale, so one global pair fits a heterogeneous
        # cohort instead of only its median.
        for om in (0.25, 0.5, 1.0):
            for sr in (0.25, 0.5, 1.0):
                specs.append(S(f"sig_w{om}_s{sr}", backbone="volumetric",
                               blend="scaled", omega=om, s_r=sr,
                               scale_by_sigma=True))
        # (c) is the backbone still the second-order effect down there?
        for bb in ("logistic", "power"):
            specs.append(S(f"grid_{bb}_w0.02_s0.02", backbone=bb,
                           blend="scaled", omega=0.02, s_r=0.02))
        return specs
    if group == "pinode2":
        # Follow-up to `pinode`, driven by what that sweep showed: the optimum
        # kept moving down and the realised physics share fell to ~0.007.
        # Two questions, plus a measurement of the noise floor.
        specs = []
        # (a) does the mechanistic branch do ANY work at the optimum?
        for sr in (0.005, 0.02, 0.05):
            specs.append(S(f"w0_s{sr}", backbone="volumetric", blend="scaled",
                           omega=0.0, s_r=sr))
        # (b) how far down does the gauge keep helping?
        for om, sr in ((0.002, 0.002), (0.001, 0.001), (0.002, 0.005)):
            specs.append(S(f"grid_w{om}_s{sr}", backbone="volumetric",
                           blend="scaled", omega=om, s_r=sr))
        # (c) noise floor: identical specs to `pinode`, re-run.  Seeding is
        # fixed, so any difference is GPU non-determinism -- which sets how
        # large a gap between two settings has to be before it means anything.
        specs.append(S("rep_grid_w0.005_s0.005", backbone="volumetric",
                       blend="scaled", omega=0.005, s_r=0.005))
        specs.append(S("rep_pub_volumetric", backbone="volumetric",
                       blend="scaled", omega=0.05, s_r=0.10))
        return specs
    if group == "pinode3":
        # The reference models re-run at the SAME budget as `pinode`/`pinode2`
        # (1200 epochs).  `main` and `weights` were run at 900, and the gap is
        # not negligible -- more epochs let an unbounded closure drift further,
        # so the published PI-NODE actually gets *worse*.  Without this the
        # headline table would be comparing across budgets.
        return [
            S("ODE-vol",  backbone="volumetric"),
            S("ODE",      backbone="logistic"),
            S("NODE",     family="node"),
            S("BC-only",  backbone="volumetric", blend="convex", lam=1.0),
        ]
    if group == "pinode4":
        # Czy omega i s_r da sie po prostu DOPASOWAC zamiast szukac siatka?
        # Teoria mowi, ze kierunek (omega, theta) -> (omega/c, c theta) jest
        # plaski, wiec gradient go nie zidentyfikuje -- ale optymalizator i tak
        # gdzies wyladuje.  Pytanie brzmi: czy ladowanie jest rownie dobre jak
        # najlepsza komorka siatki, i czy zalezy od punktu startowego.
        TW = dict(backbone="volumetric", blend="scaled")
        return [
            # odniesienia: wagi ustalone (ta sama seria, wiec porownywalne)
            S("fix_pub",   omega=0.05,  s_r=0.10,  **TW),
            S("fix_best",  omega=0.005, s_r=0.005, **TW),
            # obie wagi uczone, trzy rozne punkty startowe
            S("train_both_from_pub",  omega=0.05,  s_r=0.10,
              train_weights="both", **TW),
            S("train_both_from_mid",  omega=0.02,  s_r=0.02,
              train_weights="both", **TW),
            S("train_both_from_best", omega=0.005, s_r=0.005,
              train_weights="both", **TW),
            # start celowo zly -- czy uczenie potrafi sie z niego wygrzebac?
            S("train_both_from_bad",  omega=1.0,   s_r=1.0,
              train_weights="both", **TW),
            # po jednej wadze na raz: ktora z nich cokolwiek wnosi?
            S("train_sr_from_pub",    omega=0.05,  s_r=0.10,
              train_weights="s_r", **TW),
            S("train_om_from_pub",    omega=0.05,  s_r=0.10,
              train_weights="omega", **TW),
        ]
    raise ValueError(group)


def run_worker(group: str, names: List[str], out: Path, gpu, limit: int,
               members: int, epochs: int, restarts: int):
    from cancer_sim.gpu import dataset as DS, fit as FIT

    pids = sorted(p.name[:-11] for p in COHORT.glob("*_curves.npz"))
    if limit:
        pids = pids[:limit]
    bench = DS.build(COHORT, pids=pids, n_members=members)
    # Record exactly which columns these predictions correspond to, so the
    # aggregation and figures cannot silently rebuild a different benchmark.
    (out / "bench.json").write_text(json.dumps(
        {"pids": pids, "cases": bench.cases, "n_members": members,
         "C": bench.C, "CM": bench.CM, "epochs": epochs, "restarts": restarts},
        indent=1))
    specs = {s.name: s for s in method_specs(group, epochs, restarts)}

    warm = None
    if any(specs[n].warm_start for n in names):
        from cancer_sim.gpu.surrogates import ModelSpec
        w = FIT.fit(ModelSpec(name="_warm", backbone="power",
                              epochs=max(400, epochs // 3), restarts=restarts),
                    bench, "cuda")
        warm = {k: v for k, v in w.theta.items() if k != "q"}
        print(f"[gpu{gpu}] warm start ready ({w.fit_time_s:.0f}s)", flush=True)

    for name in names:
        spec = specs[name]
        t0 = time.time()
        res = FIT.fit(spec, bench, "cuda", warm=warm if spec.warm_start else None)
        m = FIT.score(res, bench)
        np.savez_compressed(
            out / f"pred_{name}.npz", y_pred=res.y_pred.astype(np.float32),
            lam=res.lam_traj.astype(np.float32) if res.lam_traj is not None else np.zeros(1),
            phys_share=res.phys_share.astype(np.float32),
            train_mse=res.train_mse, val_mse=res.val_mse,
            loss_curve=np.asarray(res.loss_curve),
            **{f"theta_{k}": v for k, v in res.theta.items()},
            **{f"metric_{k}": v for k, v in m.items()})
        (out / f"spec_{name}.json").write_text(json.dumps(
            {**asdict(spec), "fit_time_s": res.fit_time_s,
             "n_models": int(bench.CM * spec.restarts),
             **{f"diag_{k}": v for k, v in res.extra.items()
                if not isinstance(v, np.ndarray)}}, indent=1, default=str))
        print(f"[gpu{gpu}] {name:24s} {time.time()-t0:6.0f}s "
              f"test={np.nanmedian(m['test_rel_rmse']):6.2f}% "
              f"|e80|={np.nanmedian(m['abs_final_err']):6.1f}% "
              f"physfrac={np.nanmean(res.phys_share):.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="main")
    # Devices 0-3 on this machine are reserved for other users; pass --gpus
    # explicitly to override.
    ap.add_argument("--gpus", default="4,5,6,7")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--members", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=1200)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--out", default="")
    ap.add_argument("--per-gpu", type=int, default=2,
                    help="worker processes per GPU (launch loop is "
                         "launch-bound, so >1 packs the device)")
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--names", default="")
    args = ap.parse_args()

    out = Path(args.out or (RESULTS / args.group))
    out.mkdir(parents=True, exist_ok=True)
    gpus = [int(g) for g in args.gpus.split(",") if g]

    if args.shard >= 0:
        slots = [(g, k) for k in range(args.per_gpu) for g in gpus]
        gpu, slot = slots[args.shard]
        run_worker(args.group, args.names.split(","), out, f"{gpu}.{slot}",
                   args.limit, args.members, args.epochs, args.restarts)
        return 0

    names = [s.name for s in method_specs(args.group, args.epochs, args.restarts)]
    if args.names:            # --names zawęża grupę także w launcherze,
        want = [n for n in args.names.split(',') if n]   # nie tylko w workerze
        missing = [n for n in want if n not in names]
        if missing:
            raise SystemExit(f'nieznane metody w {args.group}: {missing}')
        names = want
    # The unrolled RK4 loop is kernel-launch bound, not compute bound: a GPU
    # sits at ~30 % during a fit, so several workers share one device well.
    slots = [(g, k) for k in range(args.per_gpu) for g in gpus]
    shards = [names[i::len(slots)] for i in range(len(slots))]
    print(f"{args.group}: {len(names)} methods over {len(slots)} workers "
          f"on {len(gpus)} GPUs -> {out}")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "src")])
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[v] = "4"
    procs, t0 = [], time.time()
    for k, ((gpu, slot), shard) in enumerate(zip(slots, shards)):
        if not shard:
            continue
        e = dict(env)
        e["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log = open(out / f"bench_gpu{gpu}_{slot}.log", "w")
        cmd = [sys.executable, __file__, "--group", args.group, "--shard", str(k),
               "--gpus", args.gpus, "--per-gpu", str(args.per_gpu),
               "--names", ",".join(shard),
               "--limit", str(args.limit), "--members", str(args.members),
               "--epochs", str(args.epochs), "--restarts", str(args.restarts),
               "--out", str(out)]
        procs.append((f"{gpu}.{slot}",
                      subprocess.Popen(cmd, env=e, stdout=log,
                                       stderr=subprocess.STDOUT), log))
    rc = 0
    for tag, p, log in procs:
        p.wait(); log.close(); rc |= p.returncode
        print(f"worker {tag} rc={p.returncode} (+{time.time()-t0:.0f}s)")
    print(f"total {time.time()-t0:.0f}s")
    return rc


if __name__ == "__main__":
    sys.exit(main())
