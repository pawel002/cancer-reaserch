"""Generate the ground-truth cohort across all available GPUs.

    PYTHONPATH=src python experiments/gen_cohort.py --gpus 4,5,6,7

For every QC-passing patient this runs, on one GPU: a growth-rate calibration
sim, an untreated counterfactual, and the four tumour--beam configurations of
the paper, writing curves / fields / manifest into ``datasets/cohort``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATIENT_ROOT = ROOT / "real_data" / "data_GliODIL_essential"
OUT = ROOT / "datasets" / "cohort"

# Patients kept at full 3-D resolution for the volume renderings.
SHOWCASE = ("data_001", "data_013", "data_407")


def worker(pids, gpu: int, grid: int, out: Path, gamma: float, hypoxia: float):
    import numpy as np  # noqa: F401  (import inside worker after CUDA pinning)
    from cancer_sim.realdata import cohort as C
    from cancer_sim.realdata.fk_rt_gpu import RTConfig

    rt = RTConfig(gamma=gamma, hypoxia=hypoxia)
    done = []
    for i, pid in enumerate(pids):
        t0 = time.time()
        try:
            m = C.generate_patient(pid, PATIENT_ROOT, out, "cuda", grid=grid,
                                   rt=rt, keep_volumes=pid in SHOWCASE)
            cases = m["cases"]
            print(f"[gpu{gpu}] {i+1}/{len(pids)} {pid} "
                  f"f={m['growth']['f']:.4f} noRT={cases['no_treatment']['final_mass_ratio']:.2f} "
                  f"full={cases['full_cover']['final_mass_ratio']:.2f} "
                  f"narrow={cases['narrow_centered']['final_mass_ratio']:.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            done.append(pid)
        except Exception as e:                       # keep the shard alive
            print(f"[gpu{gpu}] {pid} FAILED: {type(e).__name__}: {e}", flush=True)
    return done


def main():
    ap = argparse.ArgumentParser()
    # Devices 0-3 on this machine are reserved for other users.
    ap.add_argument("--gpus", default="4,5,6,7")
    ap.add_argument("--grid", type=int, default=160)
    ap.add_argument("--gamma", type=float, default=1.25)
    ap.add_argument("--hypoxia", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--patients", default="")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--shard", type=int, default=-1, help="internal: worker index")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gpus = [int(g) for g in args.gpus.split(",") if g != ""]

    screening = out / "screening.json"
    if args.patients:
        pids = args.patients.split(",")
    else:
        if not screening.exists():
            from cancer_sim.realdata import cohort as C
            rows = C.screen_cohort(PATIENT_ROOT)
            screening.write_text(json.dumps(rows, indent=1))
        rows = json.loads(screening.read_text())
        pids = [r["pid"] for r in rows if r["pass"]]
    if args.limit:
        pids = pids[:args.limit]

    if args.shard >= 0:                              # worker process
        shard = pids[args.shard::len(gpus)]
        worker(shard, gpus[args.shard], args.grid, out, args.gamma, args.hypoxia)
        return 0

    print(f"cohort: {len(pids)} patients over {len(gpus)} GPUs -> {out}")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "src")])
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[v] = "4"
    procs, t0 = [], time.time()
    for k, gpu in enumerate(gpus):
        e = dict(env)
        e["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log = open(out / f"gen_gpu{gpu}.log", "w")
        cmd = [sys.executable, __file__, "--shard", str(k), "--gpus", args.gpus,
               "--grid", str(args.grid), "--gamma", str(args.gamma),
               "--hypoxia", str(args.hypoxia), "--out", str(out),
               "--patients", ",".join(pids)]
        procs.append((gpu, subprocess.Popen(cmd, env=e, stdout=log,
                                            stderr=subprocess.STDOUT), log))
    rc = 0
    for gpu, p, log in procs:
        p.wait()
        log.close()
        rc |= p.returncode
        print(f"gpu{gpu} rc={p.returncode} (+{time.time()-t0:.0f}s)")
    n = len(list(out.glob("*_manifest.json")))
    print(f"done in {time.time()-t0:.0f}s; {n} manifests in {out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
