"""How much of the 3-D dynamics a reduced 0-D model can represent, at best.

Every forecast error mixes two things: what the reduced model *cannot represent*
(structural inadequacy) and what it *cannot infer* from 20 noisy samples
(estimation error).  This script separates them by fitting each backbone twice:

``oracle``     to the entire noise-free trajectory on [0, 80] -- the best the
               model form can possibly do, no estimation error at all;
``assimilated`` to the 20 noisy samples on [14, 35] -- what the benchmark does.

The oracle residual is the *closure gap*: the part of the spatial dynamics that
no choice of the mechanistic parameters can reproduce, and therefore exactly the
quantity a learned closure term exists to absorb.  Reporting it turns "the
closure helps" into "the closure recovers X of the Y % that the physics cannot
express".

    PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python experiments/closure_gap.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "datasets" / "cohort"
OUT = ROOT / "results" / "closure_gap"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--restarts", type=int, default=4)
    args = ap.parse_args()

    import torch
    from cancer_sim.gpu import dataset as DS, fit as FIT, surrogates as S
    from cancer_sim.gpu.surrogates import ModelSpec, Surrogate

    OUT.mkdir(parents=True, exist_ok=True)
    pids = sorted(p.name[:-11] for p in COHORT.glob("*_curves.npz"))[:args.limit]
    # one "member" per column: the oracle sees the noise-free curve
    b = DS.build(COHORT, pids=pids, n_members=1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dense noise-free targets over the WHOLE horizon
    t_all = b.t_pred
    grid = S.refined_grid(float(t_all[0]), float(t_all[-1]), (15.0, 45.0))
    U = torch.tensor(FIT.dose_rate(grid), dtype=S.DTYPE, device=dev)
    tgt = torch.tensor(
        np.stack([np.interp(grid, t_all, b.y_true[:, c]) for c in range(b.C)], 1),
        dtype=S.DTYPE, device=dev)

    rows = []
    for backbone in ("logistic", "volumetric", "power"):
        R = args.restarts
        B = b.C * R
        init = FIT.build_init(b.C, R, seed=7)
        torch.manual_seed(7)
        gen = torch.Generator(device=dev).manual_seed(7)
        spec = ModelSpec(f"oracle_{backbone}", backbone=backbone)
        model = Surrogate(spec, B, dev, init=init, gen=gen).to(dev)
        y0 = tgt[0].repeat(R)
        Y = tgt.repeat(1, R)
        opt = torch.optim.Adam(model.parameters(), lr=1.5e-2)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, 7.5e-4)
        best = torch.full((B,), float("inf"), device=dev)
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for ep in range(args.epochs):
            opt.zero_grad(set_to_none=True)
            pred = S.integrate(model, grid, U, y0, FIT.TIME_NORM)[:, :, 0]
            per = ((pred - Y) ** 2).mean(dim=0)
            per.mean().backward()
            FIT._clip_per_member(list(model.parameters()), B)
            with torch.no_grad():
                imp = per < best
                if bool(imp.any()):
                    best = torch.where(imp, per.detach(), best)
                    for k, v in model.state_dict().items():
                        best_state[k][imp] = v[imp].clone()
            opt.step()
            sch.step()
        model.load_state_dict(best_state)
        with torch.no_grad():
            pred = S.integrate(model, grid, U, y0, FIT.TIME_NORM)[:, :, 0]
            err = ((pred - Y) ** 2).mean(dim=0).view(R, b.C).min(dim=0).values
            rel = 100.0 * (err.sqrt()
                           / (tgt.pow(2).mean(dim=0).sqrt() + 1e-9)).cpu().numpy()
            th = {k: v.detach().cpu().numpy() for k, v in model.backbone.theta().items()}
        cases = np.array([c for _, c in b.labels])
        row = {"backbone": backbone,
               "oracle_rel_rmse_median": float(np.median(rel)),
               "oracle_rel_rmse_p90": float(np.percentile(rel, 90)),
               "by_case": {c: float(np.median(rel[cases == c])) for c in b.cases},
               "q_median": float(np.median(th["q"].reshape(R, b.C)[0])),
               "K_median": float(np.median(th["K"].reshape(R, b.C)[0]))}
        rows.append(row)
        print(f"{backbone:11s} oracle rel RMSE  median {row['oracle_rel_rmse_median']:6.2f}%  "
              f"p90 {row['oracle_rel_rmse_p90']:6.2f}%   "
              + "  ".join(f"{c[:6]} {v:5.2f}" for c, v in row["by_case"].items())
              + f"   q {row['q_median']:.3f}", flush=True)

    (OUT / "closure_gap.json").write_text(json.dumps(
        {"n_patients": len(pids), "epochs": args.epochs,
         "restarts": args.restarts, "rows": rows}, indent=1))
    print(f"\nwrote {OUT/'closure_gap.json'}")
    print("The oracle residual is the irreducible closure gap: the fraction of "
          "the 3-D dynamics no choice of mechanistic parameters can express.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
