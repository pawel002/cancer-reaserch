"""Animated cross-sections of irradiated tumour growth on real glioma anatomy.

Re-runs the GliODIL-derived forward model for one patient at a fine snapshot
cadence -- the cohort files keep only eight time points, which is enough for a
figure and far too few for an animation -- and renders the result as GIFs.

    PYTHONPATH=src python experiments/make_animations.py --pid data_001

Stage 1 (``--sim-only``) caches the dense planes to an npz so the rendering can
be iterated on without paying for the PDE again.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATIENT_ROOT = ROOT / "real_data" / "data_GliODIL_essential"
OUT = ROOT / "results" / "animations"
REPORT_FIGS = ROOT / "report" / "src"
CACHE = ROOT / "results" / "animations" / "_cache"

CASES = ("no_treatment", "full_cover", "narrow_centered")


def dense_sim(pid: str, device: str, dt_snap: float, T: float, grid: int) -> Path:
    """Run every case with snapshots every ``dt_snap`` and cache the planes."""
    from cancer_sim.realdata import patient as P, cohort as C
    from cancer_sim.realdata.fk_rt_gpu import RTConfig, simulate, dose_rate

    CACHE.mkdir(parents=True, exist_ok=True)
    dst = CACHE / f"{pid}_dense.npz"

    geom = P.load_geometry(pid, PATIENT_ROOT, grid=grid)
    gp = C.calibrate_growth(geom, device, T=T)
    rrng = P.patient_rng(pid, "rt")
    snaps = tuple(np.round(np.arange(0.0, T + 1e-9, dt_snap), 4))
    cfg = RTConfig(Dw=gp["Dw"], f=gp["f"], Dw_ratio=gp["Dw_ratio"],
                   gamma=float(rrng.uniform(*P.GAMMA_RANGE)),
                   hypoxia=float(rrng.uniform(*P.HYPOXIA_RANGE)),
                   T=T, dt_output=0.1, substeps=C._substeps_for(gp["Dw"], 0.1),
                   snapshot_times=snaps)

    ci = [int(np.clip(round(c), 0, s - 1))
          for c, s in zip(geom.centroid_vox, geom.shape)]
    store = {
        "t_snap": np.asarray(snaps, dtype=np.float32),
        "anat/wm": geom.WM[:, :, ci[2]].astype(np.float32),
        "anat/gm": geom.GM[:, :, ci[2]].astype(np.float32),
        "anat/brain": geom.brain[:, :, ci[2]].astype(np.uint8),
        "meta/voxel_mm": np.float32(geom.voxel_mm),
        "meta/gamma": np.float32(cfg.gamma),
        "meta/hypoxia": np.float32(cfg.hypoxia),
        "meta/Dw": np.float32(cfg.Dw),
        "meta/f": np.float32(cfg.f),
        "meta/dose_times": np.asarray(cfg.dose_times, dtype=np.float32),
    }

    for case in CASES:
        beam = None
        if case != "no_treatment":
            beam = P.beam_field(geom, P.BEAM_BY_NAME[case])
            store[f"beam/{case}"] = beam[:, :, ci[2]].astype(np.float32)
        res = simulate(geom.WM, geom.GM, geom.brain, geom.A0, beam, cfg,
                       device=device, centroid=geom.centroid_vox)
        # (n_snap, H, W) stacks -- float16 is plenty for a picture
        store[f"{case}/A"] = np.stack(
            [res.planes[s]["A_axial"] for s in snaps]).astype(np.float16)
        store[f"{case}/Z"] = np.stack(
            [res.planes[s]["Z_axial"] for s in snaps]).astype(np.float16)
        store[f"{case}/t"] = res.t.astype(np.float32)
        store[f"{case}/mass"] = res.normalized_mass.astype(np.float32)
        store[f"{case}/U"] = res.U_t.astype(np.float32)
        store[f"{case}/cover"] = res.beam_coverage.astype(np.float32)
        print(f"  {case:16s} final={res.info['final_mass_ratio']:.3f} "
              f"cover0={float(res.beam_coverage[0]):.3f}")

    np.savez_compressed(dst, **store)
    print(f"cached -> {dst}  ({dst.stat().st_size/1e6:.1f} MB)")
    return dst


def export_report_frames(dst_dir: Path = REPORT_FIGS) -> None:
    """Wytnij z animacji statyczne klatki, których używa report/main.tex.

    Rysunek 1 to moment napromieniania (klatka o najsilniejszej fioletowej
    poświacie), rysunek 3 to stan końcowy, rysunek 2 to mechanizm dawki.
    """
    from PIL import Image
    dst_dir.mkdir(parents=True, exist_ok=True)

    im = Image.open(OUT / "anim1_treatment.gif")
    best = (0, 0)
    for k in range(im.n_frames):
        im.seek(k)
        a = np.asarray(im.convert("RGB")).astype(int)
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        viol = int(((b > g + 18) & (r > g + 6)).sum())
        if viol > best[0]:
            best = (viol, k)
    for idx, name in [(best[1], "rys1-geometrie-wiazki.png"),
                      (im.n_frames - 1, "rys3-wynik-koncowy.png")]:
        im.seek(idx)
        im.convert("RGB").save(dst_dir / name)
        print(f"  -> {name}  (klatka {idx})")

    im2 = Image.open(OUT / "anim2_mechanism.gif")
    im2.seek(min(20, im2.n_frames - 1))
    im2.convert("RGB").save(dst_dir / "rys2-mechanizm-dawki.png")
    print("  -> rys2-mechanizm-dawki.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", default="data_001")
    ap.add_argument("--device", default="cuda:4")
    ap.add_argument("--dt-snap", type=float, default=0.5)
    ap.add_argument("--T", type=float, default=80.0)
    ap.add_argument("--grid", type=int, default=160)
    ap.add_argument("--sim-only", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    dst = CACHE / f"{args.pid}_dense.npz"
    if not dst.exists():
        print(f"== dense simulation: {args.pid} on {args.device}")
        dense_sim(args.pid, args.device, args.dt_snap, args.T, args.grid)
    else:
        print(f"== reusing cache {dst.name}")
    if args.sim_only:
        return 0
    from cancer_sim.viz import animate as A
    A.render_all(dst, OUT)
    export_report_frames()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
