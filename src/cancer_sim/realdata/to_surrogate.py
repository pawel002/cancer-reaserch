"""Write an :class:`fk3d.FK3DResult` into the surrogate's dataset layout.

Emits exactly the files the existing experiment runners expect, so a real
patient becomes a drop-in ``--cases <name>``:

    <data_dir>/<name>_full.csv          dense [0, T] trajectory
    <data_dir>/<name>_train.csv         assimilation-window subset
    <data_dir>/<name>_test.csv          forecast-window subset
    <obs_dir>/<name>/obs_*.csv          noisy sparse observation ensemble

Schema (matches ``datasets/*_full.csv``): ``time, normalized_mass, U_t,
W_eff_damage``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .. import config, data
from .fk3d import FK3DResult

_HEADER = "time,normalized_mass,U_t,W_eff_damage"


def _subset(arr: np.ndarray, t: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (t >= lo) & (t <= hi)
    return arr[mask]


def write_dataset(name: str, result: FK3DResult,
                  data_dir: Optional[Path] = None,
                  train_window: Tuple[float, float] = (10.0, 40.0),
                  test_window: Tuple[float, float] = (40.0, 80.0)) -> Path:
    """Write ``<name>_full/_train/_test.csv`` into ``data_dir``.

    Returns the path to the full-grid CSV.
    """
    data_dir = Path(data_dir or config.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    full = np.column_stack([result.t, result.normalized_mass,
                            result.U_t, result.W_eff])
    full_path = data_dir / f"{name}_full.csv"
    np.savetxt(full_path, full, delimiter=",", header=_HEADER, comments="")
    np.savetxt(data_dir / f"{name}_train.csv",
               _subset(full, result.t, *train_window),
               delimiter=",", header=_HEADER, comments="")
    np.savetxt(data_dir / f"{name}_test.csv",
               _subset(full, result.t, *test_window),
               delimiter=",", header=_HEADER, comments="")
    return full_path


def build_noisy_ensemble(name: str,
                         data_dir: Optional[Path] = None,
                         obs_dir: Optional[Path] = None,
                         n_points: int = config.N_TRAIN_POINTS,
                         n_ensemble: int = config.N_ENSEMBLE_RUNS,
                         seed: int = config.NOISE_SEED) -> Path:
    """Generate ``<obs_dir>/<name>/obs_*.csv`` via the existing data module."""
    return data.generate_noisy_ensemble(
        cases=[name],
        out_dir=obs_dir,
        n_points=n_points,
        n_ensemble=n_ensemble,
        data_dir=data_dir,
        seed=seed,
    )


def write_manifest(name: str, result: FK3DResult, extra: Optional[dict] = None,
                   data_dir: Optional[Path] = None) -> Path:
    """Record the FK3D parameters/diagnostics used to build this patient case."""
    data_dir = Path(data_dir or config.DATA_DIR)
    payload = {"case": name, "fk3d_info": result.info}
    if extra:
        payload.update(extra)
    path = data_dir / f"{name}_realdata_manifest.json"
    path.write_text(json.dumps(payload, indent=2))
    return path
