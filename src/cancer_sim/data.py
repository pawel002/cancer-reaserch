"""Dataset loading and sparse noisy-observation generation.

The repository ships precomputed PDE tumour-mass trajectories as CSV files with
columns ``time, normalized_mass, U_t, W_eff_damage``.  Only the first three are
needed by the surrogates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config


@dataclass
class CaseData:
    """Full-grid PDE reference trajectory for one beam-tumour configuration."""

    t_full: np.ndarray
    y_full: np.ndarray
    U_full: np.ndarray
    W_full: np.ndarray


def _load_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"{path} must have columns: time, normalized_mass, U_t[, W_eff]")
    t = arr[:, 0].astype(float)
    y = arr[:, 1].astype(float)
    U = arr[:, 2].astype(float)
    W = arr[:, 3].astype(float) if arr.shape[1] >= 4 else U.copy()
    return t, y, U, W


def load_case(case: str, data_dir: Optional[Path] = None) -> CaseData:
    """Load the full-grid reference trajectory for ``case``."""
    data_dir = Path(data_dir or config.DATA_DIR)
    t, y, U, W = _load_csv(data_dir / f"{case}_full.csv")
    return CaseData(t_full=t, y_full=y, U_full=U, W_full=W)


# ----------------------------------------------------------------------------
# Sparse noisy observation ensembles
# ----------------------------------------------------------------------------

def select_base_points(
    case: str,
    n_points: int,
    t_start: float,
    t_end: float,
    data_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pick ``n_points`` evenly-spaced samples from the assimilation window.

    Prefers the (denser) ``*_train.csv`` if present, else falls back to ``*_full``.
    """
    data_dir = Path(data_dir or config.DATA_DIR)
    train_path = data_dir / f"{case}_train.csv"
    full_path = data_dir / f"{case}_full.csv"
    t, y, U, W = _load_csv(train_path if train_path.exists() else full_path)

    mask = (t >= t_start) & (t <= t_end)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise ValueError(f"No points for {case} in [{t_start}, {t_end}]")

    if n_points >= len(idx):
        chosen = idx
    else:
        local = np.linspace(0, len(idx) - 1, n_points).round().astype(int)
        local = np.array(sorted(set(local.tolist())), dtype=int)
        chosen = idx[local]
    return t[chosen], y[chosen], U[chosen], W[chosen]


def generate_noisy_ensemble(
    cases: Optional[List[str]] = None,
    out_dir: Optional[Path] = None,
    n_points: int = config.N_TRAIN_POINTS,
    n_ensemble: int = config.N_ENSEMBLE_RUNS,
    t_start: float = config.TRAIN_SAMPLE_START,
    t_end: float = config.TRAIN_SAMPLE_END,
    noise_std: float = config.NOISE_STD,
    noise_relative: bool = config.NOISE_RELATIVE,
    seed: int = config.NOISE_SEED,
    data_dir: Optional[Path] = None,
) -> Path:
    """Create per-case noisy sparse observation sets.

    Output layout (matches the original repository):
        <out_dir>/<case>/obs_base_clean.csv
        <out_dir>/<case>/obs_000.csv ... obs_{n-1}.csv
    Returns the output directory.
    """
    cases = cases or config.CASES
    out_dir = Path(out_dir or config.NOISY_OBS_DIR)
    rng = np.random.default_rng(seed)
    y_min = 1e-6

    for case in cases:
        case_dir = out_dir / case
        case_dir.mkdir(parents=True, exist_ok=True)

        t, y, U, W = select_base_points(case, n_points, t_start, t_end, data_dir)
        header = "time,normalized_mass,U_t,W_eff_damage"
        np.savetxt(case_dir / "obs_base_clean.csv",
                   np.column_stack([t, y, U, W]),
                   delimiter=",", header=header, comments="")

        for k in range(n_ensemble):
            sigma = noise_std * np.maximum(np.abs(y), y_min) if noise_relative \
                else noise_std * np.ones_like(y)
            y_noisy = np.clip(y + rng.normal(0.0, sigma), y_min, None)
            np.savetxt(case_dir / f"obs_{k:03d}.csv",
                       np.column_stack([t, y_noisy, U, W]),
                       delimiter=",", header=header, comments="")

    return out_dir


def load_observation_set(case: str, run_idx: int, obs_dir: Optional[Path] = None):
    """Load one noisy observation set: returns (t_fit, y_fit, U_fit)."""
    obs_dir = Path(obs_dir or config.NOISY_OBS_DIR)
    path = obs_dir / case / f"obs_{run_idx:03d}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing observation set: {path}. Run generate_noise first.")
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    return arr[:, 0].astype(float), arr[:, 1].astype(float), arr[:, 2].astype(float)


def load_clean_fit_points(case: str, obs_dir: Optional[Path] = None):
    obs_dir = Path(obs_dir or config.NOISY_OBS_DIR)
    path = obs_dir / case / "obs_base_clean.csv"
    if not path.exists():
        return None
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    return arr[:, 0].astype(float), arr[:, 1].astype(float), arr[:, 2].astype(float)


def load_ensemble_matrix(case: str, n_ensemble: int, obs_dir: Optional[Path] = None):
    """Stack all noisy realisations into shared-grid arrays.

    Returns (t_fit (T,), Y_fit (T, E), U_fit (T,)) exploiting the fact that every
    realisation shares the same time grid and dose input -- only ``y`` is noisy.
    """
    obs_dir = Path(obs_dir or config.NOISY_OBS_DIR)
    t_ref = U_ref = None
    cols = []
    members = []
    for run_idx in range(n_ensemble):
        path = obs_dir / case / f"obs_{run_idx:03d}.csv"
        if not path.exists():
            continue
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
        t, y, U = arr[:, 0], arr[:, 1], arr[:, 2]
        if t_ref is None:
            t_ref, U_ref = t.astype(float), U.astype(float)
        cols.append(y.astype(float))
        members.append(run_idx)
    if not cols:
        raise FileNotFoundError(f"No observation sets found for case {case}")
    return t_ref, np.column_stack(cols), U_ref, members


def make_time_weights(t_fit: np.ndarray, last_to_first: float,
                      normalize_mean: bool = True) -> np.ndarray:
    """Exponential recency weights (discounted least squares).

    The last assimilation sample carries ``last_to_first`` times the weight of
    the first one.  Returns ones if ``last_to_first <= 1``.
    """
    t = np.asarray(t_fit, dtype=float)
    if last_to_first <= 1.0 or len(t) <= 1:
        return np.ones_like(t)
    s = (t - t.min()) / (t.max() - t.min() + 1e-12)
    w = np.exp(math.log(last_to_first) * s)
    if normalize_mean:
        w = w / (w.mean() + 1e-12)
    return w
