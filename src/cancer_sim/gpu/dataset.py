"""Assemble the benchmark tensors from the generated ground-truth cohort.

One :class:`Bench` holds every (patient, beam-configuration) trajectory plus the
noisy sparse observations every surrogate is assimilated against.  All of them
share one observation-time grid and one dose signal, so the whole benchmark is a
single dense array and a single batched fit.

Assimilation protocol (paper Table "Assimilation settings", with one fix):
    window      [14, 35]   -- the paper's [15, 35] starts *on* the first dose
                              pulse, which would hand the surrogate a non-zero
                              initial damage state it cannot know about; starting
                              one unit earlier brackets the whole first fraction.
    N_fit       20 evenly spaced samples
    noise       eps_i ~ N(0, (0.02 y_i)^2)
    forecast    (35, 80]    -- contains the *second*, unseen irradiation event
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

TRAIN_START = 14.0
TRAIN_END = 35.0
N_FIT = 20
NOISE_STD = 0.02
N_MEMBERS = 20

CASES = ("full_cover", "narrow_centered", "slight_shift", "strong_shift")


@dataclass
class Bench:
    pids: List[str]
    cases: List[str]
    labels: List[Tuple[str, str]]      # (pid, case) per column, length C
    t_obs: np.ndarray                  # (N_fit,)
    U_obs: np.ndarray                  # (N_fit,)
    Y_obs: np.ndarray                  # (N_fit, C*M) noisy
    Y_obs_clean: np.ndarray            # (N_fit, C)
    t_pred: np.ndarray                 # (T,) forecast/eval grid from t_obs[0]
    U_pred: np.ndarray                 # (T,)
    y_true: np.ndarray                 # (T, C) ground-truth mass on t_pred
    y_true_visible: np.ndarray         # (T, C) MRI-visible volume (normalised)
    y_no_treat: np.ndarray             # (T, C) untreated counterfactual
    n_members: int
    manifests: Dict[str, dict]

    @property
    def C(self) -> int:
        return len(self.labels)

    @property
    def CM(self) -> int:
        return self.C * self.n_members

    def case_of(self) -> np.ndarray:
        """Column index (0..C-1) of every one of the ``C*M`` fitted models."""
        return np.repeat(np.arange(self.C), self.n_members)

    def index(self, pid: str, case: str) -> int:
        return self.labels.index((pid, case))

    def context(self) -> np.ndarray:
        """Mass-weighted dose coverage at t = 0, one scalar per column.

        Known from the treatment plan without any extra measurement. It is the
        regime signal the reduced state (y, z, t, U) cannot carry: the whole
        point of the mismatched configurations is that they deliver a similar
        total dose through a different spatial pattern.
        """
        return np.array([self.manifests[p]["cases"][c]["beam_coverage0"]
                         for p, c in self.labels], dtype=float)


def _member_noise(pid: str, case: str, member: int, y: np.ndarray) -> np.ndarray:
    seed = int.from_bytes(
        hashlib.sha256(f"{pid}|{case}|{member}".encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    sigma = NOISE_STD * np.maximum(np.abs(y), 1e-6)
    return np.clip(y + rng.normal(0.0, sigma), 1e-6, None)


def build(cohort_dir: Path, pids: Optional[Sequence[str]] = None,
          cases: Sequence[str] = CASES, n_members: int = N_MEMBERS,
          n_fit: int = N_FIT, train_start: float = TRAIN_START,
          train_end: float = TRAIN_END) -> Bench:
    cohort_dir = Path(cohort_dir)
    if pids is None:
        pids = sorted(p.name.replace("_curves.npz", "")
                      for p in cohort_dir.glob("*_curves.npz"))
    pids = list(pids)

    manifests = {p: json.loads((cohort_dir / f"{p}_manifest.json").read_text())
                 for p in pids}
    t_obs = np.linspace(train_start, train_end, n_fit)

    labels, cols_true, cols_vis, cols_nt, cols_clean, cols_noisy = [], [], [], [], [], []
    t_pred = U_pred = None
    for pid in pids:
        z = np.load(cohort_dir / f"{pid}_curves.npz")
        t = z[f"{cases[0]}/t"].astype(float)
        mask = t >= train_start - 1e-9
        if t_pred is None:
            t_pred = t[mask]
            U_pred = z[f"{cases[0]}/U"].astype(float)[mask]
        nt = np.interp(t_pred, t, z["no_treatment/mass"].astype(float))
        for case in cases:
            y = z[f"{case}/mass"].astype(float)
            vis = z[f"{case}/mass_visible"].astype(float)
            labels.append((pid, case))
            cols_true.append(y[mask])
            cols_vis.append(vis[mask])
            cols_nt.append(nt)
            y_clean = np.interp(t_obs, t, y)
            cols_clean.append(y_clean)
            for m in range(n_members):
                cols_noisy.append(_member_noise(pid, case, m, y_clean))

    return Bench(
        pids=pids, cases=list(cases), labels=labels,
        t_obs=t_obs, U_obs=np.interp(t_obs, t_pred, U_pred),
        Y_obs=np.column_stack(cols_noisy), Y_obs_clean=np.column_stack(cols_clean),
        t_pred=t_pred, U_pred=U_pred,
        y_true=np.column_stack(cols_true),
        y_true_visible=np.column_stack(cols_vis),
        y_no_treat=np.column_stack(cols_nt),
        n_members=n_members, manifests=manifests)
