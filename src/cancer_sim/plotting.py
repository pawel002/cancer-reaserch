"""Plotting: ensemble uncertainty bands and ablation comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config


def _bands(Y: np.ndarray, low=5, high=95):
    return (np.nanpercentile(Y, low, axis=0),
            np.nanpercentile(Y, 50, axis=0),
            np.nanpercentile(Y, high, axis=0))


def plot_ensemble_bands(case, t, y_true, per_method: Dict[str, np.ndarray],
                        out_path: Path, clean_obs=None):
    """One panel: PDE truth + median/5-95% band for each method.

    ``per_method[name]`` has shape (E, T).
    """
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.plot(t, y_true, lw=3, color="k", label="PDE ground truth")
    for name, Y in per_method.items():
        q05, q50, q95 = _bands(Y)
        ax.fill_between(t, q05, q95, alpha=0.18)
        ax.plot(t, q50, lw=2.1, label=f"{name} median")
    if clean_obs is not None:
        ax.scatter(clean_obs[0], clean_obs[1], s=40, facecolors="none",
                   edgecolors="black", linewidths=0.8, zorder=6, label="clean fit points")
    ax.axvline(config.TRAIN_SAMPLE_START, ls="--", lw=1.0)
    ax.axvline(config.TRAIN_SAMPLE_END, ls="--", lw=1.0)
    ax.set_title(f"Surrogate ensemble bands: {case}")
    ax.set_xlabel("time"); ax.set_ylabel("normalized tumor mass")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_all_cases_grid(cases, results: Dict, out_path: Path):
    """Grid of ensemble bands, one row per case.

    ``results[case]`` = (t, y_true, {method: Y(E,T)}, clean_obs).
    """
    fig, axes = plt.subplots(len(cases), 1, figsize=(10.5, 4.0 * len(cases)),
                             constrained_layout=True)
    if len(cases) == 1:
        axes = [axes]
    for ax, case in zip(axes, cases):
        t, y_true, per_method, clean = results[case]
        ax.plot(t, y_true, lw=2.6, color="k", label="PDE ground truth")
        for name, Y in per_method.items():
            q05, q50, q95 = _bands(Y)
            ax.fill_between(t, q05, q95, alpha=0.12)
            ax.plot(t, q50, lw=1.8, label=name)
        if clean is not None:
            ax.scatter(clean[0], clean[1], s=24, edgecolors="black",
                       linewidths=0.5, zorder=6)
        ax.axvline(config.TRAIN_SAMPLE_START, ls="--", lw=1.0)
        ax.axvline(config.TRAIN_SAMPLE_END, ls="--", lw=1.0)
        ax.set_title(case); ax.set_xlabel("time"); ax.set_ylabel("normalized mass")
        ax.grid(alpha=0.25); ax.legend(fontsize=7)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_bands(case, t, y_true, per_ablation: Dict[str, np.ndarray],
                        out_path: Path, clean_obs=None):
    """Compare PI-NODE bands for several (omega, s_r) settings on one case."""
    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    ax.plot(t, y_true, lw=3, color="k", label="PDE ground truth")
    for label, Y in per_ablation.items():
        q05, q50, q95 = _bands(Y)
        ax.fill_between(t, q05, q95, alpha=0.12)
        ax.plot(t, q50, lw=2, label=label)
    if clean_obs is not None:
        ax.scatter(clean_obs[0], clean_obs[1], s=40, edgecolors="black",
                   linewidths=0.6, zorder=6, label="clean fit points")
    ax.axvline(config.TRAIN_SAMPLE_START, ls="--", lw=1.0)
    ax.axvline(config.TRAIN_SAMPLE_END, ls="--", lw=1.0)
    ax.set_title(f"PI-NODE ablation bands: {case}")
    ax.set_xlabel("time"); ax.set_ylabel("normalized tumor mass")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_sweep(summary, out_path: Path):
    """Test relative RMSE vs number of assimilation points, one line per case.

    ``summary``: list of dicts with keys case, n_points, test_rel_rmse_percent.
    """
    cases = sorted({r["case"] for r in summary})
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for case in cases:
        rows = sorted([r for r in summary if r["case"] == case],
                      key=lambda r: r["n_points"])
        ns = [r["n_points"] for r in rows]
        vals = [r["test_rel_rmse_percent"] for r in rows]
        ax.plot(ns, vals, "o-", label=case)
    ax.set_xlabel("number of assimilation points $N_{fit}$")
    ax.set_ylabel("median test relative RMSE [%]")
    ax.set_title("PI-NODE accuracy vs. assimilation density")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
