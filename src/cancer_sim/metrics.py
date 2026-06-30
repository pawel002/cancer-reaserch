"""Error metrics relative to the PDE reference trajectory (paper Table results)."""

from __future__ import annotations

import numpy as np

from . import config


def suffix_from_first_fit(t_full, U_full, y_full, t_fit):
    """Forecast view starting at the first observed time (not t=0)."""
    t0 = float(t_fit[0])
    mask = t_full >= t0
    return t_full[mask], U_full[mask], y_full[mask], mask


def expand_suffix(t_full, mask, pred_suffix):
    """Place a suffix prediction back on the full grid (NaN before first fit)."""
    out = np.full_like(t_full, np.nan, dtype=float)
    out[mask] = pred_suffix
    return out


def mse_on_points(t_source, y_pred_source, t_points, y_points):
    """MSE of an interpolated prediction at scattered observation points."""
    finite = np.isfinite(y_pred_source)
    if np.sum(finite) < 2:
        return float("nan")
    lo, hi = t_source[finite].min(), t_source[finite].max()
    mask = (t_points >= lo) & (t_points <= hi)
    if not np.any(mask):
        return float("nan")
    pred = np.interp(t_points[mask], t_source[finite], y_pred_source[finite])
    return float(np.mean((pred - y_points[mask]) ** 2))


def trajectory_errors(y_true, y_pred, t,
                      train_start=config.TRAIN_SAMPLE_START,
                      train_end=config.TRAIN_SAMPLE_END):
    """Train/test relative RMSE (%) and final-time error (%)."""
    finite = np.isfinite(y_pred)
    tr = (t >= train_start) & (t <= train_end) & finite
    te = ((t < train_start) | (t > train_end)) & finite

    def rel(mask):
        if not np.any(mask):
            return float("nan")
        return 100.0 * np.sqrt(np.mean((y_pred[mask] - y_true[mask]) ** 2)
                               / (np.mean(y_true[mask] ** 2) + 1e-12))

    final_error = float("nan")
    idx = np.where(finite)[0]
    if len(idx):
        i = idx[-1]
        final_error = float(100.0 * (y_pred[i] - y_true[i]) / (abs(y_true[i]) + 1e-12))

    return {
        "train_rel_rmse_percent": rel(tr),
        "test_rel_rmse_percent": rel(te),
        "final_error_percent": final_error,
    }
