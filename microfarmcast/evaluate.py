"""Metrics and chronological partitions."""
import numpy as np
import pandas as pd


def chronological_masks(frame, timestamp_col, validation_fraction, test_fraction, horizon_hours=1):
    times = np.array(sorted(frame[timestamp_col].dropna().unique()))
    if len(times) < 10:
        raise ValueError("Need at least 10 distinct timestamps for chronological train/validation/test splits")
    n_test = max(1, int(round(len(times) * test_fraction)))
    n_val = max(1, int(round(len(times) * validation_fraction)))
    if n_test + n_val >= len(times) - 1:
        raise ValueError("Too few timestamps for configured split fractions")
    val_start = times[-(n_test + n_val)]
    test_start = times[-n_test]
    # Partition by label valid-time, not forecast origin. This purges the horizon
    # across boundaries so a training label cannot land in validation/test time.
    valid_time = frame[timestamp_col] + pd.to_timedelta(int(horizon_hours), unit="h")
    return (valid_time < val_start), ((valid_time >= val_start) & (valid_time < test_start)), (valid_time >= test_start)


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if not len(y_true):
        return {"n": 0, "mae": None, "rmse": None}
    return {"n": int(len(y_true)), "mae": float(np.mean(np.abs(y_true-y_pred))), "rmse": float(np.sqrt(np.mean((y_true-y_pred)**2)))}


def classification_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp, fp = int((y_true & y_pred).sum()), int((~y_true & y_pred).sum())
    fn, tn = int((y_true & ~y_pred).sum()), int((~y_true & ~y_pred).sum())
    precision = tp / (tp+fp) if tp+fp else 0.0
    recall = tp / (tp+fn) if tp+fn else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {"n": int(len(y_true)), "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}
