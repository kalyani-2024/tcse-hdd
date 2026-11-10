"""Cleaning and missing-value handling (report Section IV-C steps 2-3).

Row-level rules (need no fitting, applied when the feature store is built):
  * physically impossible values -> NaN (negative counters, temperature outside 0-80 C)
  * forward-fill within each drive, never across drives

Train-fitted rules (sklearn transformers, fitted on the training split only):
  * ColumnCleaner : drop columns > max_missing_frac empty or with zero variance
  * MedianImputer : fill remaining NaN with training medians
  * missing-indicator selection: attributes whose training missing fraction is
    in [min_frac, max_frac] keep their binary miss_<id> column
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def clean_values(x: np.ndarray, attr_id: int, temp_attr: int, temp_range) -> np.ndarray:
    """Return a float64 copy with impossible values set to NaN."""
    x = np.asarray(x, dtype=np.float64).copy()
    with np.errstate(invalid="ignore"):
        x[x < 0] = np.nan
        if attr_id == temp_attr:
            lo, hi = temp_range
            x[(x < lo) | (x > hi)] = np.nan
    return x


def group_starts(group_ids: np.ndarray) -> np.ndarray:
    """For rows sorted by group, the index of the first row of each row's group."""
    g = np.asarray(group_ids)
    n = len(g)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    new = np.ones(n, dtype=bool)
    new[1:] = g[1:] != g[:-1]
    starts = np.where(new, np.arange(n), 0)
    return np.maximum.accumulate(starts)


def ffill_within_groups(x: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Forward-fill NaN within each contiguous group (rows sorted by group, date)."""
    n = len(x)
    valid = ~np.isnan(x)
    idx = np.where(valid, np.arange(n), -1)
    idx = np.maximum.accumulate(idx) if n else idx
    ok = idx >= starts            # last valid value belongs to the same drive
    out = np.full(n, np.nan)
    out[ok] = x[idx[ok]]
    return out


class ColumnCleaner(BaseEstimator, TransformerMixin):
    """Drop columns that are mostly empty or constant on the training data."""

    def __init__(self, max_missing_frac: float = 0.9):
        self.max_missing_frac = max_missing_frac

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        miss = np.isnan(X).mean(axis=0)
        with np.errstate(all="ignore"):
            var = np.nanvar(X, axis=0)
        var = np.nan_to_num(var, nan=0.0)
        self.missing_frac_ = miss
        self.keep_ = (miss <= self.max_missing_frac) & (var > 0)
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.keep_]


class MedianImputer(BaseEstimator, TransformerMixin):
    """NaN -> training-set median (column-wise)."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        with np.errstate(all="ignore"):
            med = np.nanmedian(X, axis=0)
        self.medians_ = np.nan_to_num(med, nan=0.0)
        return self

    def transform(self, X):
        X = np.array(X, dtype=np.float32, copy=True)
        r, c = np.where(np.isnan(X))
        X[r, c] = self.medians_[c]
        return X


def select_missing_indicators(missing_frac: dict, min_frac: float, max_frac: float) -> list:
    """Attributes whose missingness is frequent enough to be informative but not dropped."""
    return sorted(a for a, f in missing_frac.items() if min_frac <= f <= max_frac)
