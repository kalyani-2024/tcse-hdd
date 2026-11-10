"""Class-imbalance handling (report Section IV-C step 8, Eq. 6).

Training data only, inside every CV fold:
  1. RandomUnderSampler: healthy drive-days down to ``undersample_ratio`` : 1
  2. SMOTE: minority up to ``smote_ratio`` (failure : healthy)
  3. cap (KNN / SVM only): stratified random cap on the resampled set
Validation and test rows are never resampled.

For memory, step 1 is applied to row *indices* (so only the kept rows are
ever read from disk); the identical RandomUnderSampler is also the first
step of the imblearn Pipeline, where it is then a no-op. Both use the same
imblearn implementation and random state.
"""
from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn import FunctionSampler
from imblearn.under_sampling import RandomUnderSampler


def undersample_rows(rows: np.ndarray, y: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    """Positions (into ``rows``) kept by random undersampling at healthy:failure = ratio:1.

    y are the labels of ``rows``. Uses imblearn's RandomUnderSampler on an index column.
    """
    y = np.asarray(y)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0:
        raise ValueError("no positive rows in the training data")
    if n_neg <= ratio * n_pos:
        return np.arange(len(rows))
    rus = RandomUnderSampler(sampling_strategy=1.0 / ratio, random_state=seed)
    idx, _ = rus.fit_resample(np.arange(len(rows)).reshape(-1, 1), y)
    return np.sort(idx.ravel())


def _undersample(X, y, ratio: float, seed: int):
    keep = undersample_rows(np.arange(len(y)), y, ratio, seed)
    return X[keep], np.asarray(y)[keep]


def _cap(X, y, cap: int, seed: int):
    if cap is None or len(y) <= cap:
        return X, y
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        k = int(round(cap * len(idx) / len(y)))
        keep.append(rng.choice(idx, k, replace=False))
    keep = np.sort(np.concatenate(keep))
    return X[keep], y[keep]


def make_resampling_pipeline(clf, cfg: dict, seed: int, capped: bool) -> Pipeline:
    s = cfg["sampling"]
    steps = [
        ("undersample", FunctionSampler(func=_undersample, validate=False,
                                        kw_args={"ratio": s["undersample_ratio"], "seed": seed})),
        ("smote", SMOTE(sampling_strategy=s["smote_ratio"], k_neighbors=s["smote_k"], random_state=seed)),
    ]
    if capped:
        steps.append(("cap", FunctionSampler(func=_cap, validate=False,
                                             kw_args={"cap": s["cap_knn_svm"], "seed": seed})))
    steps.append(("clf", clf))
    return Pipeline(steps)


def resampled_counts(pipe: Pipeline, X, y) -> dict:
    """Class counts before / after each sampler (for logging)."""
    out = {"before": np.bincount(np.asarray(y, dtype=int), minlength=2).tolist()}
    Xc, yc = X, y
    for name, step in pipe.steps[:-1]:
        Xc, yc = step.fit_resample(Xc, yc)
        out[name] = np.bincount(np.asarray(yc, dtype=int), minlength=2).tolist()
    return out
