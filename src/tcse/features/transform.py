"""Train-only preprocessing (report Section IV-C steps 2, 3, 5, 6).

``TCSEPreprocessor`` is a scikit-learn transformer. ``fit`` sees the training
split only and learns, in order:
  1. column drops: attributes > max_missing_frac empty (raw, before forward
     fill) are dropped with all their derived features; zero-variance columns
     are dropped;
  2. which binary missing indicators to keep;
  3. training medians for imputation;
  4. the log transform (Eq. 4): ln(1+x) for non-negative columns, the
     sign-preserving sign(x) ln(1+|x|) for columns that take negative values
     (deltas and slopes), then z-score parameters (Eq. 5);
  5. feature selection: rank by mutual information with the target and drop
     one feature from each pair with |r| > corr_threshold (the lower-MI one).
``transform`` applies the frozen parameters unchanged to any rows.

``fit`` accepts a pandas DataFrame or any object exposing ``columns`` and
``column(name) -> ndarray`` (the memory-mapped store restricted to training
rows), so the full matrix never needs to be in memory.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif

from tcse.features.clean import select_missing_indicators

_ATTR_RE = re.compile(r"^smart_(\d+)_")


def attr_of(col: str) -> int | None:
    if col.startswith("miss_"):
        return int(col[5:])
    m = _ATTR_RE.match(col)
    return int(m.group(1)) if m else None


class FrameSource:
    """Column access for a DataFrame (used in tests and small data)."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.columns = list(df.columns)

    def __len__(self):
        return len(self.df)

    def column(self, name):
        return self.df[name].to_numpy(dtype=np.float32)

    def block(self, names, rows):
        return self.df[names].to_numpy(dtype=np.float32)[rows]


class StoreSource:
    """Column access to a FeatureStore restricted to given rows (None = all rows).

    ``extra`` maps additional column names (e.g. model one-hot) to full-length arrays.
    """

    def __init__(self, store, rows: np.ndarray | None = None, extra: dict | None = None):
        self.store, self.rows = store, rows
        self.extra = extra or {}
        self.columns = list(store.columns) + list(self.extra)

    def __len__(self):
        return self.store.X.shape[0] if self.rows is None else len(self.rows)

    def column(self, name):
        if name in self.extra:
            v = self.extra[name]
            return np.asarray(v if self.rows is None else v[self.rows], dtype=np.float32)
        return self.store.column(name, self.rows)

    def block(self, names, rows):
        """rows are positions within self.rows (absolute rows when self.rows is None)."""
        abs_rows = rows if self.rows is None else self.rows[rows]
        out = np.empty((len(abs_rows), len(names)), dtype=np.float32)
        for j, c in enumerate(names):
            out[:, j] = (self.extra[c][abs_rows] if c in self.extra
                         else self.store.X[abs_rows, self.store.col_idx[c]])
        return out


def _as_source(X):
    if isinstance(X, pd.DataFrame):
        return FrameSource(X)
    return X


def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


class TCSEPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, candidates=None, flag_columns=None, passthrough_log=None,
                 max_missing_frac=0.9, indicator_min=0.001, indicator_max=0.9,
                 corr_threshold=0.95, mi_neighbors=3, mi_neg_per_pos=20, mi_max_rows=60000,
                 min_mi=None, max_features=None, corr_sample_rows=500000, random_state=0):
        self.candidates = candidates
        self.flag_columns = flag_columns
        self.passthrough_log = passthrough_log
        self.max_missing_frac = max_missing_frac
        self.indicator_min = indicator_min
        self.indicator_max = indicator_max
        self.corr_threshold = corr_threshold
        self.mi_neighbors = mi_neighbors
        self.mi_neg_per_pos = mi_neg_per_pos
        self.mi_max_rows = mi_max_rows
        self.min_mi = min_mi
        self.max_features = max_features
        self.corr_sample_rows = corr_sample_rows
        self.random_state = random_state

    # ------------------------------------------------------------------ fit
    def fit(self, X, y):
        src = _as_source(X)
        y = np.asarray(y).astype(np.int8)
        rng = np.random.default_rng(self.random_state)
        cands = list(self.candidates if self.candidates is not None else src.columns)
        flags = set(self.flag_columns or [])          # binary columns: no log transform
        nolog = set(self.passthrough_log or []) | flags

        # 1. attribute-level missingness on raw (pre-ffill) data via miss_<id> flags
        attrs = sorted({a for c in cands if (a := attr_of(c)) is not None})
        miss_frac = {}
        for a in attrs:
            name = f"miss_{a}"
            if name in src.columns:
                miss_frac[a] = float(np.mean(src.column(name)))
        dropped_attrs = {a for a, f in miss_frac.items() if f > self.max_missing_frac}
        keep_ind = set(select_missing_indicators(
            {a: f for a, f in miss_frac.items() if a not in dropped_attrs},
            self.indicator_min, self.indicator_max))
        self.missing_frac_ = miss_frac
        self.dropped_attributes_ = sorted(dropped_attrs)
        self.indicators_ = sorted(keep_ind)

        kept, medians, logkind, mu, sd = [], {}, {}, {}, {}
        self.drop_reasons_ = {}
        for c in cands:
            a = attr_of(c)
            if a in dropped_attrs:
                self.drop_reasons_[c] = "attribute_missing"
                continue
            if c.startswith("miss_") and a not in keep_ind:
                self.drop_reasons_[c] = "indicator_not_informative"
                continue
            v = src.column(c).astype(np.float64)
            nan = np.isnan(v)
            if nan.mean() > self.max_missing_frac:
                self.drop_reasons_[c] = "missing"
                continue
            med = float(np.median(v[~nan])) if (~nan).any() else 0.0
            v[nan] = med
            if np.nanvar(v) <= 0:
                self.drop_reasons_[c] = "zero_variance"
                continue
            if c in nolog:
                kind = "none"
            elif v.min() >= 0:
                kind = "log1p"
                v = np.log1p(v)
            else:
                kind = "signed_log1p"
                v = signed_log1p(v)
            s = float(v.std())
            if s <= 0:
                self.drop_reasons_[c] = "zero_variance"
                continue
            kept.append(c)
            medians[c], logkind[c], mu[c], sd[c] = med, kind, float(v.mean()), s
        self.input_columns_ = kept
        self.medians_ = np.array([medians[c] for c in kept], dtype=np.float32)
        self.logkind_ = [logkind[c] for c in kept]
        self.mean_ = np.array([mu[c] for c in kept], dtype=np.float32)
        self.scale_ = np.array([sd[c] for c in kept], dtype=np.float32)

        # 5. feature selection on the transformed training data
        n = len(src)
        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)
        n_pos = min(len(pos), self.mi_max_rows // (1 + self.mi_neg_per_pos))
        pos_s = rng.choice(pos, n_pos, replace=False) if n_pos < len(pos) else pos
        neg_s = rng.choice(neg, min(len(neg), n_pos * self.mi_neg_per_pos), replace=False)
        mi_rows = np.sort(np.concatenate([pos_s, neg_s]))
        Z = self._transform_full(src.block(kept, mi_rows))
        disc = np.array([c in flags for c in kept])
        mi = mutual_info_classif(Z, y[mi_rows], discrete_features=disc,
                                 n_neighbors=self.mi_neighbors, random_state=self.random_state)
        corr_rows = np.sort(rng.choice(n, min(n, self.corr_sample_rows), replace=False))
        Zc = self._transform_full(src.block(kept, corr_rows)).astype(np.float64)
        R = np.corrcoef(Zc, rowvar=False) if Zc.shape[1] > 1 else np.ones((1, 1))
        R = np.nan_to_num(R)
        order = np.argsort(-mi, kind="stable")
        sel = []
        for j in order:
            if self.min_mi is not None and mi[j] <= self.min_mi:
                self.drop_reasons_[kept[j]] = "low_mi"
                continue
            if any(abs(R[j, k]) > self.corr_threshold for k in sel):
                partner = kept[max(sel, key=lambda k: abs(R[j, k]))]
                self.drop_reasons_[kept[j]] = f"corr>{self.corr_threshold} with {partner}"
                continue
            sel.append(j)
            if self.max_features and len(sel) >= self.max_features:
                break
        self.mutual_info_ = dict(zip(kept, map(float, mi)))
        self.selected_idx_ = np.array(sel, dtype=np.int64)
        self.selected_ = [kept[j] for j in sel]
        return self

    # ------------------------------------------------------------ transform
    def _transform_full(self, B: np.ndarray) -> np.ndarray:
        B = np.array(B, dtype=np.float32, copy=True)
        nan = np.isnan(B)
        if nan.any():
            r, c = np.nonzero(nan)
            B[r, c] = self.medians_[c]
        for j, kind in enumerate(self.logkind_):
            if kind == "log1p":
                np.log1p(np.maximum(B[:, j], 0), out=B[:, j])
            elif kind == "signed_log1p":
                B[:, j] = signed_log1p(B[:, j])
        B -= self.mean_
        B /= self.scale_
        return B

    def transform(self, X):
        """X: array whose columns are ``input_columns_`` (or a DataFrame)."""
        if isinstance(X, pd.DataFrame):
            X = X[self.input_columns_].to_numpy(dtype=np.float32)
        return self._transform_full(X)[:, self.selected_idx_]

    def transform_selected(self, B: np.ndarray) -> np.ndarray:
        """Like transform, but B already holds only the ``selected_`` columns
        (cheaper: unselected columns are never read from disk)."""
        idx = self.selected_idx_
        B = np.array(B, dtype=np.float32, copy=True)
        nan = np.isnan(B)
        if nan.any():
            r, c = np.nonzero(nan)
            B[r, c] = self.medians_[idx][c]
        for jj, j in enumerate(idx):
            kind = self.logkind_[j]
            if kind == "log1p":
                np.log1p(np.maximum(B[:, jj], 0), out=B[:, jj])
            elif kind == "signed_log1p":
                B[:, jj] = signed_log1p(B[:, jj])
        B -= self.mean_[idx]
        B /= self.scale_[idx]
        return B

    def refit_scaling(self, X) -> "TCSEPreprocessor":
        """Cross-dataset: re-estimate medians and log/z-score parameters of the
        selected columns on the *target* dataset's unlabelled features. Column
        drops, transform kinds and the selected feature list stay frozen."""
        src = _as_source(X)
        for j in self.selected_idx_:
            c = self.input_columns_[j]
            v = src.column(c).astype(np.float64)
            nan = np.isnan(v)
            med = float(np.median(v[~nan])) if (~nan).any() else 0.0
            v[nan] = med
            kind = self.logkind_[j]
            if kind == "log1p":
                v = np.log1p(np.maximum(v, 0))
            elif kind == "signed_log1p":
                v = signed_log1p(v)
            sd = float(v.std())
            self.medians_[j], self.mean_[j] = med, float(v.mean())
            self.scale_[j] = sd if sd > 0 else 1.0
        return self

    def get_feature_names_out(self, input_features=None):
        return np.array(self.selected_)
