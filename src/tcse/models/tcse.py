"""Proposed TCSE model (report Section IV-F, Eqs. 7-11).

Level 0 : out-of-fold probabilities of the four tuned baselines plus an
          Isolation Forest anomaly score fitted on healthy drives only.
Level 1 : class-weighted, L2-regularised logistic regression on
          z = [p_DT, p_KNN, p_NB, p_SVM, s(x, n), n_delta_pos].
Decision: threshold theta* minimising validation cost (tcse.evaluate).

Sign convention for the anomaly score: sklearn's ``score_samples`` returns
-s(x, n) of Eq. 7, so we use ``-score_samples``: s in (0, 1], higher = more
anomalous, s close to 1 = clear anomaly, s much below 0.5 = normal.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

META_FEATURES = ["p_dt", "p_knn", "p_nb", "p_svm", "s_iforest", "n_delta_pos"]


def make_isolation_forest(cfg: dict, seed: int) -> IsolationForest:
    c = cfg["isolation_forest"]
    return IsolationForest(n_estimators=c["n_estimators"], max_samples=c["max_samples"],
                           contamination=c["contamination"], random_state=seed, n_jobs=1)


def anomaly_score(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """s(x, n) = 2^(-E[h(x)]/c(n)) (Eq. 7); higher = more anomalous."""
    return -model.score_samples(X)


def meta_columns(use_iforest: bool = True, use_ndelta: bool = True) -> list[str]:
    cols = ["p_dt", "p_knn", "p_nb", "p_svm"]
    if use_iforest:
        cols.append("s_iforest")
    if use_ndelta:
        cols.append("n_delta_pos")
    return cols


def make_meta(C: float, cfg: dict) -> Pipeline:
    """Eqs. 8-9: logistic regression, L2 penalty (C = 1/alpha). Class weights
    (inverse class frequency) are passed as sample weights by fit_meta.
    z is standardised first so the coefficients are comparable across meta-features."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=cfg["meta"]["max_iter"])),
    ])


def balanced_weights(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Inverse class-frequency weights computed on the (IPW-)weighted counts,
    i.e. on the natural training distribution: sum of weights per class = N/2."""
    w = np.asarray(w, dtype=np.float64)
    tot = w.sum()
    out = w.copy()
    for c in (0, 1):
        m = y == c
        out[m] *= tot / (2.0 * w[m].sum())
    return out


def fit_meta(Z: np.ndarray, y: np.ndarray, groups: np.ndarray, cfg: dict, seed: int,
             sample_weight: np.ndarray | None = None):
    """Tune C by stratified group CV (groups = drive) on the OOF meta-features, then refit.

    sample_weight are inverse-probability weights of the OOF rows (1 when all
    rows are used). Class weights are inverse class frequencies of the weighted
    data (Eq. 9, omega_1 and omega_0)."""
    from tcse.split import group_folds
    scores = {}
    folds = group_folds(groups, y, cfg["split"]["cv_folds"], seed)
    w = np.ones(len(y)) if sample_weight is None else sample_weight
    for C in cfg["meta"]["C_grid"]:
        s = []
        for tr, te in folds:
            if y[te].sum() == 0:
                continue
            m = make_meta(C, cfg).fit(Z[tr], y[tr], lr__sample_weight=balanced_weights(y[tr], w[tr]))
            s.append(average_precision_score(y[te], m.predict_proba(Z[te])[:, 1], sample_weight=w[te]))
        scores[C] = float(np.mean(s))
    best = max(scores, key=lambda c: (scores[c], -c))  # ties -> stronger regularisation
    model = make_meta(best, cfg).fit(Z, y, lr__sample_weight=balanced_weights(y, w))
    return model, {"C": best, "alpha": 1.0 / best, "cv_scores": scores}


def meta_coefficients(model: Pipeline, columns: list[str]) -> dict:
    lr = model.named_steps["lr"]
    return {"intercept": float(lr.intercept_[0]),
            **{c: float(w) for c, w in zip(columns, lr.coef_[0])}}
