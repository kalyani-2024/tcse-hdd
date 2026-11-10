"""Baseline classifiers (report Section IV-E).

DT (CART, depth-limited, class-weighted), KNN (distance-weighted), Gaussian NB,
RBF SVM (class-weighted, Platt-scaled probabilities). Each is wrapped in the
imblearn resampling pipeline of tcse.sampling, so resampling happens inside
whatever data the pipeline is fitted on (a CV fold or the training split).
"""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.kernel_approximation import Nystroem
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from tcse.sampling import make_resampling_pipeline

BASELINES = ("dt", "knn", "nb", "svm")
CAPPED = {"knn", "svm"}
LABELS = {"dt": "DT", "knn": "KNN", "nb": "NB", "svm": "SVM", "tcse": "TCSE"}


class NystroemSVM(BaseEstimator, ClassifierMixin):
    """RBF-kernel SVM via the Nystroem low-rank kernel approximation
    (Williams & Seeger, 2001): an explicit RBF feature map followed by a
    class-weighted linear SVM, with Platt (sigmoid) probability calibration.
    Prediction costs O(n_components) per row instead of O(n_support_vectors)."""

    def __init__(self, C=1.0, gamma="scale", n_components=500, class_weight="balanced",
                 platt_cv=3, max_iter=5000, random_state=None):
        self.C = C
        self.gamma = gamma
        self.n_components = n_components
        self.class_weight = class_weight
        self.platt_cv = platt_cv
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        g = self.gamma
        if g == "scale":                         # same rule as sklearn SVC
            v = X.var()
            g = 1.0 / (X.shape[1] * v) if v > 0 else 1.0
        self.gamma_ = float(g)
        svm = make_pipeline(
            Nystroem(kernel="rbf", gamma=self.gamma_, n_components=min(self.n_components, len(X)),
                     random_state=self.random_state),
            LinearSVC(C=self.C, class_weight=self.class_weight, max_iter=self.max_iter,
                      random_state=self.random_state))
        self.model_ = CalibratedClassifierCV(svm, method="sigmoid", cv=self.platt_cv, ensemble=False)
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=np.float32))

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


def make_classifier(name: str, params: dict, seed: int):
    if name == "dt":
        return DecisionTreeClassifier(random_state=seed, **params)
    if name == "knn":
        return KNeighborsClassifier(n_jobs=1, **params)
    if name == "nb":
        return GaussianNB(**params)
    if name == "svm":
        params = dict(params)
        mode = params.pop("mode", "nystroem")
        if mode == "exact":                     # exact libsvm SVC, Platt scaling
            params.pop("n_components", None)
            return CalibratedClassifierCV(SVC(random_state=seed, **params), method="sigmoid",
                                          cv=3, ensemble=False)
        return NystroemSVM(random_state=seed, **params)
    raise ValueError(name)


def make_model(name: str, cfg: dict, seed: int, tuned: dict | None = None):
    params = dict(cfg["baselines"][name]["params"])
    params.update(tuned or {})
    return make_resampling_pipeline(make_classifier(name, params, seed), cfg, seed, name in CAPPED)


def param_grid(cfg: dict, name: str) -> list[dict]:
    grid = cfg["baselines"][name]["grid"]
    keys = sorted(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]
