"""Resampling scope and cost-optimal threshold search."""
import numpy as np
import pytest

from tcse.evaluate import best_threshold, confusion_curve, evaluate, metrics_from_counts, thresholds
from tcse.models.tcse import balanced_weights
from tcse.sampling import make_resampling_pipeline, undersample_rows
from tcse.split import group_folds
from tcse.utils import load_config
from sklearn.tree import DecisionTreeClassifier


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_undersample_only_within_given_rows():
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(10_000, 3000, replace=False))   # e.g. one CV fold's training rows
    y = (rng.random(len(rows)) < 0.02).astype(int)
    keep = undersample_rows(rows, y, 20, 0)
    kept_rows = rows[keep]
    assert set(kept_rows) <= set(rows)
    assert y[keep].sum() == y.sum()                            # all positives kept
    assert (y[keep] == 0).sum() == 20 * y.sum()


def test_pipeline_resamples_only_training_data(cfg):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4000, 3)).astype(np.float32)
    y = (rng.random(4000) < 0.03).astype(int)
    X[y == 1] += 2
    pipe = make_resampling_pipeline(DecisionTreeClassifier(max_depth=3, random_state=0), cfg, 0, capped=True)
    pipe.fit(X, y)
    # prediction never resamples: one output per input row, input untouched
    Xv = rng.normal(size=(777, 3)).astype(np.float32)
    Xv_copy = Xv.copy()
    assert pipe.predict_proba(Xv).shape == (777, 2)
    np.testing.assert_array_equal(Xv, Xv_copy)
    # after SMOTE the class ratio is smote_ratio
    Xr, yr = pipe.steps[0][1].fit_resample(X, y)
    Xs, ys = pipe.steps[1][1].fit_resample(Xr, yr)
    assert (ys == 1).sum() == pytest.approx(cfg["sampling"]["smote_ratio"] * (ys == 0).sum(), abs=1)


def test_oof_models_never_see_their_fold():
    rng = np.random.default_rng(2)
    sid = np.repeat(np.arange(200), 10)
    y = np.zeros(len(sid), int)
    y[rng.choice(len(sid), 60, replace=False)] = 1
    for fit_p, hold_p in group_folds(sid, y, 5, 0):
        keep = undersample_rows(fit_p, y[fit_p], 20, 0)
        assert not set(fit_p[keep]) & set(hold_p)
        assert not set(sid[fit_p]) & set(sid[hold_p])


def test_threshold_search_analytic(cfg):
    # positives score 0.8, negatives: 50 at 0.3, 10 at 0.6
    y = np.r_[np.ones(5), np.zeros(60)]
    p = np.r_[np.full(5, 0.8), np.full(50, 0.3), np.full(10, 0.6)]
    grid = thresholds(cfg)
    # r=10: any theta in (0.6, 0.8] gives cost 0 -> first such grid point 0.61
    th, c = best_threshold(y, p, grid, 10)
    assert th == pytest.approx(0.61) and c.min() == 0
    # make the 0.8-scored group mixed: 3 pos + 2 neg at 0.8, 2 pos at 0.6
    y2 = np.r_[np.ones(3), np.zeros(2), np.ones(2), np.zeros(10)]
    p2 = np.r_[np.full(5, 0.8), np.full(2, 0.6), np.full(10, 0.6)]
    # theta<=0.6: FN=0, FP=12 -> cost 12; 0.6<theta<=0.8: FN=2, FP=2 -> 2r+2; theta>0.8: FN=5 -> 5r
    th5, _ = best_threshold(y2, p2, grid, 5)      # 12 vs 12 vs 25 -> tie, first grid point wins
    assert th5 == pytest.approx(0.01)
    th3, _ = best_threshold(y2, p2, grid, 3)      # 12 vs 8 vs 15 -> (0.6, 0.8]
    assert th3 == pytest.approx(0.61)
    th50, _ = best_threshold(y2, p2, grid, 50)    # 12 vs 102 vs 250 -> theta <= 0.6
    assert th50 == pytest.approx(0.01)


def test_confusion_curve_matches_pointwise(cfg):
    rng = np.random.default_rng(3)
    y = (rng.random(500) < 0.1).astype(int)
    p = rng.random(500)
    grid = thresholds(cfg)
    cc = confusion_curve(y, p, grid)
    for j in [0, 17, 50, 98]:
        e = evaluate(y, p, grid[j], [10])
        assert (e["tp"], e["fp"], e["fn"], e["tn"]) == (cc["tp"][j], cc["fp"][j], cc["fn"][j], cc["tn"][j])


def test_metrics_and_mcc():
    m = metrics_from_counts(tp=8, fp=2, fn=2, tn=88)
    assert m["precision"] == pytest.approx(0.8) and m["recall"] == pytest.approx(0.8)
    assert m["mcc"] == pytest.approx((8 * 88 - 4) / np.sqrt(10 * 10 * 90 * 90))


def test_balanced_weights_ipw():
    y = np.r_[np.ones(10), np.zeros(100)]
    w = np.r_[np.ones(10), np.full(100, 10.0)]   # healthy rows sampled at 10 %
    bw = balanced_weights(y.astype(int), w)
    assert bw[y == 1].sum() == pytest.approx(bw[y == 0].sum())
