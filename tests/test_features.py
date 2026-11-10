"""Trend features: correctness and no look-ahead; preprocessing fitted on training data only."""
import numpy as np
import pandas as pd
import pytest

from tcse.features.clean import clean_values, ffill_within_groups, group_starts
from tcse.features.transform import TCSEPreprocessor
from tcse.features.trends import rolling_window_stats


def _series(seed=0, n_drives=5, days=40):
    rng = np.random.default_rng(seed)
    sid = np.repeat(np.arange(n_drives), days)
    x = rng.poisson(3, size=n_drives * days).cumsum().astype(float)
    return sid, x


@pytest.mark.parametrize("m", [3, 7, 14])
def test_rolling_matches_pandas(m):
    sid, x = _series()
    st = rolling_window_stats(x, group_starts(sid), m, 2)
    df = pd.DataFrame({"g": sid, "x": x})
    r = df.groupby("g")["x"].rolling(m, min_periods=2)
    np.testing.assert_allclose(st["mean"], r.mean().to_numpy(), rtol=1e-10, equal_nan=True)
    np.testing.assert_allclose(st["std"], r.std().to_numpy(), rtol=1e-8, equal_nan=True)
    # slope over a full window equals the OLS slope (Eq. 3)
    t = m + 2
    w = x[t - m + 1:t + 1]
    k = np.arange(m)
    beta = ((k - k.mean()) * (w - w.mean())).sum() / ((k - k.mean()) ** 2).sum()
    assert st["slope"][t] == pytest.approx(beta)
    assert st["delta"][t] == pytest.approx(x[t] - x[t - m + 1])


def test_short_history_and_group_boundaries():
    sid = np.array([0, 0, 0, 1, 1])
    x = np.array([1.0, 2.0, 4.0, 10.0, 13.0])
    st = rolling_window_stats(x, group_starts(sid), 7, 2)
    # first row of each drive has < 2 values -> NaN
    assert np.isnan(st["mean"][0]) and np.isnan(st["mean"][3])
    # second drive never sees the first drive's values
    assert st["mean"][4] == pytest.approx(11.5)
    assert st["delta"][4] == pytest.approx(3.0)
    assert st["delta"][2] == pytest.approx(3.0)      # earliest available value in window


@pytest.mark.parametrize("m", [3, 7, 14])
def test_no_look_ahead(m):
    sid, x = _series(1)
    st0 = rolling_window_stats(x, group_starts(sid), m, 2)
    t = 20
    x2 = x.copy()
    x2[t + 1:] += 1000.0 * np.arange(1, len(x) - t)  # change only future values
    st1 = rolling_window_stats(x2, group_starts(sid), m, 2)
    for s in st0:
        np.testing.assert_array_equal(st0[s][:t + 1], st1[s][:t + 1])


def test_ffill_within_drive_only():
    sid = np.array([0, 0, 0, 1, 1])
    x = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    out = ffill_within_groups(x, group_starts(sid))
    assert out[1] == 1.0 and np.isnan(out[3]) and out[4] == 5.0


def test_clean_values():
    x = np.array([-1.0, 30.0, 90.0, 0.0])
    t = clean_values(x, 194, 194, (0, 80))
    assert np.isnan(t[0]) and t[1] == 30 and np.isnan(t[2]) and t[3] == 0
    c = clean_values(x, 5, 194, (0, 80))
    assert np.isnan(c[0]) and c[2] == 90


def _frame(seed, n=3000):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "smart_5_raw": rng.poisson(1, n).astype(float),
        "smart_9_raw": rng.uniform(1000, 5000, n),
        "smart_9_raw__slope_w7": rng.normal(0, 3, n),
        "smart_9_raw__mean_w7": 0.0,        # filled below: near-duplicate of smart_9_raw
        "smart_187_raw": np.nan,            # empty attribute -> dropped
        "miss_5": (rng.random(n) < 0.05).astype(float),
        "miss_187": 1.0,
    })
    df["smart_9_raw__mean_w7"] = df["smart_9_raw"] + rng.normal(0, 1, n)
    df.loc[rng.random(n) < 0.05, "smart_5_raw"] = np.nan
    y = (rng.random(n) < 0.05).astype(int)
    df.loc[y == 1, "smart_5_raw"] += 5
    return df, y


def test_preprocessor_fitted_on_train_only():
    df, y = _frame(0)
    tr, te = df.iloc[:2000].copy(), df.iloc[2000:].copy()
    pre = TCSEPreprocessor(flag_columns=["miss_5", "miss_187"], corr_sample_rows=10_000,
                           mi_max_rows=10_000, random_state=0).fit(tr, y[:2000])
    snap = (pre.medians_.copy(), pre.mean_.copy(), pre.scale_.copy(), list(pre.selected_),
            list(pre.input_columns_))
    # mutate test rows: fitted parameters cannot change
    te.loc[:, :] = te.to_numpy() * 100 + 7
    Z = pre.transform(te)
    assert np.isfinite(Z).all()
    after = (pre.medians_, pre.mean_, pre.scale_, pre.selected_, pre.input_columns_)
    for a, b in zip(snap, after):
        np.testing.assert_array_equal(np.asarray(a, dtype=object), np.asarray(b, dtype=object))
    # refitting on a mutated *train* frame does change them (sanity check)
    tr2 = tr.copy()
    tr2["smart_9_raw"] += 5000.0
    pre2 = TCSEPreprocessor(flag_columns=["miss_5", "miss_187"], corr_sample_rows=10_000,
                            mi_max_rows=10_000, random_state=0).fit(tr2, y[:2000])
    j2 = pre2.input_columns_.index("smart_9_raw")
    assert pre2.medians_[j2] != pytest.approx(pre.medians_[pre.input_columns_.index("smart_9_raw")])
    # drop rules: empty attribute removed with its flag; one of a correlated pair removed
    assert "smart_187_raw" not in pre.input_columns_ and "miss_187" not in pre.input_columns_
    assert not {"smart_9_raw", "smart_9_raw__mean_w7"} <= set(pre.selected_)
    # medians are training medians
    j = pre.input_columns_.index("smart_9_raw")
    assert pre.medians_[j] == pytest.approx(np.nanmedian(tr["smart_9_raw"]), rel=1e-5)


def test_transform_selected_equals_transform():
    df, y = _frame(1)
    pre = TCSEPreprocessor(flag_columns=["miss_5", "miss_187"], corr_sample_rows=10_000,
                           mi_max_rows=10_000).fit(df, y)
    a = pre.transform(df)
    b = pre.transform_selected(df[pre.selected_].to_numpy(dtype=np.float32))
    np.testing.assert_allclose(a, b, rtol=1e-6)
