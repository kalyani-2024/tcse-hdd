"""Labelling (Eq. 1) and censoring on a hand-made drive history.
Synthetic fixtures are used only here, to test correctness."""
import numpy as np
import pandas as pd
import pytest

from tcse.data.label import NO_FAIL, label_frame, label_rows


def _history():
    # drive A fails on day 10; drive B never fails; period = days 1..20
    days = np.r_[np.arange(1, 13), np.arange(1, 21)]
    fail = np.r_[np.full(12, 10), np.full(20, NO_FAIL)]
    serial = np.r_[np.zeros(12, int), np.ones(20, int)]
    return days, fail, serial


@pytest.mark.parametrize("H", [1, 7, 14, 30])
def test_labels_match_eq1(H):
    days, fail, serial = _history()
    keep, y = label_rows(days, fail, H, label_end=20)
    for d, f, k, lab in zip(days, fail, keep, y):
        has = f != NO_FAIL
        # post-failure rows (on or after the failure day) are removed
        if has and d >= f:
            assert not k
            continue
        # censoring: rows whose label window (t, t+H] passes the period end are removed
        if d + H > 20:
            assert not k
            continue
        assert k
        expected = int(has and d < f <= d + H)
        assert lab == expected, (d, f, H)


def test_specific_values():
    days, fail, serial = _history()
    keep, y = label_rows(days, fail, 7, label_end=20)
    a = (serial == 0) & keep
    # drive A: days 3..9 are within 7 days before failure (day 10)
    assert list(days[a][y[a] == 1]) == [3, 4, 5, 6, 7, 8, 9]
    assert list(days[a][y[a] == 0]) == [1, 2]
    # rows on/after failure day 10 removed
    assert days[a].max() == 9
    # drive B: last 7 days (14..20) censored
    b = (serial == 1) & keep
    assert days[b].max() == 13 and y[b].sum() == 0


def test_h1_only_last_day_positive():
    days, fail, _ = _history()
    keep, y = label_rows(days, fail, 1, label_end=20)
    assert list(days[keep & (y == 1)]) == [9]


def test_label_frame_wrapper():
    df = pd.DataFrame({"serial": ["a"] * 5, "date": pd.date_range("2026-01-01", periods=5),
                       "fail_date": pd.Timestamp("2026-01-04")})
    out = label_frame(df, 2, "2026-01-31")
    assert out["date"].dt.day.tolist() == [1, 2, 3]
    assert out["y"].tolist() == [0, 1, 1]
