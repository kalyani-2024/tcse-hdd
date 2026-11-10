"""No leakage across splits or CV folds."""
import numpy as np
import pandas as pd
import pytest

from tcse.data.label import NO_FAIL
from tcse.split import (assert_disjoint, drive_table, group_folds, random_drive_split, rows_by_split,
                        temporal_rows)


def _meta(n_drives=400, days=30, seed=0):
    rng = np.random.default_rng(seed)
    sid = np.repeat(np.arange(n_drives), days)
    day = np.tile(np.arange(days), n_drives)
    failed = rng.random(n_drives) < 0.1
    fail = np.where(failed, days - 1, NO_FAIL)[sid]
    model = (np.arange(n_drives) % 2)[sid]
    return pd.DataFrame({"serial_id": sid, "day": day, "fail": fail.astype(np.int64), "model_id": model})


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_no_serial_in_two_splits(seed):
    meta = _meta(seed=seed)
    assign = random_drive_split(drive_table(meta), seed, (0.7, 0.15, 0.15))
    assert_disjoint(assign)
    R = rows_by_split(meta["serial_id"].to_numpy(), np.ones(len(meta), bool), assign)
    sets = {k: set(meta["serial_id"].to_numpy()[v]) for k, v in R.items()}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"])
    assert sum(len(v) for v in R.values()) == len(meta)
    # every split contains failed drives
    d = drive_table(meta).set_index("serial_id")
    for k in sets:
        assert d.loc[list(sets[k]), "ever_failed"].sum() > 0


def test_assert_disjoint_detects_overlap():
    bad = pd.Series(["train", "test"], index=pd.Index([5, 5], name="serial_id"))
    with pytest.raises(AssertionError):
        assert_disjoint(bad)


def test_group_folds_no_drive_in_two_folds():
    meta = _meta()
    sid = meta["serial_id"].to_numpy()
    y = (meta["fail"].to_numpy() != NO_FAIL).astype(int)
    folds = group_folds(sid, y, 5, 0)
    seen = set()
    for tr, ho in folds:
        assert not set(sid[tr]) & set(sid[ho])
        assert not seen & set(sid[ho])
        seen |= set(sid[ho])
        assert y[ho].sum() > 0
    assert seen == set(sid)


def test_temporal_gap():
    meta = _meta()
    day, sid = meta["day"].to_numpy(), meta["serial_id"].to_numpy()
    keep = np.ones(len(day), bool)
    y = np.zeros(len(day), int)
    y[::50] = 1
    H, tr_end, te_end = 3, 15, 29
    R = temporal_rows(day, keep, keep, sid, tr_end, te_end, H, 0.2, 0, y)
    assert day[R["train"]].max() <= tr_end - H and day[R["val"]].max() <= tr_end - H
    assert day[R["test"]].min() > tr_end
    assert not set(sid[R["train"]]) & set(sid[R["val"]])
