"""Leakage-free splitting (report Section IV-C step 7).

Main split: drives (not rows) go to train / validation / test (70/15/15),
stratified on (drive model, ever-fails) so that every split contains failed
drives. This is a stratified *group* split: the group is the serial number.

Temporal hold-out: train on the early period and test on the later one,
with a gap of H days between the last training record and the first test
record, so no training label depends on a failure inside the test period.

Grouped CV folds inside the training split: drives are assigned to folds by
stratified K-fold over drives (stratified on "drive has a positive row"),
which is a stratified group K-fold with groups = serial number.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from tcse.data.label import NO_FAIL

SPLITS = ("train", "val", "test")


def drive_table(meta: pd.DataFrame) -> pd.DataFrame:
    d = meta.groupby("serial_id", sort=True).agg(model_id=("model_id", "first"), fail=("fail", "first"))
    d["ever_failed"] = (d["fail"] != NO_FAIL).astype(np.int8)
    return d.reset_index()


def random_drive_split(drives: pd.DataFrame, seed: int, fracs=(0.70, 0.15, 0.15)) -> pd.Series:
    """serial_id -> 'train' | 'val' | 'test'."""
    tr, va, te = fracs
    strat = drives["model_id"].astype(str) + "_" + drives["ever_failed"].astype(str)
    ids = drives["serial_id"].to_numpy()
    id_tr, id_rest, _, s_rest = train_test_split(ids, strat, train_size=tr, stratify=strat,
                                                 random_state=seed)
    id_va, id_te = train_test_split(id_rest, train_size=va / (va + te), stratify=s_rest,
                                    random_state=seed)
    out = pd.Series("train", index=pd.Index(ids, name="serial_id"), dtype=object)
    out.loc[id_va] = "val"
    out.loc[id_te] = "test"
    assert_disjoint(out)
    return out


def assert_disjoint(assign: pd.Series) -> None:
    """Every serial belongs to exactly one split."""
    if assign.index.has_duplicates:
        raise AssertionError("a serial number appears in more than one split")
    sets = {s: set(assign.index[assign == s]) for s in SPLITS}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            if sets[a] & sets[b]:
                raise AssertionError(f"serial overlap between {a} and {b}")


def split_path(store_dir: Path, seed: int) -> Path:
    return store_dir / "splits" / f"seed{seed}.parquet"


def get_split(store, seed: int, fracs) -> pd.Series:
    """Load (or create and save) the serial -> split assignment for a seed."""
    path = split_path(store.dir, seed)
    if path.exists():
        s = pd.read_parquet(path).set_index("serial_id")["split"]
        assert_disjoint(s)
        return s
    s = random_drive_split(drive_table(store.meta), seed, fracs)
    path.parent.mkdir(parents=True, exist_ok=True)
    s.rename("split").reset_index().to_parquet(path, index=False)
    return s


def rows_by_split(serial_id: np.ndarray, keep: np.ndarray, assign: pd.Series) -> dict:
    lab = assign.reindex(np.arange(int(serial_id.max()) + 1)).to_numpy()
    row_split = lab[serial_id]
    return {s: np.flatnonzero(keep & (row_split == s)) for s in SPLITS}


def temporal_rows(day: np.ndarray, keep_train: np.ndarray, keep_test: np.ndarray,
                  serial_id: np.ndarray, train_end: int, test_end: int, horizon: int,
                  val_frac: float, seed: int, y_train: np.ndarray) -> dict:
    """Row indices for the temporal hold-out.

    train/val : day <= train_end - H (labels stay inside the training period),
                drives split into train / val (stratified on having a positive row)
    test      : train_end < day <= test_end (all drives present in that period)
    The H-day gap (train_end - H, train_end] holds no training or test rows.
    """
    tr_mask = keep_train & (day <= train_end - horizon)
    te_mask = keep_test & (day > train_end) & (day <= test_end)
    cand = np.flatnonzero(tr_mask)
    drv = pd.DataFrame({"sid": serial_id[cand], "y": y_train[cand]}).groupby("sid")["y"].max()
    ids = drv.index.to_numpy()
    id_tr, id_va = train_test_split(ids, test_size=val_frac, stratify=drv.to_numpy(), random_state=seed)
    is_val = np.zeros(int(serial_id.max()) + 1, dtype=bool)
    is_val[id_va] = True
    v = is_val[serial_id[cand]]
    return {"train": cand[~v], "val": cand[v], "test": np.flatnonzero(te_mask)}


def group_folds(serial_id: np.ndarray, y: np.ndarray, n_splits: int, seed: int) -> list:
    """Stratified group K-fold over rows; returns [(train_pos, heldout_pos)] as
    positions into the given arrays. No drive is in two folds."""
    drv = pd.DataFrame({"sid": serial_id, "y": y}).groupby("sid")["y"].max()
    ids, strat = drv.index.to_numpy(), drv.to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_of = np.full(int(serial_id.max()) + 1, -1, dtype=np.int16)
    for k, (_, te) in enumerate(skf.split(ids, strat)):
        fold_of[ids[te]] = k
    rf = fold_of[serial_id]
    return [(np.flatnonzero(rf != k), np.flatnonzero(rf == k)) for k in range(n_splits)]
