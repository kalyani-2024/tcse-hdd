"""Trend feature engineering (report Section IV-C step 4, Eqs. 2-3) and the
on-disk feature store.

For each key attribute x and each window m, over the backward-looking window
of the last m records of the same drive (current record included):
    mean   mu_t    = mean(x_{t-m+1..t})
    std            = sample standard deviation (ddof = 1)
    delta  Delta_t = x_t - x_{t-m+1}   (earliest available value if history < m)
    slope  beta_t  = least-squares slope of x against record index k (Eq. 3)
A statistic is NaN when fewer than ``min_history`` values are available.
Only past and current rows are used, so no future value can leak backwards.

The feature store is a column-major (Fortran-order) float32 memory-mapped
array, so single columns stream contiguously from disk and the full matrix
never has to fit in RAM.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tcse.data.integrate import interim_path
from tcse.data.label import NO_FAIL
from tcse.features.clean import clean_values, ffill_within_groups, group_starts
from tcse.utils import DataMissingError, get_logger, log_event, p, save_json, smart_col

log = get_logger()

STATS = ("mean", "std", "delta", "slope")


def trend_name(attr_id: int, stat: str, m: int) -> str:
    return f"{smart_col(attr_id)}__{stat}_w{m}"


def rolling_window_stats(x: np.ndarray, starts: np.ndarray, m: int, min_history: int = 2) -> dict:
    """Exact backward-looking rolling statistics within groups.

    x      : float64 values, rows sorted by (drive, date)
    starts : index of the first row of each row's drive (see group_starts)
    """
    n = len(x)
    if n == 0:
        return {s: np.zeros(0) for s in STATS}
    offs = np.arange(m)
    pos = np.arange(n)[:, None] - (m - 1) + offs[None, :]          # n x m, oldest -> newest
    inside = pos >= starts[:, None]                                 # same drive only
    W = np.where(inside, x[np.clip(pos, 0, None)], np.nan)
    valid = ~np.isnan(W)
    cnt = valid.sum(axis=1)
    with np.errstate(all="ignore"):
        Wz = np.where(valid, W, 0.0)
        mean = Wz.sum(axis=1) / cnt
        dev = np.where(valid, W - mean[:, None], 0.0)
        std = np.sqrt((dev ** 2).sum(axis=1) / (cnt - 1))
        first = W[np.arange(n), np.argmax(valid, axis=1)]
        delta = W[:, -1] - first
        k = np.where(valid, offs[None, :].astype(np.float64), 0.0)
        kbar = k.sum(axis=1) / cnt
        kd = np.where(valid, offs[None, :] - kbar[:, None], 0.0)
        slope = (kd * dev).sum(axis=1) / (kd ** 2).sum(axis=1)
    bad = cnt < min_history
    out = {"mean": mean, "std": std, "delta": delta, "slope": slope}
    for v in out.values():
        v[bad] = np.nan
    return out


# ---------------------------------------------------------------------- store
def store_dir(cfg: dict, subset: str) -> Path:
    return p(cfg, "processed") / subset


def feature_columns(cfg: dict, attrs: list[int] | None = None) -> list[str]:
    attrs = attrs or cfg["attributes"]
    cols = [smart_col(a) for a in attrs] + [f"miss_{a}" for a in attrs]
    if cfg["include_normalized"]:
        cols += [smart_col(a, "normalized") for a in attrs]
    for m in cfg["windows"]:
        cols += [trend_name(a, s, m) for a in attrs for s in STATS]
        cols.append(f"n_delta_pos_w{m}")
    return cols


def build_feature_store(cfg: dict, con, subset: dict, chunk_rows: int = 250_000) -> Path:
    """Clean, forward-fill and compute trend features for one modelling subset."""
    out = store_dir(cfg, subset["name"])
    done = out / "store.json"
    if done.exists():
        log.info("feature store exists: %s", out)
        return out
    src = interim_path(cfg, subset["dataset"])
    if not src.exists():
        raise DataMissingError(f"{src} missing; run --stage integrate first")
    out.mkdir(parents=True, exist_ok=True)
    attrs = cfg["attributes"]
    crit = [a for a in cfg["critical_attributes"] if a in attrs]
    models = subset["models"]
    mlist = ", ".join("'" + m.replace("'", "''") + "'" for m in models)
    cols = feature_columns(cfg)
    col_idx = {c: i for i, c in enumerate(cols)}
    n = con.execute(f"SELECT count(*) FROM read_parquet('{src.as_posix()}') WHERE model IN ({mlist})").fetchone()[0]
    log.info("building feature store %s: %d rows x %d columns", subset["name"], n, len(cols))

    X = np.lib.format.open_memmap(out / "X.npy", mode="w+", dtype=np.float32, shape=(n, len(cols)),
                                  fortran_order=True)
    raw_sel = ", ".join(smart_col(a) for a in attrs)
    norm_sel = (", " + ", ".join(smart_col(a, "normalized") for a in attrs)) if cfg["include_normalized"] else ""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE src AS
        SELECT dense_rank() OVER (ORDER BY serial) - 1 AS serial_id, serial, model,
               (date - DATE '1970-01-01')::INTEGER AS day,
               coalesce((fail_date - DATE '1970-01-01')::INTEGER, {NO_FAIL}) AS fail,
               {raw_sel}{norm_sel}
        FROM read_parquet('{src.as_posix()}') WHERE model IN ({mlist})
    """)
    con.execute(f"""COPY (SELECT DISTINCT serial_id, serial, model FROM src ORDER BY serial_id)
                   TO '{(out / 'serials.parquet').as_posix()}' (FORMAT parquet)""")
    reader = con.sql("SELECT * EXCLUDE (serial) FROM src ORDER BY serial_id, day").to_arrow_reader(chunk_rows)

    meta_parts = []
    carry = None
    pos = 0
    t_cfg = cfg["cleaning"]
    model_codes = {m: i for i, m in enumerate(models)}

    def process(df: pd.DataFrame):
        nonlocal pos
        k = len(df)
        sid = df["serial_id"].to_numpy()
        starts = group_starts(sid)
        block = np.full((k, len(cols)), np.nan, dtype=np.float32)
        deltas = {}
        for a in attrs:
            x = clean_values(df[smart_col(a)].to_numpy(dtype=np.float64, na_value=np.nan), a,
                             cfg["temperature_attribute"], t_cfg["temperature_range"])
            block[:, col_idx[f"miss_{a}"]] = np.isnan(x)
            xf = ffill_within_groups(x, starts)
            block[:, col_idx[smart_col(a)]] = xf
            if cfg["include_normalized"]:
                xn = df[smart_col(a, "normalized")].to_numpy(dtype=np.float64, na_value=np.nan)
                block[:, col_idx[smart_col(a, "normalized")]] = ffill_within_groups(xn, starts)
            for m in cfg["windows"]:
                st = rolling_window_stats(xf, starts, m, t_cfg["min_history"])
                for s in STATS:
                    block[:, col_idx[trend_name(a, s, m)]] = st[s]
                deltas[(a, m)] = st["delta"]
        for m in cfg["windows"]:
            cnt = np.zeros(k, dtype=np.float32)
            for a in crit:
                with np.errstate(invalid="ignore"):
                    cnt += np.nan_to_num(deltas[(a, m)], nan=0.0) > 0
            block[:, col_idx[f"n_delta_pos_w{m}"]] = cnt
        X[pos:pos + k, :] = block
        meta_parts.append(pd.DataFrame({
            "serial_id": sid.astype(np.int32), "day": df["day"].to_numpy(np.int32),
            "fail": df["fail"].to_numpy(np.int32),
            "model_id": df["model"].map(model_codes).to_numpy(np.int8)}))
        pos += k

    for batch in reader:
        df = batch.to_pandas()
        if carry is not None:
            df = pd.concat([carry, df], ignore_index=True)
        # keep the last (possibly incomplete) drive for the next batch
        last = df["serial_id"].iloc[-1]
        cut = int(np.searchsorted(df["serial_id"].to_numpy(), last, side="left"))
        if cut == 0:
            carry = df
            continue
        carry = df.iloc[cut:].reset_index(drop=True)
        process(df.iloc[:cut].reset_index(drop=True))
        log.info("  %s: %d / %d rows", subset["name"], pos, n)
    if carry is not None and len(carry):
        process(carry)
    con.execute("DROP TABLE src")
    if pos != n:
        raise RuntimeError(f"row count mismatch {pos} != {n}")
    X.flush()
    del X
    meta = pd.concat(meta_parts, ignore_index=True)
    meta.to_parquet(out / "meta.parquet", index=False)
    info = {"subset": subset, "n_rows": int(n), "columns": cols, "models": models,
            "n_drives": int(meta["serial_id"].nunique()),
            "n_failed_drives": int(meta.loc[meta["fail"] != NO_FAIL, "serial_id"].nunique())}
    save_json(info, done)
    log_event(cfg, "feature_store", subset=subset["name"], n_rows=n, n_drives=info["n_drives"],
              n_failed=info["n_failed_drives"])
    return out


class FeatureStore:
    """Read access to a built feature store (memory-mapped, column-major)."""

    def __init__(self, cfg: dict, subset: str):
        self.dir = store_dir(cfg, subset)
        if not (self.dir / "store.json").exists():
            raise DataMissingError(f"feature store {self.dir} missing; run --stage features")
        self.info = json.loads((self.dir / "store.json").read_text())
        self.columns: list[str] = self.info["columns"]
        self.col_idx = {c: i for i, c in enumerate(self.columns)}
        self.X = np.load(self.dir / "X.npy", mmap_mode="r")
        self.meta = pd.read_parquet(self.dir / "meta.parquet")
        self.models = self.info["models"]

    def column(self, name: str, rows: np.ndarray | None = None) -> np.ndarray:
        col = self.X[:, self.col_idx[name]]
        return np.asarray(col if rows is None else col[rows], dtype=np.float32)

    def block(self, names: list[str], rows: np.ndarray) -> np.ndarray:
        out = np.empty((len(rows), len(names)), dtype=np.float32)
        for j, c in enumerate(names):
            out[:, j] = self.X[rows, self.col_idx[c]]
        return out
