"""Config loading, run logging, seeding and small shared helpers."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


class DataMissingError(RuntimeError):
    """Raised when required raw data is absent. Never fall back to fake data."""


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_path"] = str(path)
    cfg["_hash"] = config_hash(cfg)
    return cfg


def config_hash(cfg: dict) -> str:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    blob = json.dumps(clean, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def p(cfg: dict, key: str) -> Path:
    """Absolute path for a ``paths`` entry of the config."""
    return ROOT / cfg["paths"][key]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str = "tcse", cfg: dict | None = None) -> logging.Logger:
    if name in _LOGGERS:
        return _LOGGERS[name]
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    if cfg is not None:
        logdir = p(cfg, "results") / "logs"
        logdir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logdir / f"{name}_{time.strftime('%Y%m%d')}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    _LOGGERS[name] = log
    return log


def log_event(cfg: dict, event: str, **fields) -> None:
    """Append one structured JSON line to results/logs/runs.jsonl."""
    logdir = p(cfg, "results") / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event,
           "config_hash": cfg.get("_hash"), **fields}
    with open(logdir / "runs.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def save_json(obj, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    os.replace(tmp, path)


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@contextmanager
def timed(log: logging.Logger, label: str, timings: dict | None = None):
    t0 = time.perf_counter()
    log.info("start: %s", label)
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        log.info("done:  %s (%.1fs)", label, dt)
        if timings is not None:
            timings[label] = round(dt, 2)


def duck(cfg: dict):
    import duckdb
    tmp = p(cfg, "duckdb_temp")
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(config={
        "memory_limit": cfg["duckdb"]["memory_limit"],
        "threads": cfg["duckdb"]["threads"],
        "temp_directory": str(tmp),
        "preserve_insertion_order": False,
    })
    con.execute("SET enable_progress_bar=false")
    return con


def smart_col(attr_id: int, kind: str = "raw") -> str:
    return f"smart_{attr_id}_{kind}"
