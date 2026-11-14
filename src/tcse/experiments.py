"""Experiment grid: which runs make up each table / figure of the report."""
from __future__ import annotations

import numpy as np

from tcse.data.integrate import subsets
from tcse.features.trends import FeatureStore
from tcse.pipeline import RunSpec, result_path, run
from tcse.utils import get_logger, load_json, p, save_json, smart_col

log = get_logger()

DATASETS = ("backblaze", "smartz")
FAMILIES = ("window_selection", "main", "no_trend", "temporal", "shared", "svm_exact")


def all_subsets(cfg) -> list[dict]:
    return [s for d in DATASETS for s in subsets(cfg, d)]


def window_path(cfg, subset: str):
    return p(cfg, "results") / "metrics" / "window_selection" / subset / "chosen.json"


def select_window(cfg, sub: dict) -> int:
    """Choose m in cfg.windows on the validation set: highest TCSE validation
    average precision (threshold-free, so it stays informative when the
    cost-optimal decision at some ratio is to raise no alarm); validation cost
    at the configured ratio breaks ties. Test data are not used."""
    path = window_path(cfg, sub["name"])
    if path.exists():
        return int(load_json(path)["window"])
    ws = cfg["window_selection"]
    costs, aps = {}, {}
    for m in cfg["windows"]:
        r = run(cfg, RunSpec("window_selection", sub["name"], sub["dataset"], ws["horizon"], ws["seed"], m))
        costs[m] = r["results"]["tcse"][f"r{ws['cost_ratio']}"]["val_cost"]
        aps[m] = r["validation"]["tcse"]["average_precision"]
    best = min(costs, key=lambda m: (-aps[m], costs[m], m != cfg["default_window"], m))
    save_json({"window": best, "val_average_precision": aps, "val_cost": costs, "criterion":
               f"max TCSE validation average precision (ties: validation cost at r={ws['cost_ratio']}), "
               f"H={ws['horizon']}, seed={ws['seed']}"}, path)
    log.info("window for %s: m=%d (validation AP %s, costs %s)", sub["name"], best, aps, costs)
    return best


def shared_attributes(cfg) -> list[int]:
    """SMART attributes reported (<= max_missing_frac empty) by every subset of both datasets.
    A schema property: decided from missingness only, never from labels."""
    shared = None
    for sub in all_subsets(cfg):
        st = FeatureStore(cfg, sub["name"])
        ok = set()
        for a in cfg["attributes"]:
            miss = float(np.mean(st.column(f"miss_{a}")))
            if miss <= cfg["cleaning"]["max_missing_frac"]:
                ok.add(a)
        shared = ok if shared is None else shared & ok
    return sorted(shared)


def specs_for(cfg, family: str, sub: dict, horizons=None, seeds=None) -> list[RunSpec]:
    horizons = horizons or cfg["horizons"]
    if seeds is None:
        seeds = (cfg.get("family_seeds") or {}).get(family) or cfg["project"]["seeds"]
    m = select_window(cfg, sub)
    out = []
    if family == "main":
        out = [RunSpec("main", sub["name"], sub["dataset"], h, s, m) for s in seeds for h in horizons]
    elif family == "no_trend":
        out = [RunSpec("no_trend", sub["name"], sub["dataset"], h, s, m, variant="no_trend")
               for s in seeds for h in horizons]
    elif family == "temporal":
        out = [RunSpec("temporal", sub["name"], sub["dataset"], h, s, m, split="temporal")
               for s in seeds for h in horizons]
    elif family == "shared":
        attrs = tuple(shared_attributes(cfg))
        others = [o for o in all_subsets(cfg) if o["dataset"] != sub["dataset"]]
        hs = horizons if horizons != cfg["horizons"] else cfg["cross_dataset"]["horizons"]
        for o in others:
            out += [RunSpec("shared", sub["name"], sub["dataset"], h, s, m, attrs=attrs,
                            cross_target={"subset": o["name"], "dataset": o["dataset"]})
                    for s in seeds for h in hs]
    elif family == "svm_exact":
        c = cfg["svm_check"]
        if sub["dataset"] != c["dataset"]:
            return []
        main = run(cfg, RunSpec("main", sub["name"], sub["dataset"], c["horizon"], c["seed"], m))
        override = dict(cfg["baselines"]["svm"]["exact_params"], **{k: v for k, v in main["tuning"]["svm"]["best"].items()
                                                                     if k in ("C", "gamma")})
        out = [RunSpec("svm_exact", sub["name"], sub["dataset"], c["horizon"], c["seed"], m,
                       svm_override=override)]
    else:
        raise ValueError(family)
    return out


def is_done(cfg, spec: RunSpec) -> bool:
    return result_path(cfg, spec).exists()
