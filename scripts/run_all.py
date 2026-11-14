"""Full experiment grid, resumable (finished runs are cached and skipped).

    python scripts/run_all.py --config configs/default.yaml
    python scripts/run_all.py --families main no_trend --datasets smartz

Order: download/verify -> integrate -> feature stores -> window selection ->
main grid (H x seeds) -> no-trend ablation -> temporal hold-out ->
shared-attribute / cross-dataset runs -> exact-SVC approximation check.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings

from tcse.data import download
from tcse.data.integrate import drives_summary_path, integrate_backblaze, integrate_smartz, interim_path
from tcse.experiments import FAMILIES, all_subsets, is_done, select_window, specs_for
from tcse.features.trends import build_feature_store, store_dir
from tcse.pipeline import run
from tcse.utils import DataMissingError, duck, get_logger, load_config, log_event

warnings.filterwarnings("ignore", category=FutureWarning)


def prepare(cfg, log):
    con = duck(cfg)
    for d in ("backblaze", "smartz"):
        download.verify_raw(cfg, d)
        if not (interim_path(cfg, d).exists() and drives_summary_path(cfg, d).exists()):
            (integrate_backblaze if d == "backblaze" else integrate_smartz)(cfg, con)
    for sub in all_subsets(cfg):
        if not (store_dir(cfg, sub["name"]) / "store.json").exists():
            build_feature_store(cfg, con, sub)
    con.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--families", nargs="+", default=[f for f in FAMILIES if f != "window_selection"])
    ap.add_argument("--datasets", nargs="+", default=["backblaze", "smartz"])
    ap.add_argument("--horizons", nargs="+", type=int, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    log = get_logger("tcse", cfg)
    try:
        prepare(cfg, log)
        subs = [s for d in a.datasets for s in all_subsets(cfg) if s["dataset"] == d]
        for sub in subs:
            log.info("window for %s: m=%d", sub["name"], select_window(cfg, sub))
        for fam in a.families:
            specs = [sp for sub in subs for sp in specs_for(cfg, fam, sub, a.horizons, a.seeds)]
            todo = [sp for sp in specs if not is_done(cfg, sp)]
            log.info("family %s: %d runs (%d cached)", fam, len(specs), len(specs) - len(todo))
            for i, sp in enumerate(todo, 1):
                t = time.time()
                run(cfg, sp)
                log.info("[%s %d/%d] %s/%s done in %.0fs", fam, i, len(todo), sp.subset, sp.tag, time.time() - t)
            log_event(cfg, "family_done", family=fam, n_runs=len(specs))
    except DataMissingError as e:
        log.error("DATA MISSING:\n%s", e)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
