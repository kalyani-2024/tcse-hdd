"""Run one stage of the TCSE pipeline.

Examples
  python scripts/run_pipeline.py --stage download
  python scripts/run_pipeline.py --stage integrate --dataset backblaze
  python scripts/run_pipeline.py --stage features  --dataset smartz
  python scripts/run_pipeline.py --stage run --dataset smartz --horizon 7 --seed 0
  python scripts/run_pipeline.py --stage run --dataset backblaze --horizon 30 --seed 3 --family no_trend
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings

from tcse.data import download
from tcse.data.integrate import integrate_backblaze, integrate_smartz, subsets
from tcse.experiments import select_window, specs_for
from tcse.features.trends import build_feature_store
from tcse.pipeline import run
from tcse.utils import DataMissingError, duck, get_logger, load_config

warnings.filterwarnings("ignore", category=FutureWarning)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", choices=["backblaze", "smartz"], default=None)
    ap.add_argument("--stage", required=True,
                    choices=["download", "integrate", "features", "window", "run"])
    ap.add_argument("--family", default="main",
                    choices=["main", "no_trend", "temporal", "shared", "svm_exact"])
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="recompute even if a cached result exists")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    log = get_logger("tcse", cfg)
    datasets = [a.dataset] if a.dataset else ["backblaze", "smartz"]
    try:
        if a.stage == "download":
            for d in datasets:
                if d == "backblaze":
                    download.download_backblaze(cfg)
                    download.extract_backblaze(cfg)
                else:
                    download.download_smartz(cfg)
        elif a.stage == "integrate":
            con = duck(cfg)
            for d in datasets:
                (integrate_backblaze if d == "backblaze" else integrate_smartz)(cfg, con)
        elif a.stage == "features":
            con = duck(cfg)
            for d in datasets:
                for sub in subsets(cfg, d):
                    build_feature_store(cfg, con, sub)
        elif a.stage == "window":
            for d in datasets:
                for sub in subsets(cfg, d):
                    log.info("%s: m = %d", sub["name"], select_window(cfg, sub))
        elif a.stage == "run":
            hs = [a.horizon] if a.horizon else None
            seeds = [a.seed] if a.seed is not None else None
            for d in datasets:
                for sub in subsets(cfg, d):
                    for spec in specs_for(cfg, a.family, sub, hs, seeds):
                        r = run(cfg, spec, force=a.force)
                        print(json.dumps({"run": f"{spec.family}/{spec.subset}/{spec.tag}",
                                          "tcse_r10": {k: r["results"]["tcse"]["r10"][k]
                                                       for k in ("precision", "recall", "f1", "mcc", "roc_auc")}}))
    except DataMissingError as e:
        log.error("DATA MISSING:\n%s", e)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
