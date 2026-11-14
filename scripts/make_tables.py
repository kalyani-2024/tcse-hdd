"""Build every result table (CSV + LaTeX booktabs) and figure from saved run JSONs.

    python scripts/make_tables.py --config configs/default.yaml

Nothing here is typed by hand: every number is read from results/metrics/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from tcse.utils import get_logger, load_config, p

METHODS = ["dt", "knn", "nb", "svm", "tcse"]
LABEL = {"dt": "DT", "knn": "KNN", "nb": "NB", "svm": "SVM", "tcse": "TCSE",
         "tcse_no_if": "TCSE w/o IF", "tcse_no_trend": "TCSE w/o trend", "tcse_default": "TCSE @0.5"}
METRICS = ["precision", "recall", "f1", "roc_auc", "mcc"]
MLABEL = {"precision": "Prec.", "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC", "mcc": "MCC",
          "drive_recall": "Drive recall"}
DLABEL = {"backblaze": "Backblaze", "smartz": "SMART-Z"}


def chosen_window(cfg, subset: str) -> int | None:
    f = p(cfg, "results") / "metrics" / "window_selection" / subset / "chosen.json"
    return json.loads(f.read_text())["window"] if f.exists() else None


def load_family(cfg, family: str) -> list[dict]:
    """All cached runs of a family that used the window chosen for their subset."""
    root = p(cfg, "results") / "metrics" / family
    out = []
    for f in sorted(root.glob("*/H*_m*_s*.json")):
        r = json.loads(f.read_text())
        m = chosen_window(cfg, r["spec"]["subset"])
        if m is None or r["spec"]["window"] == m:
            out.append(r)
    return out


def long_rows(runs: list[dict], ratios, key="results", rename=None) -> pd.DataFrame:
    """One row per (subset, H, seed, method, setting) with all metrics."""
    rows = []
    for r in runs:
        s = r["spec"]
        block = r[key]["results"] if key == "cross" else r[key]
        if block is None:
            continue
        for meth, res in block.items():
            for setting, m in res.items():
                rows.append({"subset": s["subset"], "dataset": s["dataset"], "H": s["horizon"], "seed": s["seed"],
                             "method": (rename or {}).get(meth, meth), "setting": setting, **m})
    return pd.DataFrame(rows)


def fmt(mean, std, digits=3):
    if pd.isna(mean):
        return "--"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def fmt_cost(mean, std):
    return f"{mean:,.0f} ± {std:,.0f}"


def to_latex(df: pd.DataFrame, caption: str, label: str, bold_best: dict | None = None) -> str:
    cols = list(df.columns)
    spec = "l" * len(cols)
    lines = ["\\begin{table*}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\footnotesize", f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
             " & ".join(_tex(c) for c in cols) + " \\\\", "\\midrule"]
    prev = None
    for _, row in df.iterrows():
        first = row.iloc[0]
        if prev is not None and first != prev and str(first) != "":
            lines.append("\\midrule")
        prev = first if str(first) != "" else prev
        lines.append(" & ".join(_tex(str(v)) for v in row.values) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(lines) + "\n"


def _tex(s: str) -> str:
    s = str(s).replace("±", "$\\pm$").replace("%", "\\%").replace("_", "\\_").replace("&", "\\&")
    s = s.replace("**", "")
    return s


def save(df: pd.DataFrame, name: str, caption: str, cfg):
    out = p(cfg, "results") / "tables"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False, encoding="utf-8")
    (out / f"{name}.tex").write_text(to_latex(df, caption, f"tab:{name}"), encoding="utf-8")


# ------------------------------------------------------------------ tables
def summary_table(L: pd.DataFrame, methods, prim: str, ratios, names=None) -> pd.DataFrame:
    rows = []
    names = names or LABEL
    for (ds, H), g in L.groupby(["dataset", "H"], sort=True):
        for mth in methods:
            gm = g[(g["method"] == mth) & (g["setting"] == prim)]
            if gm.empty:
                continue
            row = {"Dataset": DLABEL.get(ds, ds), "H": H, "Method": names.get(mth, mth), "n": gm["seed"].nunique()}
            for k in METRICS:
                row[MLABEL[k]] = fmt(gm[k].mean(), gm[k].std(ddof=1) if len(gm) > 1 else 0.0)
            for r in ratios:
                gr = g[(g["method"] == mth) & (g["setting"] == f"r{r}")]
                row[f"Cost r={r}"] = fmt_cost(gr[f"cost_r{r}"].mean(), gr[f"cost_r{r}"].std(ddof=1) if len(gr) > 1 else 0)
            rows.append(row)
    df = pd.DataFrame(rows)
    # print dataset / H only on the first row of each block
    for col in ("Dataset", "H"):
        df[col] = df[col].astype(str)
    dup = df.duplicated(["Dataset", "H"])
    df.loc[dup, ["Dataset", "H"]] = ""
    return df


def wilcoxon_table(L: pd.DataFrame, prim: str, ratios, alpha: float) -> pd.DataFrame:
    rows = []
    metric_list = METRICS + [f"cost_r{r}" for r in ratios]
    for (ds, H), g in L.groupby(["dataset", "H"], sort=True):
        for k in metric_list:
            setting = prim if not k.startswith("cost") else f"r{k.split('_r')[1]}"
            gg = g[g["setting"] == setting].pivot_table(index="seed", columns="method", values=k)
            if "tcse" not in gg or len(gg) < 2:
                continue
            lower_better = k.startswith("cost")
            base = gg[[m for m in METHODS[:-1] if m in gg]]
            best = base.mean().idxmin() if lower_better else base.mean().idxmax()
            a, b = gg["tcse"].to_numpy(), gg[best].to_numpy()
            d = a - b
            if np.allclose(d, 0):
                pval, stat = 1.0, 0.0
            else:
                stat, pval = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            better = (a.mean() < b.mean()) if lower_better else (a.mean() > b.mean())
            rows.append({"Dataset": DLABEL.get(ds, ds), "H": H, "Metric": MLABEL.get(k, k.replace("cost_r", "Cost r=")),
                         "Best baseline": LABEL[best], "TCSE mean": round(float(a.mean()), 4),
                         "Baseline mean": round(float(b.mean()), 4), "n seeds": len(d),
                         "W": float(stat), "p-value": float(pval),
                         "Significant (5%)": "yes" if pval < alpha else "no",
                         "TCSE better": "yes" if better else "no"})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    log = get_logger("tcse", cfg)
    ratios = cfg["cost_ratios"]
    prim = f"r{cfg['evaluation']['primary_cost_ratio']}"
    made = []

    main_runs = load_family(cfg, "main")
    if not main_runs:
        log.error("no results in results/metrics/main - run scripts/run_all.py first")
        return 1
    L = long_rows(main_runs, ratios)

    # 1. main comparison (primary ratio) + the same table at every other ratio
    for r in ratios:
        name = "table1_main_comparison" if f"r{r}" == prim else f"table1_main_comparison_r{r}"
        t1 = summary_table(L, METHODS, f"r{r}", ratios)
        save(t1, name,
             f"Test-set comparison (mean $\\pm$ std over seeds). Metrics at the threshold chosen on validation "
             f"for $C_{{FN}}{{:}}C_{{FP}}={r}{{:}}1$; each cost column uses the threshold chosen for that ratio.", cfg)
        made.append(name)

    # 1b. drive-level recall (supplementary) and default-threshold baselines
    rows = []
    for (ds, H), g in L.groupby(["dataset", "H"]):
        for mth in METHODS:
            for setting in (prim, "default"):
                gm = g[(g["method"] == mth) & (g["setting"] == setting)]
                if len(gm):
                    rows.append({"Dataset": DLABEL[ds], "H": H, "Method": LABEL[mth],
                                 "Threshold": "θ* (r=%s)" % prim[1:] if setting != "default" else "0.5",
                                 "Drive recall": fmt(gm["drive_recall"].mean(), gm["drive_recall"].std(ddof=1)),
                                 "Failed test drives": f"{gm['failed_drives'].mean():.1f}",
                                 "Recall": fmt(gm["recall"].mean(), gm["recall"].std(ddof=1)),
                                 "F1": fmt(gm["f1"].mean(), gm["f1"].std(ddof=1)),
                                 "MCC": fmt(gm["mcc"].mean(), gm["mcc"].std(ddof=1))})
    save(pd.DataFrame(rows), "table1b_drive_level_and_default_threshold",
         "Supplementary: drive-level recall and record-level metrics at the cost-optimal and the default 0.5 threshold.", cfg)
    made.append("table1b_drive_level_and_default_threshold")

    # 2. Wilcoxon
    t2 = wilcoxon_table(L, prim, ratios, cfg["evaluation"]["alpha"])
    save(t2, "table2_wilcoxon", "Wilcoxon signed-rank test, TCSE vs. the best baseline per metric, across seeds.", cfg)
    made.append("table2_wilcoxon")

    # 3. ablations
    A = L[L["method"].isin(["tcse", "tcse_no_if"]) & (L["setting"] == prim)].copy()
    D = L[(L["method"] == "tcse") & (L["setting"] == "default")].copy()
    D["method"] = "tcse_default"
    Ds = []
    for r in ratios:                     # the 0.5 threshold does not depend on r
        d = D.copy()
        d["setting"] = f"r{r}"
        Ds.append(d)
    A = L[L["method"].isin(["tcse", "tcse_no_if"]) & L["setting"].isin([f"r{r}" for r in ratios])].copy()
    parts = [A] + Ds
    nt = load_family(cfg, "no_trend")
    if nt:
        N = long_rows(nt, ratios)
        N = N[N["method"] == "tcse"].copy()
        N["method"] = "tcse_no_trend"
        parts.append(N)
    LA = pd.concat(parts, ignore_index=True)
    t3 = summary_table(LA, ["tcse", "tcse_no_trend", "tcse_no_if", "tcse_default"], prim, ratios)
    save(t3, "table3_ablation", "Ablations of TCSE (mean $\\pm$ std over seeds). TCSE @0.5 uses the default "
         "threshold; its cost columns are at 0.5 as well.", cfg)
    made.append("table3_ablation")

    # 4. cross-dataset
    sh = load_family(cfg, "shared")
    if sh:
        W = long_rows(sh, ratios)                         # within-dataset, shared attributes
        X = long_rows(sh, ratios, key="cross")            # source-trained, tested on target
        rows = []
        for (src_ds, H), gx in X.groupby(["dataset", "H"]):
            tgt_ds = [d for d in DLABEL if d != src_ds][0]
            for mth in METHODS:
                cx = gx[(gx["method"] == mth) & (gx["setting"] == prim)]
                wi = W[(W["dataset"] == tgt_ds) & (W["H"] == H) & (W["method"] == mth) & (W["setting"] == prim)]
                if cx.empty or wi.empty:
                    continue
                row = {"Train → Test": f"{DLABEL[src_ds]} → {DLABEL[tgt_ds]}", "H": H, "Method": LABEL[mth]}
                for k in ("recall", "f1", "mcc"):
                    row[f"{MLABEL[k]} within"] = fmt(wi[k].mean(), wi[k].std(ddof=1))
                    row[f"{MLABEL[k]} cross"] = fmt(cx[k].mean(), cx[k].std(ddof=1))
                    row[f"Δ{MLABEL[k]}"] = f"{cx[k].mean() - wi[k].mean():+.3f}"
                rows.append(row)
        t4 = pd.DataFrame(rows)
        if len(t4):
            dup = t4.duplicated(["Train → Test", "H"])
            t4["H"] = t4["H"].astype(str)
            t4.loc[dup, ["Train → Test", "H"]] = ""
        save(t4, "table4_cross_dataset", "Cross-dataset generalisation on shared SMART attributes. "
             "'within' = trained and tested on the target dataset; Δ = cross − within.", cfg)
        made.append("table4_cross_dataset")

    # 5. temporal
    tp = load_family(cfg, "temporal")
    if tp:
        t5 = summary_table(long_rows(tp, ratios), METHODS, prim, ratios)
        save(t5, "table5_temporal_holdout", "Temporal hold-out (train on the early period, test on the later "
             "period with an H-day gap).", cfg)
        made.append("table5_temporal_holdout")

    # 6. dataset statistics
    rows = []
    for sub in sorted({r["spec"]["subset"] for r in main_runs}):
        info = json.loads((p(cfg, "processed") / sub / "store.json").read_text())
        ds = info["subset"]["dataset"]
        summ = json.loads((p(cfg, "interim") / f"{ds}_summary.json").read_text())
        for model, mstat in summ["models"].items():
            rows.append({"Dataset": DLABEL[ds], "Model": model, "H": "all", "Drives": mstat["drives"],
                         "Failed drives": mstat["failures"], "Drive-days": mstat["drive_days"],
                         "Positive rows": "", "Final features": ""})
        for H in cfg["horizons"]:
            rs = [r for r in main_runs if r["spec"]["subset"] == sub and r["spec"]["horizon"] == H]
            if not rs:
                continue
            nfeat = [len(r["features"]["selected"]) for r in rs]
            rows_used = [sum(c["rows"] for c in r["counts"].values()) for r in rs]
            pos = [sum(c["positives"] for c in r["counts"].values()) for r in rs]
            rows.append({"Dataset": DLABEL[ds], "Model": "pooled" if len(info["models"]) > 1 else info["models"][0],
                         "H": H, "Drives": info["n_drives"], "Failed drives": info["n_failed_drives"],
                         "Drive-days": int(np.mean(rows_used)), "Positive rows": int(np.mean(pos)),
                         "Final features": f"{np.mean(nfeat):.1f} ({min(nfeat)}–{max(nfeat)})"})
    save(pd.DataFrame(rows), "table6_dataset_statistics",
         "Dataset statistics after preprocessing (drive-days and positives after labelling and censoring, "
         "mean over seeds; final feature count mean (min–max) over seeds).", cfg)
    made.append("table6_dataset_statistics")

    # SVM approximation check
    ex = load_family(cfg, "svm_exact")
    if ex:
        rows = []
        for r in ex:
            s = r["spec"]
            mref = json.loads((p(cfg, "results") / "metrics" / "main" / s["subset"] /
                               f"H{s['horizon']}_m{s['window']}_s{s['seed']}.json").read_text())
            for name, rr in (("Nyström RBF-SVM (used)", mref), ("Exact RBF-SVC (libsvm)", r)):
                for mth in ("svm", "tcse"):
                    m = rr["results"][mth][prim]
                    rows.append({"SVM": name, "Method": LABEL[mth], **{MLABEL[k]: round(m[k], 4) for k in METRICS},
                                 f"Cost r={prim[1:]}": m[f"cost_{prim}"],
                                 "Level-0 fit+predict s": rr["timings"]["tune_and_oof"] + rr["timings"]["refit"]
                                 + rr["timings"]["predict_val_test"]})
        save(pd.DataFrame(rows), "table7_svm_approximation_check",
             "Exact libsvm RBF-SVC vs. the Nyström-approximated RBF-SVM (same C, gamma), one configuration.", cfg)
        made.append("table7_svm_approximation_check")

    log.info("tables written: %s", made)
    if not a.no_figures:
        from tcse import plots
        plots.make_all(cfg, L, main_runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
