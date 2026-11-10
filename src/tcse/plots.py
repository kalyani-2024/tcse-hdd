"""Result figures (PNG + PDF). Every value is read from saved run JSONs."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tcse.utils import get_logger, p

log = get_logger()

# reference categorical palette (validated adjacent order), light print surface
SURFACE = "#fcfcfb"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"tcse": "#2a78d6", "dt": "#eb6834", "knn": "#1baf7a", "nb": "#eda100", "svm": "#e87ba4",
          "tcse_no_if": "#6250d6", "tcse_no_trend": "#008300", "tcse_default": "#e34948"}
MARKERS = {"tcse": "o", "dt": "s", "knn": "^", "nb": "D", "svm": "v"}
LABEL = {"dt": "DT", "knn": "KNN", "nb": "NB", "svm": "SVM", "tcse": "TCSE",
         "tcse_no_if": "TCSE w/o IF"}
DLABEL = {"backblaze": "Backblaze", "smartz": "SMART-Z"}
METHODS = ["dt", "knn", "nb", "svm", "tcse"]


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
        "axes.titlesize": 10, "legend.frameon": False, "lines.linewidth": 2,
    })


def _save(fig, name, cfg):
    out = p(cfg, "results") / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def metric_vs_horizon(cfg, L: pd.DataFrame, prim: str):
    datasets = [d for d in DLABEL if d in set(L["dataset"])]
    panels = [("recall", "Recall"), ("f1", "F1-score"), (f"cost_{prim}", f"Total cost (r = {prim[1:]})")]
    fig, axes = plt.subplots(len(datasets), 3, figsize=(10, 3.0 * len(datasets)), squeeze=False)
    for i, ds in enumerate(datasets):
        g = L[(L["dataset"] == ds) & (L["setting"] == prim)]
        for j, (k, lab) in enumerate(panels):
            ax = axes[i, j]
            for mth in METHODS:
                s = g[g["method"] == mth].groupby("H")[k].agg(["mean", "std"]).reset_index()
                if s.empty:
                    continue
                ax.errorbar(s["H"], s["mean"], yerr=s["std"].fillna(0), color=COLORS[mth], marker=MARKERS[mth],
                            markersize=5, capsize=3, linewidth=2.6 if mth == "tcse" else 1.6,
                            label=LABEL[mth], zorder=3 if mth == "tcse" else 2)
            ax.set_xticks(cfg["horizons"])
            ax.set_xlabel("Prediction horizon H (days)")
            ax.set_ylabel(lab)
            ax.set_title(f"{DLABEL[ds]}: {lab}")
            if k.startswith("cost"):
                ax.set_yscale("log")
    axes[0, 0].legend(ncol=5, loc="lower left", bbox_to_anchor=(0, 1.12))
    fig.tight_layout()
    _save(fig, "fig_metrics_vs_horizon", cfg)


def roc_curves(cfg, runs):
    H = cfg["evaluation"]["roc_horizon"]
    seed0 = cfg["project"]["seeds"][0]
    rs = [r for r in runs if r["spec"]["horizon"] == H and r["spec"]["seed"] == seed0 and r.get("roc")]
    if not rs:
        return
    fig, axes = plt.subplots(1, len(rs), figsize=(4.2 * len(rs), 3.8), squeeze=False)
    for ax, r in zip(axes[0], rs):
        for mth in METHODS:
            c = r["roc"][mth]
            auc = r["results"][mth]["r10"]["roc_auc"]
            ax.plot(c["fpr"], c["tpr"], color=COLORS[mth], linewidth=2.6 if mth == "tcse" else 1.5,
                    label=f"{LABEL[mth]} (AUC {auc:.3f})")
        ax.plot([0, 1], [0, 1], color=INK2, linewidth=0.8, linestyle="--")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"{DLABEL[r['spec']['dataset']]}, H = {H} (seed {seed0})")
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_roc_H7", cfg)


def confusion_matrices(cfg, runs, prim):
    seed0 = cfg["project"]["seeds"][0]
    datasets = [d for d in DLABEL if any(r["spec"]["dataset"] == d for r in runs)]
    fig, axes = plt.subplots(len(datasets), len(cfg["horizons"]), figsize=(2.6 * len(cfg["horizons"]), 2.6 * len(datasets)),
                             squeeze=False)
    for i, ds in enumerate(datasets):
        for j, H in enumerate(cfg["horizons"]):
            ax = axes[i, j]
            ax.grid(False)
            rs = [r for r in runs if r["spec"]["dataset"] == ds and r["spec"]["horizon"] == H and r["spec"]["seed"] == seed0]
            if not rs:
                ax.axis("off")
                continue
            m = rs[0]["results"]["tcse"][prim]
            cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], dtype=float)
            norm = cm / cm.sum(axis=1, keepdims=True)
            ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, f"{int(cm[a, b]):,}\n({norm[a, b]:.1%})", ha="center", va="center", fontsize=7.5,
                            color="white" if norm[a, b] > 0.6 else INK)
            ax.set_xticks([0, 1], ["healthy", "failure"])
            ax.set_yticks([0, 1], ["healthy", "failure"])
            ax.set_xlabel("Predicted")
            if j == 0:
                ax.set_ylabel(f"{DLABEL[ds]}\nActual")
            ax.set_title(f"H = {H}")
    fig.suptitle(f"TCSE confusion matrices on the test set (θ* for r = {prim[1:]}, seed {seed0}; row-normalised)",
                 fontsize=10)
    fig.tight_layout()
    _save(fig, "fig_confusion_tcse", cfg)


def meta_coefficients(cfg, runs):
    rows = []
    for r in runs:
        for k, v in r["meta"]["tcse"]["coefficients"].items():
            if k != "intercept":
                rows.append({"dataset": r["spec"]["dataset"], "H": r["spec"]["horizon"], "feature": k, "coef": v})
    C = pd.DataFrame(rows)
    if C.empty:
        return
    names = {"p_dt": "p_DT", "p_knn": "p_KNN", "p_nb": "p_NB", "p_svm": "p_SVM", "s_iforest": "IF score s(x,n)",
             "n_delta_pos": "n_Δ+"}
    datasets = [d for d in DLABEL if d in set(C["dataset"])]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 3.4), squeeze=False, sharey=True)
    feats = list(names)
    hcolors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    for ax, ds in zip(axes[0], datasets):
        g = C[C["dataset"] == ds]
        Hs = sorted(g["H"].unique())
        w = 0.8 / len(Hs)
        for k, H in enumerate(Hs):
            s = g[g["H"] == H].groupby("feature")["coef"].agg(["mean", "std"]).reindex(feats)
            x = np.arange(len(feats)) + (k - (len(Hs) - 1) / 2) * w
            ax.bar(x, s["mean"], width=w * 0.9, yerr=s["std"].fillna(0), color=hcolors[k % 4], capsize=2,
                   label=f"H = {H}", error_kw={"elinewidth": 0.8, "ecolor": INK2})
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_xticks(np.arange(len(feats)), [names[f] for f in feats], rotation=30, ha="right")
        ax.set_title(f"{DLABEL[ds]}: meta-learner coefficients (standardised z)")
        ax.set_ylabel("Coefficient (mean ± std over seeds)")
    axes[0, 0].legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.1))
    fig.tight_layout()
    _save(fig, "fig_meta_coefficients", cfg)


def cost_vs_threshold(cfg, runs):
    H = cfg["evaluation"]["roc_horizon"]
    seed0 = cfg["project"]["seeds"][0]
    rs = [r for r in runs if r["spec"]["horizon"] == H and r["spec"]["seed"] == seed0]
    if not rs:
        return
    rcol = {5: "#2a78d6", 10: "#eb6834", 50: "#1baf7a"}
    fig, axes = plt.subplots(1, len(rs), figsize=(4.6 * len(rs), 3.6), squeeze=False)
    for ax, r in zip(axes[0], rs):
        c = r["cost_curves"]["tcse"]
        th = np.array(c["thresholds"])
        for rr in cfg["cost_ratios"]:
            v = np.array(c["val"][f"r{rr}"])
            t = np.array(c["test"][f"r{rr}"])
            ax.plot(th, t, color=rcol.get(rr, INK), linewidth=2, label=f"r = {rr} (test)")
            ax.plot(th, v, color=rcol.get(rr, INK), linewidth=1.2, linestyle="--", label=f"r = {rr} (validation)")
            star = r["results"]["tcse"][f"r{rr}"]["threshold"]
            ax.axvline(star, color=rcol.get(rr, INK), linewidth=0.8, linestyle=":")
        ax.set_yscale("log")
        ax.set_xlabel("Decision threshold θ")
        ax.set_ylabel("Total cost C(θ) = r·FN + FP")
        ax.set_title(f"{DLABEL[r['spec']['dataset']]}: TCSE, H = {H} (dotted: θ*)")
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    _save(fig, "fig_cost_vs_threshold", cfg)


def make_all(cfg, L, runs):
    style()
    prim = f"r{cfg['evaluation']['primary_cost_ratio']}"
    metric_vs_horizon(cfg, L, prim)
    roc_curves(cfg, runs)
    confusion_matrices(cfg, runs, prim)
    meta_coefficients(cfg, runs)
    cost_vs_threshold(cfg, runs)
    log.info("figures written to %s", p(cfg, "results") / "figures")
