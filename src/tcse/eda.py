"""Exploratory data analysis (report Section IV-D), on the TRAINING split only.

All figures/tables use the seed-0 training drives, so no design choice is
informed by validation or test data.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA

from tcse.data.label import NO_FAIL, label_rows
from tcse.features.trends import STATS, FeatureStore, trend_name
from tcse.pipeline import day_of
from tcse.plots import COLORS, INK, INK2, SURFACE, style
from tcse.split import get_split
from tcse.utils import p, save_json, smart_col

ATTR_NAME = {5: "Reallocated sectors", 9: "Power-on hours", 187: "Reported uncorrectable",
             188: "Command timeout", 193: "Load/unload cycles", 194: "Temperature",
             197: "Pending sectors", 198: "Offline uncorrectable", 199: "UDMA CRC errors"}
DLABEL = {"backblaze": "Backblaze", "smartz": "SMART-Z"}


class EDA:
    def __init__(self, cfg, dataset: str, subset: str | None = None, seed: int = 0):
        self.cfg, self.dataset = cfg, dataset
        self.store = FeatureStore(cfg, subset or dataset)
        m = self.store.meta
        self.sid, self.day, self.fail = m["serial_id"].to_numpy(), m["day"].to_numpy(), m["fail"].to_numpy()
        assign = get_split(self.store, seed, (cfg["split"]["train"], cfg["split"]["val"], cfg["split"]["test"]))
        lab = assign.reindex(np.arange(int(self.sid.max()) + 1)).to_numpy()
        self.train = lab[self.sid] == "train"
        self.end = day_of(cfg[dataset]["period_end"])
        self.rng = np.random.default_rng(seed)
        self.attrs = [a for a in cfg["attributes"]
                      if np.mean(self.store.column(f"miss_{a}")[self.train]) <= cfg["cleaning"]["max_missing_frac"]]
        self.figdir = p(cfg, "results") / "figures"
        self.tabdir = p(cfg, "results") / "tables"
        self.figdir.mkdir(parents=True, exist_ok=True)
        self.tabdir.mkdir(parents=True, exist_ok=True)
        style()

    def _save(self, fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(self.figdir / f"eda_{self.dataset}_{name}.{ext}", dpi=170, bbox_inches="tight")
        plt.close(fig)

    # 1. summary statistics and class ratio per horizon
    def class_ratio(self) -> pd.DataFrame:
        rows = []
        for H in self.cfg["horizons"]:
            keep, y = label_rows(self.day, self.fail, H, self.end)
            k = keep & self.train
            npos = int(y[k].sum())
            rows.append({"H": H, "train_rows": int(k.sum()), "positives": npos,
                         "positive_pct": round(100 * npos / k.sum(), 4), "healthy_per_failure": round((k.sum() - npos) / npos, 1),
                         "failed_drives_with_positives": int(len(np.unique(self.sid[k][y[k] == 1])))})
        df = pd.DataFrame(rows)
        df.to_csv(self.tabdir / f"eda_{self.dataset}_class_ratio.csv", index=False)
        return df

    def summary_stats(self) -> pd.DataFrame:
        rows = []
        tr = np.flatnonzero(self.train)
        for a in self.attrs:
            x = self.store.column(smart_col(a), tr)
            rows.append({"attribute": a, "name": ATTR_NAME.get(a, ""), "missing_pct": round(100 * np.isnan(x).mean(), 3),
                         "mean": np.nanmean(x), "std": np.nanstd(x), "median": np.nanmedian(x),
                         "p99": np.nanquantile(x, 0.99), "max": np.nanmax(x), "nonzero_pct": round(100 * np.nanmean(x > 0), 3)})
        df = pd.DataFrame(rows).round(3)
        df.to_csv(self.tabdir / f"eda_{self.dataset}_summary_stats.csv", index=False)
        return df

    # 2. missingness heat map (raw, before forward fill) by attribute and month
    def missingness(self):
        tr = np.flatnonzero(self.train)
        month = pd.to_datetime(self.day[tr], unit="D").to_period("M").astype(str)
        M = pd.DataFrame({f"{a} {ATTR_NAME.get(a, '')}": self.store.column(f"miss_{a}", tr) for a in self.cfg["attributes"]})
        M["month"] = month
        H = M.groupby("month").mean().T * 100
        fig, ax = plt.subplots(figsize=(1.0 + 0.7 * H.shape[1], 4))
        im = ax.imshow(H.to_numpy(), cmap="Blues", vmin=0, vmax=100, aspect="auto")
        ax.grid(False)
        ax.set_xticks(range(H.shape[1]), H.columns, rotation=45, ha="right")
        ax.set_yticks(range(H.shape[0]), H.index)
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                v = H.iat[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white" if v > 60 else INK)
        fig.colorbar(im, ax=ax, label="% records missing (raw)")
        ax.set_title(f"{DLABEL[self.dataset]}: missing SMART values by month (training drives)")
        self._save(fig, "missingness")
        H.round(2).to_csv(self.tabdir / f"eda_{self.dataset}_missingness.csv")
        return H

    # 3. box plots + KS tests, failed vs healthy drives
    def failed_vs_healthy(self):
        tr = self.train
        failed = tr & (self.fail != NO_FAIL) & (self.day < self.fail)
        # last pre-failure record of each failed training drive
        f_rows = pd.Series(np.flatnonzero(failed)).groupby(self.sid[failed]).max().to_numpy()
        healthy = np.flatnonzero(tr & (self.fail == NO_FAIL))
        # one random record per healthy training drive
        h_rows = pd.Series(healthy).groupby(self.sid[healthy]).agg(lambda s: s.iloc[self.rng.integers(len(s))]).to_numpy()
        rows, data = [], []
        for a in self.attrs:
            xf = self.store.column(smart_col(a), f_rows)
            xh = self.store.column(smart_col(a), h_rows)
            xf, xh = xf[~np.isnan(xf)], xh[~np.isnan(xh)]
            ks = ks_2samp(xf, xh)
            rows.append({"attribute": a, "name": ATTR_NAME.get(a, ""), "n_failed": len(xf), "n_healthy": len(xh),
                         "median_failed": float(np.median(xf)), "median_healthy": float(np.median(xh)),
                         "mean_failed": float(xf.mean()), "mean_healthy": float(xh.mean()),
                         "KS_D": float(ks.statistic), "KS_p": float(ks.pvalue)})
            data.append((a, np.log1p(xh), np.log1p(xf)))
        df = pd.DataFrame(rows).sort_values("KS_D", ascending=False)
        df.to_csv(self.tabdir / f"eda_{self.dataset}_ks_tests.csv", index=False)
        n = len(data)
        fig, axes = plt.subplots(1, n, figsize=(1.7 * n, 3.4))
        for ax, (a, h, f) in zip(np.atleast_1d(axes), data):
            bp = ax.boxplot([h, f], widths=0.6, patch_artist=True, showfliers=True,
                            flierprops={"markersize": 2, "markeredgecolor": INK2},
                            medianprops={"color": INK})
            for patch, c in zip(bp["boxes"], [COLORS["knn"], COLORS["dt"]]):
                patch.set_facecolor(c)
                patch.set_alpha(0.75)
            ax.set_xticks([1, 2], ["healthy", "failed"])
            d = df.set_index("attribute").loc[a, "KS_D"]
            ax.set_title(f"SMART {a}\nKS D = {d:.2f}", fontsize=8)
        np.atleast_1d(axes)[0].set_ylabel("ln(1 + raw value)")
        fig.suptitle(f"{DLABEL[self.dataset]}: last pre-failure record of failed drives vs. a random record "
                     f"of healthy drives (training split)", fontsize=9)
        fig.tight_layout()
        self._save(fig, "boxplots_ks")
        return df

    # 4. correlation heat map of the candidate features (window = default)
    def correlation(self, m: int | None = None, n_rows: int = 300_000):
        m = m or self.cfg["default_window"]
        tr = np.flatnonzero(self.train)
        rows = np.sort(self.rng.choice(tr, min(n_rows, len(tr)), replace=False))
        cols = [smart_col(a) for a in self.attrs] + [trend_name(a, s, m) for a in self.attrs for s in STATS]
        X = pd.DataFrame({c: self.store.column(c, rows) for c in cols})
        X = X.loc[:, X.std() > 0]
        X = np.sign(X) * np.log1p(np.abs(X.fillna(X.median())))
        R = X.corr()
        fig, ax = plt.subplots(figsize=(0.32 * len(R) + 2, 0.32 * len(R) + 1.5))
        im = ax.imshow(R.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.grid(False)
        short = [c.replace("smart_", "").replace("_raw", "") for c in R.columns]
        ax.set_xticks(range(len(R)), short, rotation=90, fontsize=6)
        ax.set_yticks(range(len(R)), short, fontsize=6)
        fig.colorbar(im, ax=ax, shrink=0.6, label="Pearson r (log-transformed)")
        ax.set_title(f"{DLABEL[self.dataset]}: feature correlations (m = {m}, training rows)")
        self._save(fig, "correlation")
        high = [(a, b, round(R.loc[a, b], 3)) for i, a in enumerate(R.columns) for b in R.columns[i + 1:]
                if abs(R.loc[a, b]) > self.cfg["feature_selection"]["corr_threshold"]]
        return R, high

    # 5. average 30-day pre-failure trajectories
    def prefailure_trajectories(self, days: int = 30):
        tr = self.train & (self.fail != NO_FAIL) & (self.day < self.fail)
        idx = np.flatnonzero(tr)
        dtf = self.fail[idx] - self.day[idx]            # days to failure, >= 1
        sel = dtf <= days
        idx, dtf = idx[sel], dtf[sel]
        healthy = np.flatnonzero(self.train & (self.fail == NO_FAIL))
        hs = self.rng.choice(healthy, min(200_000, len(healthy)), replace=False)
        attrs = [a for a in self.attrs if a != self.cfg["age_attribute"]]
        n = len(attrs)
        cols = min(4, n)
        rows_ = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows_, cols, figsize=(3.2 * cols, 2.5 * rows_), squeeze=False)
        out = {}
        for ax, a in zip(axes.ravel(), attrs):
            x = self.store.column(smart_col(a), idx)
            s = pd.DataFrame({"d": dtf, "x": x}).groupby("d")["x"].agg(["mean", "sem"])
            hm = float(np.nanmean(self.store.column(smart_col(a), hs)))
            ax.plot(-s.index, s["mean"], color=COLORS["tcse"], linewidth=2, label="failed drives")
            ax.fill_between(-s.index, s["mean"] - s["sem"], s["mean"] + s["sem"], color=COLORS["tcse"], alpha=0.15,
                            linewidth=0)
            ax.axhline(hm, color=COLORS["dt"], linewidth=1.4, linestyle="--", label="healthy mean")
            ax.set_title(f"SMART {a}: {ATTR_NAME.get(a, '')}", fontsize=8)
            ax.set_xlabel("days before failure")
            out[a] = s["mean"].round(4).to_dict()
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"{DLABEL[self.dataset]}: mean raw value over the {days} days before failure "
                     f"(training failed drives; band = ±1 s.e.)", fontsize=9)
        fig.tight_layout()
        self._save(fig, "prefailure_trajectories")
        return out

    # 6. 2-D PCA projection
    def pca(self, H: int = 7, m: int | None = None, n_neg: int = 20_000):
        m = m or self.cfg["default_window"]
        keep, y = label_rows(self.day, self.fail, H, self.end)
        k = keep & self.train
        pos = np.flatnonzero(k & (y == 1))
        neg = self.rng.choice(np.flatnonzero(k & (y == 0)), n_neg, replace=False)
        rows = np.sort(np.concatenate([pos, neg]))
        cols = [smart_col(a) for a in self.attrs] + [trend_name(a, s, m) for a in self.attrs for s in STATS]
        X = pd.DataFrame({c: self.store.column(c, rows) for c in cols})
        X = X.fillna(X.median())
        X = X.loc[:, X.std() > 0]
        X = np.sign(X) * np.log1p(np.abs(X))
        X = (X - X.mean()) / X.std()
        pc = PCA(2, random_state=0)
        Z = pc.fit_transform(X.to_numpy())
        yy = y[rows]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(Z[yy == 0, 0], Z[yy == 0, 1], s=4, color=COLORS["knn"], alpha=0.35, label=f"healthy (sample of {n_neg:,})",
                   linewidths=0)
        ax.scatter(Z[yy == 1, 0], Z[yy == 1, 1], s=10, color=COLORS["dt"], alpha=0.8, label=f"fails within {H} d",
                   edgecolors=SURFACE, linewidths=0.4)
        ev = pc.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({ev[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({ev[1]:.1%} variance)")
        ax.set_title(f"{DLABEL[self.dataset]}: PCA of training records, H = {H}, m = {m}")
        ax.legend(fontsize=7, markerscale=2)
        self._save(fig, "pca")
        return ev

    # 7. window comparison: separation of trend features for each m
    def window_separation(self, H: int = 7):
        keep, y = label_rows(self.day, self.fail, H, self.end)
        k = keep & self.train
        pos = np.flatnonzero(k & (y == 1))
        neg = self.rng.choice(np.flatnonzero(k & (y == 0)), min(100_000, int((k & (y == 0)).sum())), replace=False)
        rows = []
        for m in self.cfg["windows"]:
            for a in self.attrs:
                for s in ("delta", "slope", "std"):
                    c = trend_name(a, s, m)
                    xp, xn = self.store.column(c, pos), self.store.column(c, neg)
                    xp, xn = xp[~np.isnan(xp)], xn[~np.isnan(xn)]
                    if len(xp) > 5 and len(xn) > 5:
                        rows.append({"m": m, "attribute": a, "stat": s, "KS_D": ks_2samp(xp, xn).statistic})
        df = pd.DataFrame(rows)
        piv = df.pivot_table(index=["attribute", "stat"], columns="m", values="KS_D").round(3)
        piv.to_csv(self.tabdir / f"eda_{self.dataset}_window_ks.csv")
        summ = df.groupby("m")["KS_D"].agg(["mean", "median", "max"]).round(4)
        save_json(summ.to_dict(), self.tabdir / f"eda_{self.dataset}_window_ks_summary.json")
        return piv, summ
