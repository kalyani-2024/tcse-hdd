"""Metrics, cost and cost-optimal threshold (report Sections IV-F3 and IV-G)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def thresholds(cfg: dict) -> np.ndarray:
    """Report grid 0.01..0.99 (step 0.01), plus an optional log-spaced tail
    1 - 10^-k above 0.99 (models trained on rebalanced data give inflated
    probabilities, so the cost optimum can lie above 0.99)."""
    t = cfg["threshold"]
    n = int(round((t["stop"] - t["start"]) / t["step"])) + 1
    grid = np.round(t["start"] + t["step"] * np.arange(n), 10)
    tail = t.get("high_tail")
    if tail:
        k = np.linspace(tail["from_exp"], tail["to_exp"], tail["points"])
        grid = np.unique(np.r_[grid, 1.0 - 10.0 ** (-k)])
    return grid


def confusion_at(y: np.ndarray, p: np.ndarray, theta: float, w: np.ndarray | None = None) -> dict:
    """Confusion counts with prediction = (p >= theta); optional row weights."""
    y = np.asarray(y).astype(bool)
    pred = np.asarray(p) >= theta
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64)
    tp = float(w[y & pred].sum()); fp = float(w[~y & pred].sum())
    fn = float(w[y & ~pred].sum()); tn = float(w[~y & ~pred].sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def confusion_curve(y: np.ndarray, p: np.ndarray, grid: np.ndarray) -> dict:
    """TP/FP/FN/TN at every threshold of the grid in one pass (vectorised)."""
    y = np.asarray(y).astype(bool)
    p = np.asarray(p, dtype=np.float64)
    pos, neg = np.sort(p[y]), np.sort(p[~y])
    tp = len(pos) - np.searchsorted(pos, grid, side="left")
    fp = len(neg) - np.searchsorted(neg, grid, side="left")
    return {"tp": tp, "fp": fp, "fn": len(pos) - tp, "tn": len(neg) - fp}


def cost(fn, fp, c_fn: float, c_fp: float = 1.0):
    """Eq. 10: C = C_FN * FN + C_FP * FP."""
    return c_fn * np.asarray(fn) + c_fp * np.asarray(fp)


def best_threshold(y: np.ndarray, p: np.ndarray, grid: np.ndarray, ratio: float, c_fp: float = 1.0):
    """Eq. 11: theta* = argmin over the grid of validation cost (first minimum on ties)."""
    cc = confusion_curve(y, p, grid)
    c = cost(cc["fn"], cc["fp"], ratio * c_fp, c_fp)
    j = int(np.argmin(c))
    return float(grid[j]), c


def metrics_from_counts(tp, fp, fn, tn) -> dict:
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    d = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (tp * tn - fp * fn) / np.sqrt(d) if d > 0 else 0.0
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1), "mcc": float(mcc)}


def weighted_f1(y, p, theta, w=None) -> float:
    c = confusion_at(y, p, theta, w)
    return metrics_from_counts(c["tp"], c["fp"], c["fn"], c["tn"])["f1"]


def evaluate(y: np.ndarray, p: np.ndarray, theta: float, ratios, c_fp: float = 1.0) -> dict:
    """Eqs. 12-15 plus total cost at every ratio, at threshold theta."""
    c = confusion_at(y, p, theta)
    out = {k: int(v) for k, v in c.items()}
    out.update(metrics_from_counts(c["tp"], c["fp"], c["fn"], c["tn"]))
    y = np.asarray(y)
    out["roc_auc"] = float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")
    out["threshold"] = float(theta)
    for r in ratios:
        out[f"cost_r{r}"] = float(cost(c["fn"], c["fp"], r * c_fp, c_fp))
    return out


def drive_level_recall(y: np.ndarray, p: np.ndarray, theta: float, serial: np.ndarray) -> dict:
    """A failed drive counts as caught if any of its positive (within-H) records is flagged."""
    pos = np.asarray(y) == 1
    if not pos.any():
        return {"failed_drives": 0, "caught_drives": 0, "drive_recall": float("nan")}
    sid = np.asarray(serial)[pos]
    hit = np.asarray(p)[pos] >= theta
    drives = np.unique(sid)
    caught = np.unique(sid[hit])
    return {"failed_drives": int(len(drives)), "caught_drives": int(len(caught)),
            "drive_recall": float(len(caught) / len(drives))}


def roc_points(y, p, max_points: int = 400) -> dict:
    fpr, tpr, _ = roc_curve(y, p)
    if len(fpr) > max_points:
        idx = np.unique(np.linspace(0, len(fpr) - 1, max_points).astype(int))
        fpr, tpr = fpr[idx], tpr[idx]
    return {"fpr": fpr.round(6).tolist(), "tpr": tpr.round(6).tolist()}
