"""Experiment runner: one (subset, horizon, seed, window, variant, split) run.

Steps (report Fig. 1):
  label + censor -> drive-level split -> train-only preprocessing ->
  5-fold stratified group CV on the training split: grid search for every
  baseline (resampling inside each fold) whose held-out predictions are also
  the out-of-fold level-0 meta-features -> Isolation Forest on healthy drives
  of each fold -> class-weighted logistic-regression meta-learner on the OOF
  meta-features -> refit level 0 on the full training split -> validation /
  test predictions -> cost-optimal thresholds on validation -> test metrics.

Results are cached as JSON under results/metrics/<family>/<subset>/, so an
interrupted grid resumes where it stopped.
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from tcse.data.label import label_rows, to_days
from tcse.evaluate import (best_threshold, confusion_curve, cost, drive_level_recall, evaluate,
                           roc_points, thresholds, weighted_f1)
from tcse.features.transform import StoreSource, TCSEPreprocessor
from tcse.features.trends import STATS, FeatureStore, trend_name
from joblib import Parallel, delayed

from tcse.models.baselines import BASELINES, knn_multi_k_proba, make_model, param_grid
from tcse.models.tcse import (anomaly_score, fit_meta, make_isolation_forest, meta_coefficients,
                              meta_columns)
from tcse.sampling import resampled_counts, undersample_rows
from tcse.split import get_split, group_folds, rows_by_split, temporal_rows
from tcse.utils import get_logger, load_json, log_event, p, save_json, seed_everything, smart_col

log = get_logger()


@dataclass
class RunSpec:
    family: str                 # main | no_trend | temporal | shared | window_selection | svm_exact
    subset: str                 # feature-store name
    dataset: str                # backblaze | smartz
    horizon: int
    seed: int
    window: int
    variant: str = "full"       # full | no_trend
    split: str = "random"       # random | temporal
    attrs: tuple | None = None  # restrict attributes (cross-dataset: shared attributes)
    cross_target: dict | None = None   # {"subset":..., "dataset":...}
    svm_override: dict | None = None   # replace SVM params (exact-SVC check)
    extra: dict = field(default_factory=dict)

    @property
    def tag(self) -> str:
        return f"H{self.horizon}_m{self.window}_s{self.seed}"


def result_path(cfg: dict, spec: RunSpec) -> Path:
    return p(cfg, "results") / "metrics" / spec.family / spec.subset / f"{spec.tag}.json"


def day_of(s: str) -> int:
    return int(to_days([s])[0])


def temporal_bounds(cfg: dict, dataset: str) -> tuple[int, int, int]:
    t = cfg["temporal"][dataset]
    if dataset == "backblaze":
        return day_of(t["train_end"]), day_of(t["test_end"]), day_of(t["label_end"])
    start = pd.Timestamp(cfg[dataset]["period_start"])
    train_end = (start.to_period("M") + t["train_months"] - 1).to_timestamp(how="end").normalize()
    end = day_of(cfg[dataset]["period_end"])
    return day_of(str(train_end.date())), end, end


def candidate_columns(cfg: dict, spec: RunSpec, extra: dict) -> tuple[list, list]:
    attrs = list(spec.attrs or cfg["attributes"])
    cols = [smart_col(a) for a in attrs] + [f"miss_{a}" for a in attrs]
    if cfg["include_normalized"]:
        cols += [smart_col(a, "normalized") for a in attrs]
    if spec.variant == "full":
        cols += [trend_name(a, s, spec.window) for a in attrs for s in STATS]
        cols.append("n_delta_pos")
    cols += [c for c in extra if c.startswith("model_")]
    flags = [f"miss_{a}" for a in attrs] + [c for c in extra if c.startswith("model_")]
    return cols, flags


def n_delta_pos(cfg: dict, store: FeatureStore, attrs: list, m: int) -> np.ndarray:
    """Number of critical attributes with a positive m-day delta (from the feature store)."""
    out = np.zeros(store.X.shape[0], dtype=np.float32)
    for a in cfg["critical_attributes"]:
        if a in attrs:
            d = store.column(trend_name(a, "delta", m))
            with np.errstate(invalid="ignore"):
                out += np.nan_to_num(d, nan=0.0) > 0
    return out


class Gather:
    """rows -> transformed design matrix (reads only the selected columns)."""

    def __init__(self, store, extra, pre: TCSEPreprocessor):
        self.src = StoreSource(store, None, extra)
        self.pre = pre

    def __call__(self, rows: np.ndarray) -> np.ndarray:
        return self.pre.transform_selected(self.src.block(self.pre.selected_, rows))


def predict_level0(models: dict, iforest, gather: Gather, rows: np.ndarray, batch: int,
                   desc: str = "") -> np.ndarray:
    """Columns: p_dt, p_knn, p_nb, p_svm, s_iforest (batched)."""
    out = np.empty((len(rows), len(BASELINES) + 1), dtype=np.float32)
    tt = {n: 0.0 for n in ("gather",) + tuple(BASELINES) + ("iforest",)}
    for i in range(0, len(rows), batch):
        t0 = time.perf_counter()
        X = gather(rows[i:i + batch])
        tt["gather"] += time.perf_counter() - t0
        for j, name in enumerate(BASELINES):
            t0 = time.perf_counter()
            out[i:i + len(X), j] = models[name].predict_proba(X)[:, 1]
            tt[name] += time.perf_counter() - t0
        t0 = time.perf_counter()
        out[i:i + len(X), -1] = anomaly_score(iforest, X)
        tt["iforest"] += time.perf_counter() - t0
    PREDICT_TIMES[desc] = {k: round(v, 1) for k, v in tt.items()}
    return out


PREDICT_TIMES: dict = {}


def _grid_fit_predict(name, grid, cfg, seed, Xf, yf, Xh):
    """Fit every grid point on the fold's training data and score the held-out rows.
    KNN: one neighbour query serves all k. SVM / DT: grid points in parallel threads
    (liblinear and the tree builder release the GIL)."""
    if name == "knn" and len(grid) > 1 and all(set(g) == {"n_neighbors"} for g in grid):
        t1 = time.perf_counter()
        ks = [g["n_neighbors"] for g in grid]
        mdl = make_model("knn", cfg, seed, {"n_neighbors": max(ks)}).fit(Xf, yf)
        t2 = time.perf_counter()
        probs = knn_multi_k_proba(mdl.steps[-1][1], Xh, ks)
        tp = (time.perf_counter() - t2) / len(ks)
        return [(probs[k], (t2 - t1) / len(ks), tp) for k in ks]

    def one(params):
        t1 = time.perf_counter()
        mdl = make_model(name, cfg, seed, params).fit(Xf, yf)
        t2 = time.perf_counter()
        ph = mdl.predict_proba(Xh)[:, 1]
        return ph, t2 - t1, time.perf_counter() - t2

    nj = min(cfg["project"]["n_jobs"], len(grid)) if name in ("svm", "dt") else 1
    if nj > 1:
        return Parallel(n_jobs=nj, backend="threading")(delayed(one)(g) for g in grid)
    return [one(g) for g in grid]


def _fit_iforest(cfg, gather, rows, healthy_drive_mask, sid, seed):
    rng = np.random.default_rng(seed)
    h = rows[healthy_drive_mask[sid[rows]]]
    n = min(len(h), cfg["isolation_forest"]["fit_rows"])
    sample = np.sort(rng.choice(h, n, replace=False))
    return make_isolation_forest(cfg, seed).fit(gather(sample)), int(len(h))


def run(cfg: dict, spec: RunSpec, force: bool = False) -> dict:
    out_path = result_path(cfg, spec)
    if out_path.exists() and not force:
        return load_json(out_path)
    t_start = time.perf_counter()
    timings: dict = {}
    seed = spec.seed
    seed_everything(seed)
    log.info("=== run %s/%s/%s (%s, %s) ===", spec.family, spec.subset, spec.tag, spec.variant, spec.split)
    store = FeatureStore(cfg, spec.subset)
    meta = store.meta
    sid = meta["serial_id"].to_numpy()
    day = meta["day"].to_numpy()
    fail = meta["fail"].to_numpy()
    H = spec.horizon
    attrs = list(spec.attrs or cfg["attributes"])
    batch = cfg["project"]["batch_rows"]

    # ---------------------------------------------------------- labels & split
    t0 = time.perf_counter()
    if spec.split == "random":
        keep, y = label_rows(day, fail, H, day_of(cfg[spec.dataset]["period_end"]))
        assign = get_split(store, seed, (cfg["split"]["train"], cfg["split"]["val"], cfg["split"]["test"]))
        R = rows_by_split(sid, keep, assign)
    else:
        tr_end, te_end, lab_end = temporal_bounds(cfg, spec.dataset)
        keep_te, y = label_rows(day, fail, H, lab_end)
        keep_tr, _ = label_rows(day, fail, H, tr_end)
        R = temporal_rows(day, keep_tr, keep_te, sid, tr_end, te_end, H, cfg["temporal"]["val_frac"], seed, y)
    tr, va, te = R["train"], R["val"], R["test"]
    for name, rr in R.items():
        if y[rr].sum() == 0:
            raise RuntimeError(f"{name} split has no positive rows ({spec})")
    counts = {k: {"rows": int(len(v)), "positives": int(y[v].sum()), "drives": int(len(np.unique(sid[v]))),
                  "failed_drives": int(len(np.unique(sid[v][y[v] == 1])))} for k, v in R.items()}
    log.info("rows/positives: %s", {k: (v["rows"], v["positives"]) for k, v in counts.items()})
    timings["split"] = round(time.perf_counter() - t0, 1)

    # ---------------------------------------------------------- preprocessing
    extra = {}
    if len(store.models) > 1 and spec.attrs is None:
        mid = meta["model_id"].to_numpy()
        extra = {f"model_{j}": (mid == j).astype(np.float32) for j in range(len(store.models))}
    ndp = n_delta_pos(cfg, store, attrs, spec.window)
    if spec.variant == "full":
        extra["n_delta_pos"] = ndp
    cands, flags = candidate_columns(cfg, spec, extra)
    fs, cl, mi = cfg["feature_selection"], cfg["cleaning"], cfg["missing_indicator"]
    t0 = time.perf_counter()
    pre = TCSEPreprocessor(candidates=cands, flag_columns=flags, passthrough_log=[],
                           max_missing_frac=cl["max_missing_frac"], indicator_min=mi["min_frac"],
                           indicator_max=mi["max_frac"], corr_threshold=fs["corr_threshold"],
                           mi_neighbors=fs["mi_neighbors"], mi_neg_per_pos=fs["mi_negatives_per_positive"],
                           mi_max_rows=fs["mi_max_rows"], min_mi=fs["min_mi"], max_features=fs["max_features"],
                           corr_sample_rows=fs["corr_sample_rows"], random_state=seed)
    pre.fit(StoreSource(store, tr, extra), y[tr])
    timings["preprocess_fit"] = round(time.perf_counter() - t0, 1)
    log.info("features selected (%d): %s", len(pre.selected_), pre.selected_)
    gather = Gather(store, extra, pre)

    # ---------------------------------------------------------- CV folds & tuning sample
    t0 = time.perf_counter()
    folds = group_folds(sid[tr], y[tr], cfg["split"]["cv_folds"], seed)
    rng = np.random.default_rng(seed)
    n_sid = int(sid.max()) + 1
    pos_drive = np.zeros(n_sid, bool)
    pos_drive[sid[tr][y[tr] == 1]] = True
    frac = cfg["tuning"]["heldout_healthy_drive_frac"]
    sampled = rng.random(n_sid) < frac
    in_tune = (pos_drive | sampled)[sid[tr]]                   # positions in tr
    w_tr = np.where(pos_drive[sid[tr]], 1.0, 1.0 / frac)       # inverse-probability weights
    never_failed = np.ones(n_sid, bool)
    never_failed[np.unique(sid[fail != np.iinfo(np.int32).max])] = False

    grids = {n: param_grid(cfg, n) for n in BASELINES}
    if spec.svm_override:                 # exact-SVC check: one fixed parameter set
        grids["svm"] = [dict(spec.svm_override)]
    tune_pos = np.flatnonzero(in_tune)
    pos_index = -np.ones(len(tr), dtype=np.int64)
    pos_index[tune_pos] = np.arange(len(tune_pos))
    oof = {n: np.full((len(grids[n]), len(tune_pos)), np.nan, np.float32) for n in BASELINES}
    oof_if = np.full(len(tune_pos), np.nan, np.float32)
    f1s = {n: np.zeros((len(grids[n]), len(folds))) for n in BASELINES}
    model_time = {n: {"fit": 0.0, "predict": 0.0} for n in list(BASELINES) + ["iforest"]}
    fold_info = []
    for k, (fit_p, hold_p) in enumerate(folds):
        rows_fit = tr[fit_p]
        u = undersample_rows(rows_fit, y[rows_fit], cfg["sampling"]["undersample_ratio"], seed * 100 + k)
        Xf, yf = gather(rows_fit[u]), y[rows_fit[u]]
        hp = hold_p[in_tune[hold_p]]
        Xh, yh, wh = gather(tr[hp]), y[tr[hp]], w_tr[hp]
        idx = pos_index[hp]
        for n in BASELINES:
            for g, (ph, tf, tp) in enumerate(_grid_fit_predict(n, grids[n], cfg, seed, Xf, yf, Xh)):
                oof[n][g, idx] = ph
                f1s[n][g, k] = weighted_f1(yh, ph, 0.5, wh)
                model_time[n]["fit"] += tf
                model_time[n]["predict"] += tp
        t1 = time.perf_counter()
        iso, n_h = _fit_iforest(cfg, gather, rows_fit, never_failed, sid, seed * 100 + k)
        oof_if[idx] = anomaly_score(iso, Xh)
        model_time["iforest"]["fit"] += time.perf_counter() - t1
        fold_info.append({"fit_rows": int(len(rows_fit)), "undersampled": int(len(u)),
                          "heldout_rows": int(len(hold_p)), "heldout_scored": int(len(hp)),
                          "heldout_pos": int(yh.sum()), "iforest_healthy_rows": n_h})
        log.info("fold %d/%d done: fit %d -> %d rows, held-out scored %d (%d pos)", k + 1, len(folds),
                 len(rows_fit), len(u), len(hp), int(yh.sum()))
    best = {}
    tuning = {}
    for n in BASELINES:
        mean = f1s[n].mean(axis=1)
        g = int(np.argmax(mean))
        best[n] = grids[n][g]
        tuning[n] = {"grid": [{"params": grids[n][i], "f1_mean": float(mean[i]),
                               "f1_folds": f1s[n][i].round(5).tolist()} for i in range(len(grids[n]))],
                     "best": best[n], "best_f1": float(mean[g])}
    log.info("tuned: %s", {n: (best[n], round(tuning[n]["best_f1"], 4)) for n in BASELINES})
    timings["tune_and_oof"] = round(time.perf_counter() - t0, 1)
    timings["tune_by_model"] = {n: {k: round(v, 1) for k, v in d.items()} for n, d in model_time.items()}
    log.info("tuning time by model: %s", timings["tune_by_model"])

    # ---------------------------------------------------------- meta-learner on OOF
    t0 = time.perf_counter()
    rows_tune = tr[tune_pos]
    oof_tab = {"p_dt": oof["dt"][grids["dt"].index(best["dt"])], "p_knn": oof["knn"][grids["knn"].index(best["knn"])],
               "p_nb": oof["nb"][grids["nb"].index(best["nb"])], "p_svm": oof["svm"][grids["svm"].index(best["svm"])],
               "s_iforest": oof_if, "n_delta_pos": ndp[rows_tune]}
    y_tune, w_tune, g_tune = y[rows_tune], w_tr[tune_pos], sid[rows_tune]
    use_nd = spec.variant == "full"
    metas = {}
    variants = {"tcse": meta_columns(True, use_nd)}
    if spec.variant == "full":
        variants["tcse_no_if"] = meta_columns(False, use_nd)
    for vname, cols in variants.items():
        Z = np.column_stack([oof_tab[c] for c in cols])
        model, info = fit_meta(Z, y_tune, g_tune, cfg, seed, sample_weight=w_tune)
        info["coefficients"] = meta_coefficients(model, cols)
        info["columns"] = cols
        metas[vname] = (model, info)
    timings["meta_fit"] = round(time.perf_counter() - t0, 1)

    # ---------------------------------------------------------- refit level 0 on full training split
    t0 = time.perf_counter()
    u = undersample_rows(tr, y[tr], cfg["sampling"]["undersample_ratio"], seed)
    Xfull, yfull = gather(tr[u]), y[tr[u]]
    models, resample_log = {}, {}
    for n in BASELINES:
        mdl = make_model(n, cfg, seed, best[n])
        resample_log[n] = resampled_counts(mdl, Xfull, yfull)
        resample_log[n]["before"] = [int((y[tr] == 0).sum()), int((y[tr] == 1).sum())]
        models[n] = mdl.fit(Xfull, yfull)
    iso, _ = _fit_iforest(cfg, gather, tr, never_failed, sid, seed)
    del Xfull
    timings["refit"] = round(time.perf_counter() - t0, 1)
    log.info("class counts (healthy, failure) before/after resampling: %s", resample_log)

    t0 = time.perf_counter()
    L0 = {"val": predict_level0(models, iso, gather, va, batch, "val"),
          "test": predict_level0(models, iso, gather, te, batch, "test")}
    timings["predict_val_test"] = round(time.perf_counter() - t0, 1)
    timings["predict_by_model"] = dict(PREDICT_TIMES)
    log.info("prediction time by model: %s", PREDICT_TIMES)

    def scores(split_rows, l0, nd):
        tab = {"p_dt": l0[:, 0], "p_knn": l0[:, 1], "p_nb": l0[:, 2], "p_svm": l0[:, 3],
               "s_iforest": l0[:, 4], "n_delta_pos": nd}
        out = {n: l0[:, j] for j, n in enumerate(BASELINES)}
        for vname, (model, info) in metas.items():
            Z = np.column_stack([tab[c] for c in info["columns"]])
            out[vname] = model.predict_proba(Z)[:, 1].astype(np.float32)
        return out

    S = {"val": scores(va, L0["val"], ndp[va]), "test": scores(te, L0["test"], ndp[te])}
    grid = thresholds(cfg)
    ratios = cfg["cost_ratios"]
    c_fp = cfg["c_fp"]
    results, curves = {}, {}
    for mname in S["test"]:
        res = {}
        for r in ratios:
            th, vc = best_threshold(y[va], S["val"][mname], grid, r, c_fp)
            m = evaluate(y[te], S["test"][mname], th, ratios, c_fp)
            m.update(drive_level_recall(y[te], S["test"][mname], th, sid[te]))
            m["val_cost"] = float(vc.min())
            res[f"r{r}"] = m
        m = evaluate(y[te], S["test"][mname], 0.5, ratios, c_fp)
        m.update(drive_level_recall(y[te], S["test"][mname], 0.5, sid[te]))
        res["default"] = m
        results[mname] = res
        if mname.startswith("tcse"):
            cv_ = confusion_curve(y[va], S["val"][mname], grid)
            ct = confusion_curve(y[te], S["test"][mname], grid)
            curves[mname] = {"thresholds": grid.tolist(),
                             "val": {f"r{r}": cost(cv_["fn"], cv_["fp"], r * c_fp, c_fp).tolist() for r in ratios},
                             "test": {f"r{r}": cost(ct["fn"], ct["fp"], r * c_fp, c_fp).tolist() for r in ratios}}
    roc = None
    if H == cfg["evaluation"]["roc_horizon"]:
        roc = {mname: roc_points(y[te], S["test"][mname]) for mname in S["test"]}

    # ---------------------------------------------------------- cross-dataset test
    cross = None
    if spec.cross_target:
        t0 = time.perf_counter()
        cross = cross_dataset_eval(cfg, spec, pre, models, metas, results, ratios, c_fp)
        timings["cross"] = round(time.perf_counter() - t0, 1)

    res_all = {
        "spec": asdict(spec), "config_hash": cfg["_hash"], "counts": counts,
        "features": {"candidates": cands, "selected": pre.selected_, "dropped": pre.drop_reasons_,
                     "indicators": pre.indicators_, "dropped_attributes": pre.dropped_attributes_,
                     "mutual_info": pre.mutual_info_},
        "folds": fold_info, "tuning": tuning, "resampling": resample_log,
        "meta": {k: v[1] for k, v in metas.items()},
        "results": results, "cost_curves": curves, "roc": roc, "cross": cross,
        "timings": {**timings, "total": round(time.perf_counter() - t_start, 1)},
    }
    save_json(res_all, out_path)
    if seed == cfg["project"]["seeds"][0]:
        mdir = p(cfg, "results") / "models" / spec.family / spec.subset
        mdir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"preprocessor": pre, "best_params": best,
                     "meta": {k: v[0] for k, v in metas.items()}}, mdir / f"{spec.tag}.joblib")
    log_event(cfg, "run", family=spec.family, subset=spec.subset, tag=spec.tag, seed=seed,
              counts=counts, timings=res_all["timings"])
    log.info("TCSE r10: %s | total %.0fs", {k: round(results["tcse"]["r10"][k], 4)
                                            for k in ("precision", "recall", "f1", "mcc", "roc_auc")},
             res_all["timings"]["total"])
    return res_all


def cross_dataset_eval(cfg, spec, pre, models, metas, results, ratios, c_fp) -> dict:
    """Apply the source-trained models to the target dataset's test split.

    Scalers (medians, log/z-score parameters) are refitted on the target's
    unlabelled training-split features; selected features, models and
    thresholds (chosen on the source validation set) stay frozen."""
    tgt = spec.cross_target
    tstore = FeatureStore(cfg, tgt["subset"])
    tm = tstore.meta
    sid, day, fail = tm["serial_id"].to_numpy(), tm["day"].to_numpy(), tm["fail"].to_numpy()
    keep, y = label_rows(day, fail, spec.horizon, day_of(cfg[tgt["dataset"]]["period_end"]))
    assign = get_split(tstore, spec.seed, (cfg["split"]["train"], cfg["split"]["val"], cfg["split"]["test"]))
    R = rows_by_split(sid, keep, assign)
    attrs = list(spec.attrs)
    ndp = n_delta_pos(cfg, tstore, attrs, spec.window)
    extra = {"n_delta_pos": ndp} if spec.variant == "full" else {}
    tpre = copy.deepcopy(pre).refit_scaling(StoreSource(tstore, R["train"], extra))
    gather = Gather(tstore, extra, tpre)
    te = R["test"]
    iso = _target_iso(cfg, gather, R["train"], spec.seed)
    l0 = predict_level0(models, iso, gather, te, cfg["project"]["batch_rows"], "cross")
    tab = {"p_dt": l0[:, 0], "p_knn": l0[:, 1], "p_nb": l0[:, 2], "p_svm": l0[:, 3],
           "s_iforest": l0[:, 4], "n_delta_pos": ndp[te]}
    S = {n: l0[:, j] for j, n in enumerate(BASELINES)}
    for vname, (model, info) in metas.items():
        S[vname] = model.predict_proba(np.column_stack([tab[c] for c in info["columns"]]))[:, 1]
    out = {"target": tgt, "counts": {"test_rows": int(len(te)), "test_positives": int(y[te].sum())}, "results": {}}
    for mname, pt in S.items():
        res = {}
        for r in ratios:
            th = results[mname][f"r{r}"]["threshold"]       # chosen on the SOURCE validation set
            m = evaluate(y[te], pt, th, ratios, c_fp)
            m.update(drive_level_recall(y[te], pt, th, sid[te]))
            res[f"r{r}"] = m
        out["results"][mname] = res
    return out


def _target_iso(cfg, gather, train_rows, seed):
    """The anomaly detector learns 'normal' behaviour without labels; for the
    target domain it is refitted on the target's unlabelled training rows
    (like the scalers). Failure labels of the target are never used."""
    rng = np.random.default_rng(seed)
    n = min(len(train_rows), cfg["isolation_forest"]["fit_rows"])
    sample = np.sort(rng.choice(train_rows, n, replace=False))
    return make_isolation_forest(cfg, seed).fit(gather(sample))
