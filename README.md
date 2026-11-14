# TCSE: Trend-Aware Cost-Sensitive Stacking for Early Hard Drive Failure Prediction

Full implementation of the Stage-1 report *"Trend-Aware Cost-Sensitive Stacking for Early Hard
Drive Failure Prediction"* (K. Baiju Sindhu, M. Mohan, A. Choudhary, BITS Pilani Dubai).
Everything runs on the real Backblaze Drive Stats (Q1–Q2 2026) and SMART-Z datasets; every number
in `results/` is produced by code from those files and a fixed config + seed.

<!-- RESULTS-SUMMARY -->

## 1. Setup

```bash
# Python 3.11
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on Linux)
pip install -r requirements.txt
pip install -e .                  # makes `tcse` importable (src layout)
```

## 2. Data

| Dataset | Source | What to download | Where |
|---|---|---|---|
| Backblaze Drive Stats | <https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data> | `data_Q1_2026.zip`, `data_Q2_2026.zip` (1.31 GB + 2.09 GB) | `data/raw/backblaze/` |
| SMART-Z | OSF, doi:[10.17605/OSF.IO/24Y6G](https://doi.org/10.17605/OSF.IO/24Y6G) (Wei et al., *Sci. Data* 12, 1125, 2025), CC BY 4.0 | `log_hdsmart_base.csv`, `log_hdsmart_param.zip`, `offline_hdsmart.csv` | `data/raw/smartz/` |

`python scripts/run_pipeline.py --stage download` fetches all five files (Backblaze from its public
B2 bucket, SMART-Z through the OSF download links), checks their sizes and extracts the zips. If a
host is unreachable the script stops and prints what to download and where to put it; it never
substitutes other data.

### Real column names (inspected with DuckDB before any processing code was written)

| File | Columns |
|---|---|
| Backblaze daily CSV (197 columns) | `date, serial_number, model, capacity_bytes, failure, datacenter, cluster_id, vault_id, pod_id, pod_slot_num, is_legacy_format, smart_1_normalized, smart_1_raw, …` (raw + normalised value per SMART id) |
| `log_hdsmart_base.csv` (14,525,830 rows) | `keyHDid, keyScanTime, omcip, czname, Type, Model, Capacity, bureau` |
| `log_hdsmart_param/log-hdsmart-YYYYMM_param.csv` (12 files, 15.4 GB, long format) | `keyHDid, keyScanTime, SMART_ID, SMART_NAME, rawvalue, thresh, type, update, value, worst` |
| `offline_hdsmart.csv` (712 rows, 711 distinct serials) | `brand, bureau, capacity, devicetype, disktype, hdid, hdnum, lastlogtime, omcip, powerontime` |

`keyScanTime` is `YYYYMMDD` (stored as a float in the base log), `lastlogtime` is `YYYYMMDD`.
SAS drives in SMART-Z report SCSI log pages under ids 900+, not ATA SMART ids, so only SATA HDDs
share attributes with Backblaze.

<!-- DATASET-COUNTS -->

## 3. Reproducing every table and figure

```bash
python scripts/run_all.py --config configs/default.yaml     # full grid, resumable
python scripts/make_tables.py --config configs/default.yaml # results/tables + results/figures
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_backblaze.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_eda_smartz.ipynb
pytest
```

Single stages / runs:

```bash
python scripts/run_pipeline.py --stage integrate --dataset backblaze
python scripts/run_pipeline.py --stage features  --dataset smartz
python scripts/run_pipeline.py --stage window    --dataset smartz
python scripts/run_pipeline.py --stage run --dataset smartz --horizon 7 --seed 0
python scripts/run_pipeline.py --stage run --dataset backblaze --horizon 30 --seed 3 --family no_trend
```

`run_all.py` caches every finished run as JSON in `results/metrics/<family>/<subset>/H<H>_m<m>_s<seed>.json`
and skips it on the next call, so an interrupted grid resumes where it stopped. Every run logs the
config hash, seed, row/positive counts and per-stage timings to `results/logs/runs.jsonl`.

## 4. Repository layout

```
configs/default.yaml        every tunable value (paths, models, horizons, windows, seeds, costs, grids, caps)
src/tcse/data/download.py   download + size checks, clear errors when data are missing
src/tcse/data/integrate.py  DuckDB integration to a common schema (report §IV-C1), pooling rule
src/tcse/data/label.py      Eq. 1 labels, post-failure removal, right-censoring
src/tcse/features/clean.py  impossible values, per-drive forward fill, train-fitted drops / medians
src/tcse/features/trends.py rolling mean/std/delta/slope (Eqs. 2-3) -> column-major memmap feature store
src/tcse/features/transform.py  train-only sklearn transformer: drops, medians, log1p, z-score, MI + |r| selection
src/tcse/split.py           drive-level stratified group split, temporal hold-out, grouped CV folds
src/tcse/sampling.py        undersampling + SMOTE (+ KNN/SVM cap) inside an imblearn Pipeline
src/tcse/models/baselines.py DT, KNN, NB, RBF-SVM
src/tcse/models/tcse.py     Isolation Forest score, class-weighted logistic meta-learner
src/tcse/pipeline.py        one experiment run (tuning, OOF stacking, refit, thresholds, metrics, cross-dataset)
src/tcse/experiments.py     experiment grid, window selection, shared attributes
src/tcse/evaluate.py        Eqs. 10-15, cost-optimal threshold, drive-level recall
src/tcse/eda.py, plots.py   EDA (training split only) and result figures
scripts/                    run_pipeline.py, run_all.py, make_tables.py, make_notebooks.py
notebooks/                  01_eda_backblaze.ipynb, 02_eda_smartz.ipynb
tests/                      pytest suite (synthetic fixtures only here)
```

## 5. How the pipeline implements the report

| Report | Implementation |
|---|---|
| §IV-C1 integration | DuckDB scans the 181 daily Backblaze CSVs (`union_by_name`, values `TRY_CAST`), keeps the two models, de-duplicates `(serial, date)` and writes Parquet. SMART-Z: the long parameter log is filtered to the selected drives and attribute ids and pivoted month by month to one row per drive-day, then joined to the drive model and the offline list. |
| Failure day | Backblaze: first day with `failure = 1`. SMART-Z: `lastlogtime` in `offline_hdsmart.csv`. |
| Eq. 1 | `y = 1` iff the failure day is in (t, t+H]. Rows on/after the failure day are removed; rows with t + H past the end of the period are removed (censoring). |
| §IV-C2 cleaning | Negative counters and temperatures outside 0–80 °C become missing. Columns > 90 % empty (raw, pre-fill missingness) or with zero variance on the **training** split are dropped. |
| §IV-C3 missing values | Forward fill inside each drive only; remaining gaps get the training median; binary `miss_<id>` indicators are kept for attributes whose training missing rate is in [0.1 %, 90 %]. |
| §IV-C4 trends (Eqs. 2–3) | Exact backward-looking windows over the last m records of the same drive: mean, sample std, Δ = x_t − x_{t−m+1} (earliest available value for short histories), OLS slope against the record index; NaN with < 2 values. Plus drive age (SMART 9) and `n_delta_pos`, the number of critical attributes (5, 187, 188, 197, 198 when present) with Δ > 0. Vectorised NumPy sliding windows that never cross drive boundaries, stored in a column-major memory-mapped float32 matrix. |
| m ∈ {3, 7, 14} | Chosen per dataset on the **validation** set (H = 7, seed 0): the window with the highest TCSE validation average precision. |
| §IV-C5 (Eqs. 4–5) | `ln(1+x)` for non-negative columns, `sign(x)·ln(1+abs(x))` for columns that take negative values (Δ, slope), then z-score; parameters fitted on training rows only. |
| §IV-C6 selection | Mutual information with the target (training rows) ranks the features; walking down the ranking, a feature is dropped if abs(r) > 0.95 with one already kept. |
| §IV-C7 split | Drives go to 70/15/15 train/val/test, stratified on (drive model, ever fails); disjointness is asserted; assignments are saved per seed. Temporal hold-out: Backblaze trains on Jan–Mar 2026 (records up to 31 Mar − H) and tests on Apr–May 2026, with June used only for labels; SMART-Z trains on the first 9 months (Mar–Nov 2017) and tests on Dec 2017–Feb 2018. |
| §IV-C8 (Eq. 6) | Inside every CV fold and the final training split: random undersampling to 20 : 1 healthy : failure, SMOTE to 2 : 1, then (KNN, SVM only) a stratified cap of 50,000 rows, as an imblearn `Pipeline`. Validation and test keep the natural distribution. |
| §IV-E baselines | DT (CART, `class_weight=balanced`, depth grid), KNN (distance-weighted, k ∈ {5, 15, 31}), Gaussian NB, RBF SVM (class-weighted, Platt scaling). Grid search with 5-fold stratified group CV (groups = serial number), scored by F1. |
| §IV-F TCSE | Out-of-fold `p_DT, p_KNN, p_NB, p_SVM` from the same 5 grouped folds; an Isolation Forest fitted on never-failing drives of each fold (`s = −score_samples`, Eq. 7, higher = more anomalous); `n_delta_pos`. L2 logistic regression with inverse class-frequency weights (Eqs. 8–9), α = 1/C tuned by grouped CV (average precision). Level 0 is refitted on the full training split for validation and test. |
| Eqs. 10–11 | θ* per method and per r ∈ {5, 10, 50} minimises validation cost C = r·FN + FP and is applied unchanged to the test set. Baselines get the same treatment. |
| §IV-G | Precision, recall, F1, ROC-AUC, MCC, total cost, confusion matrix, drive-level recall; 10 seeds; Wilcoxon signed-rank test (TCSE vs. the best baseline per metric). Ablations: no trend features (re-run with raw current-day attributes only), no Isolation Forest (meta-learner refitted without `s`), default threshold 0.5. |
| Cross-dataset | Shared attributes (5, 9, 193, 194, 197, 198, 199). Each method is trained on one dataset and tested on the other's test split. Scalers, medians and the label-free Isolation Forest are refitted on the target's unlabelled training features; thresholds come from the source validation set. |

## 6. Deviations from the report, and why

1. **8 GB RAM instead of 16 GB.** The laptop used has 7.8 GB RAM (often < 1 GB free). This is the
   report's "8 GB minimum with chunked loading" setting: DuckDB with a 2 GB memory limit and disk
   spill, a column-major memory-mapped feature store, column-streamed preprocessing fits and
   batched prediction. No data were dropped to fit in memory.
2. **RBF-SVM via the Nyström approximation.** Exact libsvm `SVC(kernel="rbf")` with Platt scaling
   took about 16 min to fit on 10k resampled rows and predicted about 2,700 rows/s on 4 threads.
   One Backblaze run needs about 6.5 M predictions plus 4 fits per grid point per fold, i.e. weeks
   for the whole grid. The SVM is therefore an RBF-kernel SVM with a 500-component Nyström feature
   map, a class-weighted linear SVM (liblinear) and Platt scaling
   (`CalibratedClassifierCV(method="sigmoid")`). The exact SVC is still available
   (`baselines.svm.exact_params`) and is compared with the approximation on one configuration
   (Table 7). C ∈ {0.1, 1}; C = 10 did not converge in reasonable time.
3. **Rows used for grid-search scoring and meta-learner training.** Scoring every held-out fold
   (about 2 M rows per fold for Backblaze) for every grid point was too slow. Held-out folds are
   scored on all rows of every failed drive plus all rows of a random 15 % of healthy drives. Healthy
   rows carry an inverse-probability weight (1/0.15), so the F1 estimate is unbiased for the full
   fold. The out-of-fold meta-features of exactly these rows train the meta-learner, with the class
   weights of Eq. 9 computed on the weighted (natural) class totals. Validation and test sets are
   **never** subsampled.
4. **Threshold grid extended above 0.99.** Models trained on rebalanced data output inflated
   probabilities. With positives at 0.002–0.06 % of drive-days, the validation-cost optimum sat at
   the 0.99 edge of the report's grid for almost every method. The grid keeps 0.01…0.99 (step 0.01)
   and adds 41 log-spaced points 1 − 10^−k, k ∈ [2, 6] (`threshold.high_tail`; set it to `null` for
   the report's grid only).
5. **Window-selection criterion.** The report does not fix one. Validation cost at r = 10 was
   identical for every m on SMART-Z (the cost-optimal decision there is to raise no alarm), so m is
   the window with the highest TCSE validation average precision; cost breaks ties.
6. **Pooling.** The report's rule fired for both datasets (WDC: 74 failed drives in the training
   split; SMART-Z: 88 and 31), so each dataset is modelled as one pooled subset with drive-model
   one-hot features.
7. **SMART-Z model choice.** SAS drives in SMART-Z report SCSI ids (900+), so the two most populous
   **SATA** models were used: WDC WD4000FYYZ-01UL1B2 and WDC WD2000FYYZ-01UL1B2.
8. **Seeds for the secondary experiments.** The main grid and the no-trend ablation use all 10
   seeds. The temporal hold-out and cross-dataset runs use seeds 0–2 (`family_seeds` in the config)
   because of compute time. Runs are cached, so adding seeds only computes the new ones.
9. **Rolling windows are record-based** (the last m records of a drive). Backblaze is almost
   perfectly daily; some SMART-Z drives have gaps, so there a window can span more than m days.
10. **Isolation Forest input.** Each tree sees `max_samples = 256` rows, so the forest is fitted on
    a random 100,000 healthy drive-days of the fold (the same distribution as sampling from all).
11. **Correlation and MI estimation.** Pearson r is estimated on 500,000 random training rows, and
    MI on all positive training rows plus 20 negatives per positive (with 0.01 % positives, MI on
    raw rows is dominated by the majority class). Both use training data only.

