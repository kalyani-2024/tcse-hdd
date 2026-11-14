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

<!-- DESIGN -->
