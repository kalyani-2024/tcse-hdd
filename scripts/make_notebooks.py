"""Generate the EDA notebooks (thin wrappers around tcse.eda). Execute them with
    jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_backblaze.ipynb
"""
import nbformat as nbf

CELLS = [
    ("md", "# Exploratory data analysis: {label}\n\nReport Section IV-D. Everything below uses the "
           "**training split of seed 0 only** (no validation/test data informs a design choice). "
           "All numbers are computed from the real feature store built by `scripts/run_pipeline.py --stage features`."),
    ("code", "import warnings; warnings.filterwarnings('ignore')\nimport pandas as pd\nfrom IPython.display import Image, display\n"
             "from tcse.utils import load_config\nfrom tcse.eda import EDA\npd.set_option('display.width', 200)\n"
             "cfg = load_config('../configs/default.yaml')\neda = EDA(cfg, '{ds}')\n"
             "fig = lambda n: display(Image(filename=f'../results/figures/eda_{ds}_{{n}}.png'))\n"
             "print('attributes present in training data:', eda.attrs)"),
    ("md", "## 1. Summary statistics and class ratio per horizon"),
    ("code", "display(eda.summary_stats())\ndisplay(eda.class_ratio())"),
    ("md", "## 2. Missing-data patterns (raw values, before forward fill)"),
    ("code", "miss = eda.missingness(); fig('missingness')"),
    ("md", "## 3. Failed vs. healthy drives: box plots and Kolmogorov–Smirnov tests"),
    ("code", "ks = eda.failed_vs_healthy(); display(ks); fig('boxplots_ks')"),
    ("md", "## 4. Attribute correlations"),
    ("code", "R, high = eda.correlation(); fig('correlation')\nprint('pairs with |r| > 0.95 (one of each is dropped by feature selection):')\nfor h in high: print(h)"),
    ("md", "## 5. Average 30-day pre-failure trajectories"),
    ("code", "traj = eda.prefailure_trajectories(); fig('prefailure_trajectories')"),
    ("md", "## 6. Two-dimensional PCA projection"),
    ("code", "ev = eda.pca(); print('explained variance ratio:', ev.round(4)); fig('pca')"),
    ("md", "## 7. Choice of the trend window m\n\nKS separation (healthy vs. fails-within-7-days rows) of the "
           "delta / slope / std trend features for each candidate window, plus the window chosen on the "
           "validation set by `tcse.experiments.select_window` (TCSE validation cost)."),
    ("code", "piv, summ = eda.window_separation(); display(summ); display(piv)\nimport json, pathlib\n"
             "for f in pathlib.Path('../results/metrics/window_selection').glob('{ds}*/chosen.json'): print(f.parent.name, json.loads(f.read_text()))"),
]

for ds, label, name in [("backblaze", "Backblaze Drive Stats (Q1–Q2 2026)", "01_eda_backblaze"),
                        ("smartz", "SMART-Z (2017–2018)", "02_eda_smartz")]:
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(c.format(label=label, ds=ds)) if k == "md"
                else nbf.v4.new_code_cell(c.format(label=label, ds=ds)) for k, c in CELLS]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, f"notebooks/{name}.ipynb")
print("notebooks written")
