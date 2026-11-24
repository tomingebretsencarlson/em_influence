"""Standalone runner for bottom_twins.py (no notebook needed): rebuilds the notebook's df and
FIT_DATA["middle"] via rubric_eval_additions/common.py (verified to reproduce the notebook's
numbers), runs the twins, writes CSVs to <rubric_eval>/fit_results/bottom_twins and PNGs to
notebooks/bottom_twins_figures. Prints the tables.

    .venv/bin/python run_bottom_twins.py [n_boot]
"""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "rubric_eval_additions"))
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import bottom_twins as BT  # noqa: E402

n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else C.N_BOOT
t0 = time.time()
df, METRICS = C.load_df()
fd = C.fit_data(df, "middle")
TW = BT.run(df, fd, METRICS, x=C.X_FIT, n_boot=n_boot, out_dir=C.RUBRIC_EVAL_DIR / "fit_results" / "bottom_twins")
BT.plot_all(TW, fig_dir=HERE / "bottom_twins_figures", show=False)
pd.set_option("display.width", 250)
print(f"\n[{time.time() - t0:.0f}s] singles (AUROC, positives = top / bottom, negatives = middle):")
print(TW["singles"].round(3).to_string())
print("\ndrop-one (Δ AUROC without − with; negative = helps):")
print(TW["drop"].round(4).to_string())
print("\nadd-one cross terms, sorted by top delta (positive = helps):")
print(TW["add_one"].sort_values("add dAUROC top", ascending=False).round(4).to_string())
print("figures written")
