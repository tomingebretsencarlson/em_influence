"""Standalone runner for square_twins.py: rebuilds df / FIT_DATA["middle"] via rubric_eval_additions/common.py,
asserts main + all squares against common.oof_predict(..., {"sq"}), writes CSVs to
<rubric_eval>/fit_results/bottom_twins and PNGs to notebooks/bottom_twins_figures.

    .venv/bin/python run_square_twins.py [n_boot]
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
import square_twins as SQ  # noqa: E402

n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else C.N_BOOT
t0 = time.time()
df, METRICS = C.load_df()
fd = C.fit_data(df, "middle")
SCORE_COLS = [f"score_{m}" for m in METRICS]
ref = C.oof_predict(fd, SCORE_COLS, {"sq"})
TW = SQ.run(df, fd, METRICS, x=C.X_FIT, n_boot=n_boot, out_dir=C.RUBRIC_EVAL_DIR / "fit_results" / "bottom_twins", ref_main_sq=ref)
SQ.plot_all(TW, fig_dir=HERE / "bottom_twins_figures", show=False)
pd.set_option("display.width", 250)
print(f"\n[{time.time() - t0:.0f}s] block: all nine squares added at once (Δ AUROC, positive = helps)")
print(TW["block"].round(4).to_string())
print("\nper rubric alone: linear vs quadratic")
print(TW["singles"].round(4).to_string())
print("\nadd one z² / drop one z², by base model")
print(TW["sq"].round(4).to_string())
print("figures written")
