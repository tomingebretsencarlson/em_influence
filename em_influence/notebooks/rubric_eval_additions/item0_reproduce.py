"""Reproduction check: recompute the notebook's headline numbers with common.py.

Targets (from the notebook's saved outputs, cells 18-23):
  middle: main AUROC 0.825 AP 0.513 R-prec 0.487; cross AUROC 0.840 AP 0.568 R-prec 0.525; base rate 0.139, n 23939
  rest:   main AUROC 0.734 AP 0.106 R-prec 0.167; cross AUROC 0.753 AP 0.125 R-prec 0.199; base rate 0.032, n 105090
  cell 21 CIs (middle): joint main [0.818, 0.833], joint cross [0.833, 0.848]
  cell 23 drop-one directive_onset, middle, cross: dAUROC -0.035 [-0.040, -0.030], dAP -0.069 [-0.080, -0.060]
  cell 4 group table: seen 400 docs 26766 tokens 1665 top2; random_unseen 1000 docs 65356 tokens 1005 top2
"""
import json
import numpy as np
import pandas as pd
import common as C

df, METRICS = C.load_df()
SCORE_COLS = [f"score_{m}" for m in METRICS]
out = {}

summary = df.groupby("group").agg(docs=("example_idx", "nunique"), tokens=("position", "size"),
                                  top2=("pctile", lambda p: (p <= 2).sum())).reindex(C.GROUPS)
print(summary)
out["groups"] = summary.to_dict()

for contrast in C.FIT_CONTRASTS:
    fd = C.fit_data(df, contrast)
    print(f"\n{contrast}: n={len(fd['sub'])}, base rate {fd['is_pos'].mean():.3f}")
    preds = {"main": C.oof_predict(fd, SCORE_COLS, set()), "cross": C.oof_predict(fd, SCORE_COLS, {"cross"})}
    for n, p in preds.items():
        m = C.fit_metrics(p, fd["is_pos"])
        print(f"  {n:>5}: " + "  ".join(f"{k} {v:.4f}" for k, v in m.items()))
        out[f"{contrast}/{n}"] = m
    if contrast == "middle":
        ci = C.abs_boot(preds, fd["is_pos"], fd["doc_rows"], {"AUROC": C.auroc})
        print(ci.round(4).to_string())
        out["middle/ci"] = ci.to_dict()
        drop = C.oof_predict(fd, [c for c in SCORE_COLS if c != "score_directive_onset"], {"cross"})
        d = C.paired_boot({"cross": preds["cross"], "drop directive_onset": drop}, fd["is_pos"], fd["doc_rows"],
                          {"AUROC": C.auroc, "AP": C.average_precision}, ref="cross")
        print(d.round(4).to_string())
        out["middle/drop_directive_onset"] = d.loc["drop directive_onset"].to_dict()

(C.OUT_DIR / "item0_reproduce.json").write_text(json.dumps(out, indent=1, default=float))
