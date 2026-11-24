"""Item 2: what do the cross terms capture? Nested models with paired document-bootstrap deltas.

main                : 9 standardized rubric scores
main + z²           : + each rubric's square (per-rubric curvature)
main + spread       : + each rubric's judge spread (SD of the 0-9 digit distribution)
main + spread + z²  : both
cross               : main + 36 pairwise products (the notebook's reference model)
cross + z², cross + spread, cross + spread + z²
If main + spread + z² reproduces cross, the products are proxies for curvature and confidence.
If cross still adds on top of spread + z², interactions carry unique signal.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import common as C
import plotting as P

df, METRICS = C.load_df()
SCORE_COLS = [f"score_{m}" for m in METRICS]
MODELS = {"main": set(), "main + z²": {"sq"}, "main + spread": {"spread"}, "main + spread + z²": {"spread", "sq"},
          "cross": {"cross"}, "cross + z²": {"cross", "sq"}, "cross + spread": {"cross", "spread"},
          "cross + spread + z² (reference)": {"cross", "spread", "sq"}}
N_PARAMS = {n: 1 + 9 + 9 * ("sq" in t) + 9 * ("spread" in t) + 36 * ("cross" in t) for n, t in MODELS.items()}
COMPARE = [("main + z²", "main"), ("main + spread", "main"), ("main + spread + z²", "main"), ("cross", "main"),
           ("cross", "main + spread + z²"), ("cross + spread + z² (reference)", "main + spread + z²"),
           ("cross + spread + z² (reference)", "cross"), ("cross + z²", "cross"), ("cross + spread", "cross")]
METRIC_FNS = {"AUROC": C.auroc, "AP": C.average_precision, "pseudo-R2": C.pseudo_r2, "R-prec": C.r_precision}


def subset_doc_rows(fd, mask):
    sub = fd["sub"][mask].reset_index(drop=True)
    return list(sub.groupby("example_idx").indices.values())


abs_rows, delta_rows, ABS = [], [], {}
for contrast in C.FIT_CONTRASTS:
    fd = C.fit_data(df, contrast)
    y = fd["is_pos"]
    ru = (fd["sub"]["group"] == "random_unseen").to_numpy()
    ru_rows = subset_doc_rows(fd, ru)
    preds = {n: C.oof_predict(fd, SCORE_COLS, t) for n, t in MODELS.items()}
    ci_all = C.abs_boot(preds, y, fd["doc_rows"], METRIC_FNS)
    ci_ru = C.abs_boot({k: v[ru] for k, v in preds.items()}, y[ru], ru_rows, METRIC_FNS)
    ABS[contrast] = (ci_all, ci_ru)
    for n in MODELS:
        abs_rows.append({"contrast": contrast, "model": n, "n_params": N_PARAMS[n],
                         **{f"{k} all": ci_all.loc[n, k] for k in METRIC_FNS},
                         **{f"{k} random_unseen": ci_ru.loc[n, k] for k in METRIC_FNS},
                         "tokens found all": round(ci_all.loc[n, "R-prec"] * y.sum()),
                         "tokens found random_unseen": round(ci_ru.loc[n, "R-prec"] * y[ru].sum())})
    for a, b in COMPARE:
        d_all = C.paired_boot({a: preds[a], b: preds[b]}, y, fd["doc_rows"], METRIC_FNS, ref=b).loc[a]
        d_ru = C.paired_boot({a: preds[a][ru], b: preds[b][ru]}, y[ru], ru_rows, METRIC_FNS, ref=b).loc[a]
        for scope, d in [("all", d_all), ("random_unseen", d_ru)]:
            delta_rows.append({"contrast": contrast, "scope": scope, "model": a, "minus": b, **d.to_dict()})
    print(f"done {contrast}")

abs_table = pd.DataFrame(abs_rows)
deltas = pd.DataFrame(delta_rows)
abs_table.to_csv(C.OUT_DIR / "item2_models.csv", index=False)
deltas.to_csv(C.OUT_DIR / "item2_deltas.csv", index=False)
pd.set_option("display.width", 250)
print(abs_table.round(4).to_string())
print(deltas.round(4).to_string())

# ---- figure: paired deltas for the discriminating comparisons, AUROC and AP, both contrasts
labels = {("main + z²", "main"): "main + z²  vs  main", ("main + spread", "main"): "main + spread  vs  main",
          ("main + spread + z²", "main"): "main + spread + z²  vs  main", ("cross", "main"): "cross  vs  main",
          ("cross", "main + spread + z²"): "cross  vs  main + spread + z²",
          ("cross + spread + z² (reference)", "main + spread + z²"): "reference  vs  main + spread + z²",
          ("cross + spread + z² (reference)", "cross"): "reference  vs  cross",
          ("cross + z²", "cross"): "cross + z²  vs  cross", ("cross + spread", "cross"): "cross + spread  vs  cross"}
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
for i, contrast in enumerate(C.FIT_CONTRASTS):
    ys = np.arange(len(COMPARE))[::-1]
    for j, k in enumerate(["AUROC", "AP"]):
        ax = axes[i, j]
        # red-team 2: show both scopes; on random_unseen the AUROC ordering of cross vs spread+z² reverses
        for off, scope, color, mk, lab in [(0.17, "all", P.OI["black"], "s", "all judged tokens"),
                                            (-0.17, "random_unseen", P.OI["blue"], "o", "random_unseen (held-out docs)")]:
            t = deltas[(deltas["contrast"] == contrast) & (deltas["scope"] == scope)].set_index(["model", "minus"])
            v = np.array([t.loc[c, f"d{k}"] for c in COMPARE])
            lo = np.array([t.loc[c, f"d{k} lo"] for c in COMPARE]); hi = np.array([t.loc[c, f"d{k} hi"] for c in COMPARE])
            ax.errorbar(v, ys + off, xerr=P.errbars(v, lo, hi), fmt=mk, color=color, ecolor=color, capsize=3, markersize=4, label=lab)
        ax.axvline(0, **P.CHANCE_KW)
        ax.set_yticks(ys, [labels[c] for c in COMPARE])
        ax.set_xlabel(f"Δ{k}, paired document bootstrap 95% CI")
        ax.set_title(f"top 2% vs {contrast}: {k}", fontsize=9)
axes[0, 1].legend(fontsize=8, loc="lower right")
fig.suptitle("Curvature (z²) plus judge spread match the cross model on pooled rows; on held-out documents the interactions generalize better.\n"
             "The reference (all three) is at least as good as either on every scope and metric.", fontsize=10)
plt.tight_layout()
plt.savefig(C.FIG_DIR / "item2_nested_deltas.png")

# ---- figure: absolute AUROC per model, all vs random_unseen
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
order = list(MODELS)
for ax, contrast in zip(axes, C.FIT_CONTRASTS):
    ci_all, ci_ru = ABS[contrast]
    ys = np.arange(len(order))[::-1]
    for off, ci, lab, mk in [(0.18, ci_all, "all judged tokens", "s"), (-0.18, ci_ru, "random_unseen", "o")]:
        v = ci.loc[order, "AUROC"].to_numpy()
        ax.errorbar(v, ys + off, xerr=P.errbars(v, ci.loc[order, "AUROC lo"], ci.loc[order, "AUROC hi"]), fmt=mk,
                    color=P.OI["black"] if mk == "s" else P.OI["blue"], capsize=2, markersize=4, label=lab)
    ax.set_yticks(ys, [f"{o}  ({N_PARAMS[o]} params)" for o in order])
    ax.set_xlabel("out-of-fold AUROC (95% document-bootstrap CI)")
    ax.set_title(f"top 2% vs {contrast}", fontsize=9)
axes[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
fig.suptitle("Nested models: absolute AUROC. Absolute CIs overlap; the paired deltas above are the test.", fontsize=10)
plt.tight_layout()
plt.savefig(C.FIG_DIR / "item2_nested_abs.png")
print("figures written")
