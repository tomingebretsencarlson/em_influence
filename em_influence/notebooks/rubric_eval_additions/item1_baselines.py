"""Item 1: non-LLM baselines that define what "signal" means.

Adds to the notebook's cell-19 comparison: an out-of-fold word lookup P(top | token), positional
baselines (log reply length, log reply position, first-token flag), and the rubrics on top of
those. Headline numbers on random_unseen (1,000 held-out documents), pooled beside them.
Per rubric: AUROC alone, AUROC of its leave-one-out word mean, residual AUROC (raw minus word
mean), and the out-of-fold increment from adding the rubric to the lookup, with paired
document-bootstrap CIs.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import common as C
import plotting as P

df, METRICS = C.load_df()
SCORE_COLS = [f"score_{m}" for m in METRICS]
METRIC_FNS = {"AUROC": C.auroc, "AP": C.average_precision, "R-prec": C.r_precision}


def subset_doc_rows(fd, mask):
    """Document row groups in the index space of the masked subset (positions 0..mask.sum()-1),
    for bootstrapping predictions that have themselves been restricted with the same mask."""
    sub = fd["sub"][mask].reset_index(drop=True)
    return list(sub.groupby("example_idx").indices.values())


model_rows, screen_rows, delta_rows = [], [], []
fig_models = {}
for contrast in C.FIT_CONTRASTS:
    fd = C.fit_data(df, contrast)
    sub, y = fd["sub"], fd["is_pos"]
    ru = (sub["group"] == "random_unseen").to_numpy()
    ru_rows = subset_doc_rows(fd, ru)

    lookup = C.oof_lookup(fd, alpha=5.0)
    L = C.logit(lookup)[:, None]
    POS = np.column_stack([np.log(sub["reply_len"]), np.log1p(sub["reply_idx"]), sub["is_firstpos"].astype(float)])
    best = max(SCORE_COLS, key=lambda c: C.auroc(sub[c].to_numpy(), y))

    preds = {
        "word lookup": L[:, 0],
        "positional (length, position, first token)": C.oof_predict(fd, [], set(), extra=POS),
        "lookup + positional": C.oof_predict(fd, [], set(), extra=np.column_stack([L, POS])),
        "best single rubric": sub[best].to_numpy(),
        "rubrics main": C.oof_predict(fd, SCORE_COLS, set()),
        "rubrics cross": C.oof_predict(fd, SCORE_COLS, {"cross"}),
        "lookup + rubrics main": C.oof_predict(fd, SCORE_COLS, set(), extra=L),
        "lookup + rubrics cross": C.oof_predict(fd, SCORE_COLS, {"cross"}, extra=L),
        "lookup + positional + rubrics cross": C.oof_predict(fd, SCORE_COLS, {"cross"}, extra=np.column_stack([L, POS])),
    }
    n_pos_all, n_pos_ru = int(y.sum()), int(y[ru].sum())
    ci_ru = C.abs_boot({k: v[ru] for k, v in preds.items()}, y[ru], ru_rows, METRIC_FNS)
    for name, p in preds.items():
        m_all, m_ru = C.fit_metrics(p, y), C.fit_metrics(p[ru], y[ru])
        model_rows.append({"contrast": contrast, "model": name if name != "best single rubric" else f"best single rubric ({best.removeprefix('score_')})",
                           "AUROC all": m_all["AUROC"], "AP all": m_all["AP"], "R-prec all": m_all["R-prec"],
                           "tokens found all": round(m_all["R-prec"] * n_pos_all),
                           "AUROC random_unseen": m_ru["AUROC"], "AUROC ru lo": ci_ru.loc[name, "AUROC lo"], "AUROC ru hi": ci_ru.loc[name, "AUROC hi"],
                           "AP random_unseen": m_ru["AP"], "R-prec random_unseen": m_ru["R-prec"],
                           "tokens found random_unseen": round(m_ru["R-prec"] * n_pos_ru),
                           "n positives all": n_pos_all, "n positives random_unseen": n_pos_ru,
                           "base rate all": y.mean(), "base rate random_unseen": y[ru].mean()})
    fig_models[contrast] = (ci_ru, y[ru].mean())

    # paired deltas on random_unseen rows: does anything beat the lookup?
    for ref, cands in [("word lookup", ["rubrics cross", "lookup + rubrics main", "lookup + rubrics cross"]),
                       ("lookup + positional", ["lookup + positional + rubrics cross"])]:
        d = C.paired_boot({k: preds[k][ru] for k in [ref] + cands}, y[ru], ru_rows, METRIC_FNS, ref=ref)
        for k in cands:
            delta_rows.append({"contrast": contrast, "scope": "random_unseen", "model": k, "minus": ref, **d.loc[k].to_dict()})
        d = C.paired_boot({k: preds[k] for k in [ref] + cands}, y, fd["doc_rows"], METRIC_FNS, ref=ref)
        for k in cands:
            delta_rows.append({"contrast": contrast, "scope": "all", "model": k, "minus": ref, **d.loc[k].to_dict()})

    # per-rubric screening
    lookup_only = C.oof_predict(fd, [], set(), extra=L)
    incr_preds = {"lookup only": lookup_only}
    for m in METRICS:
        incr_preds[m] = C.oof_predict(fd, [f"score_{m}"], set(), extra=L)
    d_ru = C.paired_boot({k: v[ru] for k, v in incr_preds.items()}, y[ru], ru_rows, {"AUROC": C.auroc, "AP": C.average_precision}, ref="lookup only")
    d_all = C.paired_boot(incr_preds, y, fd["doc_rows"], {"AUROC": C.auroc, "AP": C.average_precision}, ref="lookup only")
    for m in METRICS:
        s = sub[f"score_{m}"].to_numpy()
        wm = C.loo_word_mean(sub, f"score_{m}")
        # residual against the LEAVE-ONE-OUT mean (red-team 1): with the plain mean the residual is
        # exactly 0 for singleton words and shrunk by 1 - 1/n for the rest.
        resid = s - wm
        screen_rows.append({"contrast": contrast, "rubric": m,
                            "AUROC alone": C.auroc(s, y), "AUROC alone ru": C.auroc(s[ru], y[ru]),
                            "AUROC LOO word-mean": C.auroc(wm, y), "AUROC residual": C.auroc(resid, y),
                            "AUROC residual ru": C.auroc(resid[ru], y[ru]),
                            "incr over lookup dAUROC ru": d_ru.loc[m, "dAUROC"], "incr lo ru": d_ru.loc[m, "dAUROC lo"], "incr hi ru": d_ru.loc[m, "dAUROC hi"],
                            "incr over lookup dAP ru": d_ru.loc[m, "dAP"],
                            "incr over lookup dAUROC all": d_all.loc[m, "dAUROC"], "incr lo all": d_all.loc[m, "dAUROC lo"], "incr hi all": d_all.loc[m, "dAUROC hi"]})
    print(f"done {contrast}")

models = pd.DataFrame(model_rows)
screen = pd.DataFrame(screen_rows)
deltas = pd.DataFrame(delta_rows)
models.to_csv(C.OUT_DIR / "item1_models.csv", index=False)
screen.to_csv(C.OUT_DIR / "item1_rubric_screen.csv", index=False)
deltas.to_csv(C.OUT_DIR / "item1_deltas.csv", index=False)
pd.set_option("display.width", 250)
print(models.round(3).to_string())
print(deltas.round(4).to_string())
print(screen.round(3).to_string())

# ---- figure 1: models on random_unseen
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
order = ["word lookup", "positional (length, position, first token)", "lookup + positional", "best single rubric",
         "rubrics main", "rubrics cross", "lookup + rubrics main", "lookup + rubrics cross", "lookup + positional + rubrics cross"]
for ax, contrast in zip(axes, C.FIT_CONTRASTS):
    ci, base = fig_models[contrast]
    ys = np.arange(len(order))[::-1]
    vals = ci.loc[order, "AUROC"].to_numpy()
    ax.barh(ys, vals, color=[P.MODEL_COLOR[o] for o in order], height=0.6,
            xerr=P.errbars(vals, ci.loc[order, "AUROC lo"], ci.loc[order, "AUROC hi"]), capsize=2, error_kw={"linewidth": 1})
    ax.axvline(0.5, **P.CHANCE_KW)
    ax.axvline(ci.loc["word lookup", "AUROC"], color=P.OI["black"], linewidth=1, linestyle=":")
    ax.set_yticks(ys, order)
    ax.set_xlim(0.45, 0.95)
    ax.set_xlabel("AUROC on random_unseen (95% document-bootstrap CI)")
    ax.set_title(f"top 2% vs {contrast}: the word lookup (dotted) beats every rubric model", fontsize=9)
plt.tight_layout()
plt.savefig(C.FIG_DIR / "item1_models.png")

# ---- figure 2: per rubric screening
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for ax, contrast in zip(axes, C.FIT_CONTRASTS):
    t = screen[screen["contrast"] == contrast].set_index("rubric").sort_values("AUROC alone")
    ys = np.arange(len(t))
    ax.barh(ys + 0.22, t["AUROC alone"] - 0.5, left=0.5, height=0.2, color=P.OI["sky"], label="alone")
    ax.barh(ys, t["AUROC LOO word-mean"] - 0.5, left=0.5, height=0.2, color=P.OI["black"], label="its word mean (no context)")
    ax.barh(ys - 0.22, t["AUROC residual"] - 0.5, left=0.5, height=0.2, color=P.OI["orange"], label="residual (context part only)")
    ax.axvline(0.5, **P.CHANCE_KW)
    ax.set_yticks(ys, t.index)
    ax.set_xlim(0.45, 0.85)
    ax.set_xlabel("AUROC, all judged tokens in the contrast")
    n_ge = int((t["AUROC LOO word-mean"] >= t["AUROC alone"]).sum())
    ax.set_title(f"top 2% vs {contrast}: the word mean matches or beats the raw score for {n_ge} of {len(t)} rubrics", fontsize=9)
axes[0].legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(C.FIG_DIR / "item1_rubric_screen.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for ax, contrast in zip(axes, C.FIT_CONTRASTS):
    t = screen[screen["contrast"] == contrast].set_index("rubric").sort_values("incr over lookup dAUROC ru")
    ys = np.arange(len(t))
    v = t["incr over lookup dAUROC ru"].to_numpy()
    ax.barh(ys, v, height=0.55, color=[P.RUBRIC_COLOR[r] for r in t.index],
            xerr=P.errbars(v, t["incr lo ru"], t["incr hi ru"]), capsize=2, error_kw={"linewidth": 1})
    ax.axvline(0, **P.CHANCE_KW)
    ax.set_yticks(ys, t.index)
    ax.set_xlabel("ΔAUROC from adding the rubric to the word lookup, random_unseen (paired 95% CI)")
    ax.set_title(f"top 2% vs {contrast}: what each rubric adds beyond the word", fontsize=9)
plt.tight_layout()
plt.savefig(C.FIG_DIR / "item1_increment.png")
print("figures written")
