"""Item 3: the notebook's interpretive sections rerun on the reference model (cross + spread + z²),
side by side with the plain cross model they currently use.

 (a) drop-one per rubric: remove its linear term, square, spread and all 8 products; paired deltas.
 (b) drop-one per cross term (36) and per rubric group (its 8 products, linear/square/spread kept).
 (c) full-data coefficient heatmap: contribution spread (beta x SD of the term), fold-sign agreement,
     cross model vs reference model, and the ratio per pair.
AME plots are not rerun (secondary; see report).
"""
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import common as C
import plotting as P

df, METRICS = C.load_df()
SCORE_COLS = [f"score_{m}" for m in METRICS]
SD_COLS = [f"sd_{m}" for m in METRICS]
n = len(METRICS)
PAIRS = list(combinations(range(n), 2))
REF = {"cross", "spread", "sq"}
CROSS = {"cross"}
MODEL_TERMS = {"cross (notebook)": CROSS, "reference": REF}
LIGHT = {"AUROC": C.auroc, "pseudo-R2": C.pseudo_r2}
FULL = {"AUROC": C.auroc, "AP": C.average_precision, "pseudo-R2": C.pseudo_r2}

drop_rubric_rows, drop_pair_rows, coef_rows = [], [], []
HEAT = {}
for contrast in C.FIT_CONTRASTS:
    fd = C.fit_data(df, contrast)
    y = fd["is_pos"]
    base = {name: C.oof_predict(fd, SCORE_COLS, terms) for name, terms in MODEL_TERMS.items()}

    # (a) drop one rubric entirely
    for name, terms in MODEL_TERMS.items():
        preds = {"full": base[name]}
        for m in METRICS:
            preds[m] = C.oof_predict(fd, [c for c in SCORE_COLS if c != f"score_{m}"], terms)
        d = C.paired_boot(preds, y, fd["doc_rows"], FULL, ref="full")
        for m in METRICS:
            drop_rubric_rows.append({"contrast": contrast, "model": name, "dropped": m, **d.loc[m].to_dict()})
        print(f"  {contrast} {name}: drop-rubric done")

    # (b) drop one cross term / one rubric's products
    for name, terms in MODEL_TERMS.items():
        preds = {"full": base[name]}
        for i, j in PAIRS:
            preds[f"pair:{METRICS[i]} x {METRICS[j]}"] = C.oof_predict(fd, SCORE_COLS, terms, pairs=[p for p in PAIRS if p != (i, j)])
        for r in range(n):
            preds[f"group:{METRICS[r]}"] = C.oof_predict(fd, SCORE_COLS, terms, pairs=[p for p in PAIRS if r not in p])
        d = C.paired_boot(preds, y, fd["doc_rows"], LIGHT, ref="full")
        for k in preds:
            if k == "full":
                continue
            kind, term = k.split(":", 1)
            drop_pair_rows.append({"contrast": contrast, "model": name, "kind": kind, "term": term, **d.loc[k].to_dict()})
        print(f"  {contrast} {name}: drop-pair done")

    # (c) full-data coefficients
    S, SD = fd["sub"][SCORE_COLS].to_numpy(), fd["sub"][SD_COLS].to_numpy()
    Z, SDz = C.standardize(S, S)[0], C.standardize(SD, SD)[0]
    yf = y.astype(float)
    HEAT[contrast] = {}
    for name, terms in MODEL_TERMS.items():
        X = C.build_design(Z, SDz, terms)
        names = C.term_names(METRICS, terms)
        beta = C.fit_logit(X, yf)
        fold_betas = np.array([C.fit_logit(X[fd["fold"] != k], yf[fd["fold"] != k]) for k in range(C.N_FOLDS)])
        spread = beta * X.std(0)
        agree = (np.sign(fold_betas) == np.sign(beta)).all(0)
        HEAT[contrast][name] = (names, spread, agree)
        for t, (nm, sp, ag, b) in enumerate(zip(names, spread, agree, beta)):
            if t == 0:
                continue
            coef_rows.append({"contrast": contrast, "model": name, "term": nm, "beta": b, "contribution spread": sp,
                              "same sign all folds": bool(ag)})

drop_rubric = pd.DataFrame(drop_rubric_rows)
drop_pair = pd.DataFrame(drop_pair_rows)
coefs = pd.DataFrame(coef_rows)
drop_rubric.to_csv(C.OUT_DIR / "item3_drop_rubric.csv", index=False)
drop_pair.to_csv(C.OUT_DIR / "item3_drop_pair.csv", index=False)
coefs.to_csv(C.OUT_DIR / "item3_coefficients.csv", index=False)

# pair table: cross vs reference, contribution spread and drop-one delta
pair_rows = []
for contrast in C.FIT_CONTRASTS:
    cc = coefs[(coefs["contrast"] == contrast) & (coefs["model"] == "cross (notebook)")].set_index("term")
    cr = coefs[(coefs["contrast"] == contrast) & (coefs["model"] == "reference")].set_index("term")
    dc = drop_pair[(drop_pair["contrast"] == contrast) & (drop_pair["model"] == "cross (notebook)") & (drop_pair["kind"] == "pair")].set_index("term")
    dr = drop_pair[(drop_pair["contrast"] == contrast) & (drop_pair["model"] == "reference") & (drop_pair["kind"] == "pair")].set_index("term")
    for i, j in PAIRS:
        t = f"{METRICS[i]} x {METRICS[j]}"
        sc, sr = cc.loc[t, "contribution spread"], cr.loc[t, "contribution spread"]
        pair_rows.append({"contrast": contrast, "term": t, "spread cross": sc, "spread reference": sr,
                          "ratio ref/cross": sr / sc if abs(sc) > 1e-9 else np.nan, "same sign": np.sign(sc) == np.sign(sr),
                          "stable cross": cc.loc[t, "same sign all folds"], "stable reference": cr.loc[t, "same sign all folds"],
                          "drop dpseudoR2 cross": dc.loc[t, "dpseudo-R2"], "drop dpseudoR2 ref": dr.loc[t, "dpseudo-R2"],
                          "drop CI excl 0 cross": (dc.loc[t, "dpseudo-R2 hi"] < 0) or (dc.loc[t, "dpseudo-R2 lo"] > 0),
                          "drop CI excl 0 ref": (dr.loc[t, "dpseudo-R2 hi"] < 0) or (dr.loc[t, "dpseudo-R2 lo"] > 0)})
pairs_table = pd.DataFrame(pair_rows)
pairs_table.to_csv(C.OUT_DIR / "item3_pairs_cross_vs_reference.csv", index=False)

pd.set_option("display.width", 250)
for contrast in C.FIT_CONTRASTS:
    print(f"\n=== {contrast}: drop one rubric (negative = rubric helps) ===")
    t = drop_rubric[drop_rubric["contrast"] == contrast].pivot(index="dropped", columns="model", values=["dAUROC", "dAP", "dpseudo-R2"])
    print(t.round(4).to_string())
    print(f"\n=== {contrast}: rubric interaction groups ===")
    print(drop_pair[(drop_pair["contrast"] == contrast) & (drop_pair["kind"] == "group")].pivot(index="term", columns="model", values=["dAUROC", "dpseudo-R2"]).round(4).to_string())
    print(f"\n=== {contrast}: 12 largest cross terms under the notebook's cross model, and what they become in the reference ===")
    pt = pairs_table[pairs_table["contrast"] == contrast].set_index("term")
    print(pt.reindex(pt["spread cross"].abs().sort_values(ascending=False).index).head(12).round(3).to_string())
    print(f"  pairs keeping sign: {pt['same sign'].sum()}/36; stable in all folds: cross {pt['stable cross'].sum()}, reference {pt['stable reference'].sum()}; "
          f"drop-one CI excludes 0: cross {pt['drop CI excl 0 cross'].sum()}, reference {pt['drop CI excl 0 ref'].sum()}; "
          f"median |ratio| {pt['ratio ref/cross'].abs().median():.2f}")

# ---- figure (a): drop-one per rubric, both models
fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharey=True)
for i, contrast in enumerate(C.FIT_CONTRASTS):
    t0 = drop_rubric[drop_rubric["contrast"] == contrast]
    order = t0[t0["model"] == "reference"].set_index("dropped")["dpseudo-R2"].sort_values().index
    ys = np.arange(len(order))
    for j, k in enumerate(["AUROC", "AP", "pseudo-R2"]):
        ax = axes[i, j]
        for off, name, color, mk in [(0.18, "cross (notebook)", P.OI["vermilion"], "s"), (-0.18, "reference", P.OI["green"], "o")]:
            t = t0[t0["model"] == name].set_index("dropped").loc[order]
            v = t[f"d{k}"].to_numpy()
            ax.errorbar(v, ys + off, xerr=P.errbars(v, t[f"d{k} lo"], t[f"d{k} hi"]), fmt=mk, color=color, capsize=2, markersize=4, label=name)
        ax.axvline(0, **P.CHANCE_KW)
        ax.set_yticks(ys, order)
        ax.set_xlabel(f"Δ{k} when the rubric is dropped (negative = helps)")
        ax.set_title(f"top 2% vs {contrast}", fontsize=9)
axes[0, 0].legend(fontsize=8, loc="lower right")
fig.suptitle("Drop one rubric (all its terms): three tiers under both models. directive_onset dominates; a middle tier of five is small but real\n"
             "and reorders between models; safeguard_omission is null, harm_potential and wrongness nearly so", fontsize=10)
plt.tight_layout()
plt.savefig(C.FIG_DIR / "item3_drop_rubric.png")

# ---- figure (b): heatmaps of cross-term contribution spread, cross vs reference, per contrast
for contrast in C.FIT_CONTRASTS:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"wspace": 0.45})
    # colour limit over the cells actually drawn: the 36 products and the 9 linear terms, both models
    drawn = []
    for nm in MODEL_TERMS:
        names_, spread_, _ = HEAT[contrast][nm]
        idx_ = {t: k for k, t in enumerate(names_)}
        drawn += [spread_[idx_[f"{METRICS[i]} x {METRICS[j]}"]] for i, j in PAIRS] + [spread_[idx_[m]] for m in METRICS]
    lim = max(np.nanmax(np.abs(drawn)), 1e-9)
    for ax, name in zip(axes, MODEL_TERMS):
        names, spread, agree = HEAT[contrast][name]
        idx = {nm: t for t, nm in enumerate(names)}
        M = np.full((n, n), np.nan)
        for i, j in PAIRS:
            M[j, i] = spread[idx[f"{METRICS[i]} x {METRICS[j]}"]]
        for i in range(n):
            M[i, i] = spread[idx[METRICS[i]]]
        im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim)
        for i, j in PAIRS:
            ax.text(i, j, f"{M[j, i]:+.2f}" + ("*" if agree[idx[f'{METRICS[i]} x {METRICS[j]}']] else ""), ha="center", va="center", fontsize=6.5)
        for i in range(n):
            ax.text(i, i, f"{M[i, i]:+.2f}" + ("*" if agree[idx[METRICS[i]]] else ""), ha="center", va="center", fontsize=6.5, fontweight="bold")
        ax.set_xticks(range(n), METRICS, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n), METRICS, fontsize=8)
        ax.grid(False)
        ax.set_title(f"{name}: products (lower), linear terms (diagonal)", fontsize=9)
    fig.colorbar(im, ax=axes, label="sign(beta) x SD of the term's log-odds contribution", shrink=0.8)
    fig.suptitle(f"top 2% vs {contrast}: cross-term coefficients under the notebook's model and the reference model (* = same sign in all 5 fold fits)", fontsize=10)
    plt.savefig(C.FIG_DIR / f"item3_heatmap_{contrast}.png", bbox_inches="tight")

# ---- figure (c): per-pair contribution spread, cross vs reference scatter
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, contrast in zip(axes, C.FIT_CONTRASTS):
    pt = pairs_table[pairs_table["contrast"] == contrast]
    ax.scatter(pt["spread cross"], pt["spread reference"], s=22, color=np.where(pt["same sign"], P.OI["black"], P.OI["vermilion"]))
    lim = max(pt["spread cross"].abs().max(), pt["spread reference"].abs().max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], **P.CHANCE_KW)
    ax.plot([-lim, lim], [-lim / 2, lim / 2], color="#bbbbbb", linewidth=0.8, linestyle=":")
    ax.axhline(0, color="#cccccc", linewidth=0.8); ax.axvline(0, color="#cccccc", linewidth=0.8)
    for _, r in pt.reindex(pt["spread cross"].abs().sort_values(ascending=False).index).head(6).iterrows():
        ax.annotate(r["term"], (r["spread cross"], r["spread reference"]), fontsize=6.5, textcoords="offset points", xytext=(4, 2))
    ax.set_xlabel("contribution spread under the cross model")
    ax.set_ylabel("contribution spread under the reference model")
    ax.set_title(f"top 2% vs {contrast} (red = sign flips; dotted = half size)", fontsize=9)
fig.suptitle("Each cross term's weight before and after adding curvature and judge spread", fontsize=10)
plt.tight_layout()
plt.savefig(C.FIG_DIR / "item3_pairs_scatter.png")
print("figures written")
