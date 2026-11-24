"""Do the top-2% detectors also fire on the bottom 2% (most protective tokens)?

Every fit is exactly the existing one (top 2% vs middle, top 2% vs rest; same folds). Each fold's
model additionally scores the bottom-2% tokens in documents it did not train on. Under `middle`
the bottom tokens are never in any fit; under `rest` they are negatives in training.
Evaluated on random_unseen. No bootstrap (quick version).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import common as C
import plotting as P

df, METRICS = C.load_df()
SCORE_COLS = [f"score_{m}" for m in METRICS]
bottom = df[df["pctile"] >= 100 - C.X_FIT].reset_index(drop=True)
ru_bot = (bottom["group"] == "random_unseen").to_numpy()


def positional(frame):
    return np.column_stack([np.log(frame["reply_len"]), np.log1p(frame["reply_idx"]), frame["is_firstpos"].astype(float)])


ORDER = ["word lookup (prior = base rate)", "word lookup (prior = 0)", "positional", "lookup + positional", "lookup (prior = 0) + positional",
         "rubrics main", "rubrics cross", "reference (cross + spread + z²)", "lookup + reference", "lookup (prior = 0) + reference"]
COLORS = {"word lookup (prior = base rate)": P.OI["black"], "word lookup (prior = 0)": P.OI["black"], "positional": P.OI["grey"],
          "lookup + positional": P.OI["grey"], "lookup (prior = 0) + positional": P.OI["grey"], "rubrics main": P.OI["orange"], "rubrics cross": P.OI["vermilion"],
          "reference (cross + spread + z²)": P.OI["green"], "lookup + reference": P.OI["green"], "lookup (prior = 0) + reference": P.OI["green"]}
rows, RES = [], {}
for contrast in C.FIT_CONTRASTS:
    fd = C.fit_data(df, contrast)
    sub, y = fd["sub"], fd["is_pos"]
    app_fold = C.fold_of_rows(fd, bottom["example_idx"].to_numpy())
    L, L_app = C.oof_lookup_apply(fd, bottom, app_fold)
    Lg, Lg_app = C.logit(L)[:, None], C.logit(L_app)[:, None]
    L0, L0_app = C.oof_lookup_apply(fd, bottom, app_fold, prior="zero")       # unseen / never-top words score 0, not the base rate
    Lg0, Lg0_app = C.logit(L0)[:, None], C.logit(L0_app)[:, None]
    # as a FEATURE in a fit the prior-0 lookup enters as the rate itself (not its logit, which is -inf at 0):
    # a word never seen as top is the lowest value and nothing in the fit can lift it. (A never-top flag
    # would let the fit re-learn a base-rate score for such words, i.e. reintroduce the prior.)
    F0, F0_app = L0[:, None], L0_app[:, None]
    POS, POS_app = positional(sub), positional(bottom)
    preds = {
        "word lookup (prior = base rate)": (Lg[:, 0], Lg_app[:, 0]),
        "word lookup (prior = 0)": (Lg0[:, 0], Lg0_app[:, 0]),
        "positional": C.oof_predict_apply(fd, [], set(), bottom, app_fold, extra=POS, extra_app=POS_app),
        "lookup + positional": C.oof_predict_apply(fd, [], set(), bottom, app_fold, extra=np.column_stack([Lg, POS]), extra_app=np.column_stack([Lg_app, POS_app])),
        "lookup (prior = 0) + positional": C.oof_predict_apply(fd, [], set(), bottom, app_fold, extra=np.column_stack([F0, POS]), extra_app=np.column_stack([F0_app, POS_app])),
        "rubrics main": C.oof_predict_apply(fd, SCORE_COLS, set(), bottom, app_fold),
        "rubrics cross": C.oof_predict_apply(fd, SCORE_COLS, {"cross"}, bottom, app_fold),
        "reference (cross + spread + z²)": C.oof_predict_apply(fd, SCORE_COLS, {"cross", "spread", "sq"}, bottom, app_fold),
        "lookup + reference": C.oof_predict_apply(fd, SCORE_COLS, {"cross", "spread", "sq"}, bottom, app_fold, extra=Lg, extra_app=Lg_app),
        "lookup (prior = 0) + reference": C.oof_predict_apply(fd, SCORE_COLS, {"cross", "spread", "sq"}, bottom, app_fold, extra=F0, extra_app=F0_app),
    }
    # self-check 1: the new fit path reproduces oof_predict / oof_lookup exactly on the fit rows
    assert np.allclose(preds["rubrics cross"][0], C.oof_predict(fd, SCORE_COLS, {"cross"}))
    assert np.allclose(L, C.oof_lookup(fd))
    # self-check 2 (rest only): bottom tokens are inside sub; the applied scores must equal their OOF scores
    if contrast == "rest":
        in_sub = sub["pctile"].to_numpy() >= 100 - C.X_FIT
        key_sub = sub.loc[in_sub, ["example_idx", "position"]].apply(tuple, axis=1)
        key_app = bottom[["example_idx", "position"]].apply(tuple, axis=1)
        align = pd.Series(np.flatnonzero(in_sub), index=key_sub).loc[key_app].to_numpy()
        for n, (ps, pa) in preds.items():
            assert np.allclose(ps[align], pa), n
        print("rest: applied bottom scores equal their out-of-fold scores (self-check passed)")
    assert not np.isnan(np.concatenate([pa for _, pa in preds.values()])).any()

    ru = (sub["group"] == "random_unseen").to_numpy()
    is_mid = sub["pctile"].between(40, 60).to_numpy()
    is_neg = ~y
    if contrast == "rest":
        is_neg = is_neg & (sub["pctile"].to_numpy() < 100 - C.X_FIT)   # bottom tokens enter the pool once, via `app`
    n_top, n_bot = int((ru & y).sum()), int(ru_bot.sum())
    RES[contrast] = {}
    for name, (ps, pa) in preds.items():
        s_top, s_mid, s_neg, s_bot = ps[ru & y], ps[ru & is_mid], ps[ru & is_neg], pa[ru_bot]
        pool = np.concatenate([s_top, s_neg, s_bot])
        lab = np.concatenate([np.full(len(s_top), "top"), np.full(len(s_neg), "other"), np.full(len(s_bot), "bottom")])
        sel = lab[np.argsort(-pool)[:n_top]]
        r = {"contrast": contrast, "model": name,
             "AUROC top vs middle": C.auroc(np.r_[s_top, s_mid], np.r_[np.ones(len(s_top)), np.zeros(len(s_mid))]),
             "AUROC bottom vs middle": C.auroc(np.r_[s_bot, s_mid], np.r_[np.ones(len(s_bot)), np.zeros(len(s_mid))]),
             "AUROC top vs bottom": C.auroc(np.r_[s_top, s_bot], np.r_[np.ones(len(s_top)), np.zeros(len(s_bot))]),
             "frac bottom in selected": (sel == "bottom").mean(), "frac top in selected": (sel == "top").mean(),
             "chance frac bottom": n_bot / len(pool), "n selected": n_top, "n bottom ru": n_bot, "n pool": len(pool),
             "mean score top": s_top.mean(), "mean score middle": s_mid.mean(), "mean score bottom": s_bot.mean()}
        rows.append(r); RES[contrast][name] = r
    print(f"done {contrast}")

res = pd.DataFrame(rows)
res.to_csv(C.OUT_DIR / "item_bottom.csv", index=False)
pd.set_option("display.width", 250)
print(res.round(3).to_string())

# ---- per-word: P(top | word) vs P(bottom | word), all judged tokens, words with >= 40 occurrences
top_m, bot_m = df["pctile"] <= C.X_FIT, df["pctile"] >= 100 - C.X_FIT
w = pd.DataFrame({"n": df.groupby("word").size(), "top": df[top_m].groupby("word").size(), "bot": df[bot_m].groupby("word").size()}).fillna(0)
w = w[w["n"] >= 40]
w["p_top"], w["p_bot"] = w["top"] / w["n"], w["bot"] / w["n"]
rho = w[["p_top", "p_bot"]].corr(method="spearman").iloc[0, 1]
w.sort_values("p_top", ascending=False).to_csv(C.OUT_DIR / "item_bottom_words.csv")
print(f"words with n>=40: {len(w)}; Spearman(P(top|word), P(bottom|word)) = {rho:.3f}")
print("top 12 by P(top):"); print(w.sort_values("p_top", ascending=False).head(12).round(3).to_string())
print("top 12 by P(bottom):"); print(w.sort_values("p_bot", ascending=False).head(12).round(3).to_string())

# ---- positional: first-position fraction and relative-position deciles, top vs bottom vs all
rel = (df["reply_idx"] / (df["reply_len"] - 1).clip(lower=1)).to_numpy()
for nm, m in [("all judged", np.ones(len(df), bool)), ("top 2%", top_m.to_numpy()), ("bottom 2%", bot_m.to_numpy())]:
    print(f"{nm:10s}: frac first pos {df.loc[m, 'is_firstpos'].mean():.3f}; rel-pos deciles (share) {np.round(np.histogram(rel[m], bins=10, range=(0, 1))[0] / m.sum(), 3)}")

# ---- figure 1: AUROCs and selected-set composition
fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5), sharey="row")
for j, contrast in enumerate(C.FIT_CONTRASTS):
    t = res[res["contrast"] == contrast].set_index("model").loc[ORDER]
    ys = np.arange(len(ORDER))[::-1]
    ax = axes[0, j]
    ax.plot(t["AUROC top vs middle"], ys + 0.22, "s", color=P.OI["black"], label="top vs middle (what it was fit for)")
    ax.plot(t["AUROC bottom vs middle"], ys, "o", color=P.OI["vermilion"], label="bottom vs middle (does it fire on bottom?)")
    ax.plot(t["AUROC top vs bottom"], ys - 0.22, "^", color=P.OI["blue"], label="top vs bottom (can it tell the tails apart?)")
    ax.axvline(0.5, **P.CHANCE_KW)
    ax.set_yticks(ys, ORDER); ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("AUROC on random_unseen, out of fold")
    ax.set_title(f"fit: top 2% vs {contrast}", fontsize=9)
    ax = axes[1, j]
    v = t["frac bottom in selected"].to_numpy()
    ax.barh(ys, v, color=[COLORS[o] for o in ORDER], height=0.55)
    ax.axvline(t["chance frac bottom"].iloc[0], **P.CHANCE_KW)
    ax.set_yticks(ys, ORDER)
    ax.set_xlabel("share of bottom-2% tokens in the selected set (dashed: chance)")
    ax.set_title(f"fit: top 2% vs {contrast}: top {int(t['n selected'].iloc[0])} of {int(t['n pool'].iloc[0]):,} pooled tokens", fontsize=9)
axes[0, 0].legend(fontsize=7.5, loc="lower left")
fig.suptitle("Do the top-2% detectors also fire on the bottom 2% (most protective) tokens?  Under `middle` the bottom tokens were never in any fit; under `rest` they were negatives.\n"
             "Word lookup with prior = base rate scores words it never saw at the base rate (above most middle tokens); with prior = 0 unseen and never-top words score 0\n"
             "Inside a fit the prior-0 lookup enters as the rate itself, so never-top words stay lowest.", fontsize=9)
plt.tight_layout()
plt.savefig(C.FIG_DIR / "item_bottom_auroc.png")

# ---- figure 2: words
fig, ax = plt.subplots(figsize=(6.2, 5.6))
ax.scatter(w["p_top"], w["p_bot"], s=np.sqrt(w["n"]) * 0.9, color=P.OI["blue"], alpha=0.5, linewidths=0)
lim = max(w["p_top"].max(), w["p_bot"].max()) * 1.05
ax.plot([0, lim], [0, lim], **P.CHANCE_KW)
for _, r in pd.concat([w.sort_values("p_top", ascending=False).head(8), w.sort_values("p_bot", ascending=False).head(8)]).iterrows():
    ax.annotate(repr(_), (r["p_top"], r["p_bot"]), fontsize=6.5, textcoords="offset points", xytext=(3, 2))
ax.set_xlabel("P(top 2% | word)"); ax.set_ylabel("P(bottom 2% | word)")
ax.set_title(f"Per word (n ≥ 40, {len(w)} words, size ∝ √n): Spearman = {rho:.2f}. Dashed: equal rates.", fontsize=9)
plt.tight_layout()
plt.savefig(C.FIG_DIR / "item_bottom_words.png")
print("figures written")
