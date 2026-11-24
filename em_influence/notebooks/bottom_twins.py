"""Bottom-2% twins of the notebook's single-rubric, drop-one and add-one plots.

Every fit is the notebook's own: top X% vs middle, logistic regression on the standardized rubric
scores (+ selected pairwise products), out of fold over the notebook's five document folds. The
TOP twin evaluates the usual way (positives = top X%, negatives = the middle band). The BOTTOM
twin scores the bottom-X% tokens with the same fold models (each bottom token is scored by the
model whose training fold excluded its document) and evaluates with positives = bottom X%,
negatives = the same middle band. Metric: AUROC only (a likelihood metric against a label the
model was not fit to is meaningless). All deltas are paired document bootstraps.

Conventions copied from analyse_rubric_eval.ipynb (fit_logit, standardize, design, folds, bootstrap
draws) so that the TOP twin reproduces the notebook's numbers exactly; `run()` asserts this
whenever the notebook's tables are passed in.

Used from the notebook (thin cell) and from run_bottom_twins.py (standalone, for red-teaming).
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib.pyplot as plt

N_FOLDS = 5


# ----------------------------------------------------------------------------- notebook functions, verbatim

def auroc(scores, is_pos):
    is_pos = np.asarray(is_pos, bool)
    n_pos = is_pos.sum()
    n_neg = len(is_pos) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(scores)
    return (ranks[is_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def fit_logit(X, y, ridge=1e-4, max_iter=100, tol=1e-7):
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        p = 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30)))
        H = X.T @ (X * (p * (1 - p))[:, None]) + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p) - ridge * beta)
        beta += step
        if np.abs(step).max() < tol:
            break
    return beta


def standardize(train, test):
    mu, sd = train.mean(0), train.std(0)
    sd[sd == 0] = 1
    return (train - mu) / sd, (test - mu) / sd


# ----------------------------------------------------------------------------- bottom rows and fold-apply fits

def bottom_rows(df: pd.DataFrame, fd: dict, x: float, seed: int = 1):
    """The bottom-X% tokens (global percentile >= 100 - x) with a document fold each. Documents that
    are in the fit set keep their fold; documents absent from it get one from a separate rng, so
    the notebook's folds are untouched. Asserts that no bottom token is inside the fit set."""
    bottom = df[df["pctile"] >= 100 - x].reset_index(drop=True)
    key_sub = set(zip(fd["sub"]["example_idx"], fd["sub"]["position"]))
    assert not any(k in key_sub for k in zip(bottom["example_idx"], bottom["position"])), \
        "bottom tokens must not be in the fit set (use the middle contrast)"
    known = dict(zip(fd["sub"]["example_idx"].to_numpy(), fd["fold"]))
    new_docs = np.array(sorted(set(bottom["example_idx"]) - set(known)))
    if len(new_docs):
        known.update(zip(np.random.default_rng(seed).permutation(new_docs), np.arange(len(new_docs)) % N_FOLDS))
    return bottom, bottom["example_idx"].map(known).to_numpy()


def oof_apply(fd: dict, cols: list[str], pairs: list[tuple[int, int]], app: pd.DataFrame, app_fold: np.ndarray):
    """The notebook's oof_predict_pairs (all linear terms of `cols` + the given products, pairs are
    index pairs into `cols`), additionally scoring `app` rows with the fold model that did not train
    on their document. pairs=[] is the main model; all pairs is the cross model. Returns (pred_sub, pred_app)."""
    S, S_app = fd["sub"][cols].to_numpy(), app[cols].to_numpy()
    y, fold = fd["is_pos"].astype(float), fd["fold"]
    design_pairs = lambda Z: np.column_stack([np.ones(len(Z)), *Z.T, *[Z[:, i] * Z[:, j] for i, j in pairs]])
    pred, pred_app = np.empty(len(S)), np.full(len(app), np.nan)
    for k in range(N_FOLDS):
        tr, te, ta = fold != k, fold == k, app_fold == k
        mu, sd = S[tr].mean(0), S[tr].std(0)
        sd[sd == 0] = 1
        Ztr, Zte, Zta = (S[tr] - mu) / sd, (S[te] - mu) / sd, (S_app[ta] - mu) / sd
        beta = fit_logit(design_pairs(Ztr), y[tr])
        pred[te], pred_app[ta] = design_pairs(Zte) @ beta, design_pairs(Zta) @ beta
    assert not np.isnan(pred_app).any()
    return pred, pred_app


# ----------------------------------------------------------------------------- twin evaluation

class Twins:
    """Holds the two evaluation sets. TOP: the fit rows (positives = top, negatives = middle), with
    the notebook's doc_rows. BOTTOM: middle rows + bottom rows (positives = bottom), with document
    groups over the union. A prediction is passed as (pred_sub, pred_app)."""

    def __init__(self, fd: dict, bottom: pd.DataFrame):
        self.fd = fd
        self.neg = ~fd["is_pos"]
        self.y_top = fd["is_pos"]
        self.doc_rows_top = fd["doc_rows"]
        n_neg, n_bot = int(self.neg.sum()), len(bottom)
        self.y_bot = np.r_[np.zeros(n_neg, bool), np.ones(n_bot, bool)]
        docs_bot = np.r_[fd["sub"]["example_idx"].to_numpy()[self.neg], bottom["example_idx"].to_numpy()]
        self.doc_rows_bot = [np.asarray(v) for v in pd.DataFrame({"d": docs_bot}).groupby("d").indices.values()]
        # One shared document list for the paired draws across twins. The notebook's doc_rows are the
        # fit-set documents in sorted order; as long as every bottom document is also a fit-set
        # document (asserted), drawing document indices from this list reproduces the notebook's
        # draws for the TOP twin exactly, and the same draw indexes the BOTTOM twin's groups.
        docs_top = np.sort(fd["sub"]["example_idx"].unique())
        assert set(docs_bot) <= set(docs_top), "bottom documents outside the fit set: paired draws across twins would not line up"
        self.docs = docs_top
        g_bot = dict(zip(np.sort(np.unique(docs_bot)), self.doc_rows_bot))
        self.rows_bot_by_doc = [g_bot.get(d, np.empty(0, int)) for d in self.docs]
        assert len(self.doc_rows_top) == len(self.docs)

    def score_bot(self, pred_sub, pred_app):
        return np.r_[pred_sub[self.neg], pred_app]

    def point(self, pred_sub, pred_app):
        return {"top": auroc(pred_sub, self.y_top), "bottom": auroc(self.score_bot(pred_sub, pred_app), self.y_bot)}

    def _boot_draws(self, doc_rows, n_boot, seed):
        rng = np.random.default_rng(seed)  # same draw procedure as the notebook's paired_delta_light / boot_auroc_ci
        for _ in range(n_boot):
            yield np.concatenate([doc_rows[i] for i in rng.integers(0, len(doc_rows), len(doc_rows))])

    def _boot_draws_paired(self, n_boot, seed):
        """One document draw per iteration, applied to both twins: (rows_top, rows_bottom)."""
        rng = np.random.default_rng(seed)
        for _ in range(n_boot):
            idx = rng.integers(0, len(self.docs), len(self.docs))
            yield (np.concatenate([self.doc_rows_top[i] for i in idx]),
                   np.concatenate([self.rows_bot_by_doc[i] for i in idx]))

    def abs_ci(self, preds: dict, n_boot: int, seed: int = 0) -> pd.DataFrame:
        """Absolute AUROC with document-bootstrap 95% CI, both twins. preds: name -> (pred_sub, pred_app)."""
        rows = {}
        for twin, y, doc_rows, get in [("top", self.y_top, self.doc_rows_top, lambda p: p[0]),
                                       ("bottom", self.y_bot, self.doc_rows_bot, lambda p: self.score_bot(*p))]:
            S = {k: get(v) for k, v in preds.items()}
            boots = pd.DataFrame([{k: auroc(s[r], y[r]) for k, s in S.items()} for r in self._boot_draws(doc_rows, n_boot, seed)])
            for k, s in S.items():
                rows.setdefault(k, {})
                rows[k] |= {f"{twin} AUROC": auroc(s, y), f"{twin} lo": boots[k].quantile(0.025), f"{twin} hi": boots[k].quantile(0.975)}
        return pd.DataFrame(rows).T

    def delta_ci(self, base, alt, n_boot: int, seed: int = 0) -> dict:
        """AUROC(alt) - AUROC(base), paired document bootstrap, both twins. Notebook sign conventions:
        drop-one passes (base=full, alt=reduced) -> negative = term helps; add-one passes
        (base=main, alt=main+term) -> positive = term helps.
        Also returns "bottom-top": the per-draw difference of the two deltas (same document draw for
        both twins), so the scatter of bottom delta against top delta has a CI on the distance from
        the diagonal (red-team 4, F3)."""
        a_t, b_t = base[0], alt[0]
        a_b, b_b = self.score_bot(*base), self.score_bot(*alt)
        pt_t, pt_b = auroc(b_t, self.y_top) - auroc(a_t, self.y_top), auroc(b_b, self.y_bot) - auroc(a_b, self.y_bot)
        boots = np.array([(auroc(b_t[r], self.y_top[r]) - auroc(a_t[r], self.y_top[r]),
                           auroc(b_b[s], self.y_bot[s]) - auroc(a_b[s], self.y_bot[s]))
                          for r, s in self._boot_draws_paired(n_boot, seed)])
        q = lambda v: (np.quantile(v, 0.025), np.quantile(v, 0.975))
        return {"top": (pt_t, *q(boots[:, 0])), "bottom": (pt_b, *q(boots[:, 1])), "bottom-top": (pt_b - pt_t, *q(boots[:, 1] - boots[:, 0]))}


def _flat(res: dict, prefix: str = "d", negate: bool = False) -> dict:
    """Flatten {twin: (point, lo, hi)} to columns '<prefix><twin>', '... lo', '... hi'. With negate the
    sign is flipped and lo/hi swapped, so a drop-one result becomes a 'unique' result with a valid CI."""
    out = {}
    for t, (pt, lo, hi) in res.items():
        pt, lo, hi = (-pt, -hi, -lo) if negate else (pt, lo, hi)
        out |= {f"{prefix}{t}": pt, f"{prefix}{t} lo": lo, f"{prefix}{t} hi": hi}
    return out


# ----------------------------------------------------------------------------- main computation

def run(df, fd, metrics, x=2, n_boot=200, out_dir: Path | None = None, pred_full=None, drop_table=None,
        add_one=None, pair_drop=None, verbose=True) -> dict:
    """Compute all twin tables for the middle contrast. If the notebook's PRED_FULL / drop_table /
    ADD_ONE / PAIR_DROP are given, the TOP twin is asserted to reproduce them."""
    cols = [f"score_{m}" for m in metrics]
    n = len(metrics)
    pair_idx = list(combinations(range(n), 2))
    bottom, app_fold = bottom_rows(df, fd, x)
    tw = Twins(fd, bottom)
    log = (lambda *a, **k: print(*a, **k, flush=True)) if verbose else (lambda *a, **k: None)
    log(f"fit: top {x}% vs middle, {len(fd['sub'])} tokens ({int(fd['is_pos'].sum())} top, {int(tw.neg.sum())} middle); "
        f"bottom {x}%: {len(bottom)} tokens in {bottom['example_idx'].nunique()} documents, {int((~bottom['example_idx'].isin(fd['sub']['example_idx'])).sum())} tokens in documents outside the fit set")

    main = oof_apply(fd, cols, [], bottom, app_fold)
    cross = oof_apply(fd, cols, pair_idx, bottom, app_fold)
    if pred_full is not None:
        assert np.allclose(main[0], pred_full[("middle", False)]) and np.allclose(cross[0], pred_full[("middle", True)]), \
            "fold-apply fits must reproduce the notebook's PRED_FULL"
        log("self-check: main and cross fits reproduce PRED_FULL")

    # (1) single rubrics (raw judge scores) + joint models
    preds = {m: (fd["sub"][f"score_{m}"].to_numpy(), bottom[f"score_{m}"].to_numpy()) for m in metrics}
    preds |= {"joint main": main, "joint cross": cross}
    singles = tw.abs_ci(preds, n_boot)
    singles.index.name = "predictor"

    # (2) drop one rubric, main and cross models
    rows = []
    for model, use_pairs in [("main", False), ("cross", True)]:
        full = cross if use_pairs else main
        for r, m in enumerate(metrics):
            keep = [c for c in cols if c != f"score_{m}"]
            kp = list(combinations(range(len(keep)), 2)) if use_pairs else []
            res = tw.delta_ci(full, oof_apply(fd, keep, kp, bottom, app_fold), n_boot)
            rows.append({"model": model, "dropped": m, **_flat(res, "dAUROC ")})
        log(f"drop-one {model}: done")
    drop = pd.DataFrame(rows).set_index(["model", "dropped"])
    if drop_table is not None:
        nb = np.array([drop_table.loc[("middle", mo, m), "dAUROC"] for mo in ["main", "cross"] for m in metrics])
        assert np.allclose(nb, drop["dAUROC top"].to_numpy(), atol=1e-10), "TOP drop-one must reproduce the notebook's drop_table"
        log("self-check: TOP drop-one deltas reproduce drop_table")

    # (3) add one cross term to the main model; unique part = minus drop-one from the cross model
    rows = []
    for i, j in pair_idx:
        term = f"{metrics[i]} x {metrics[j]}"
        add = tw.delta_ci(main, oof_apply(fd, cols, [(i, j)], bottom, app_fold), n_boot)
        dr = tw.delta_ci(cross, oof_apply(fd, cols, [p for p in pair_idx if p != (i, j)], bottom, app_fold), n_boot)
        rows.append({"term": term, **_flat(add, "add dAUROC "), **_flat(dr, "unique dAUROC ", negate=True)})
    add_one_t = pd.DataFrame(rows).set_index("term")
    log("add-one / drop-one pairs: done")
    if add_one is not None:
        nb = add_one["middle"]
        assert np.allclose(nb["add dAUROC"].to_numpy(), add_one_t["add dAUROC top"].to_numpy(), atol=1e-10), "TOP add-one must reproduce ADD_ONE"
        assert np.allclose(nb["unique dAUROC"].to_numpy(), add_one_t["unique dAUROC top"].to_numpy(), atol=1e-10), "TOP unique must reproduce ADD_ONE"
        log("self-check: TOP add-one and unique deltas reproduce ADD_ONE")
    if pair_drop is not None:
        nb = pair_drop["middle"].loc["pair"]["dAUROC"].to_numpy()
        assert np.allclose(nb, -add_one_t["unique dAUROC top"].to_numpy(), atol=1e-10), "TOP pair drop-one must reproduce PAIR_DROP"

    out = {"singles": singles, "drop": drop, "add_one": add_one_t, "x": x, "metrics": list(metrics),
           "n_top": int(fd["is_pos"].sum()), "n_middle": int(tw.neg.sum()), "n_bottom": len(bottom)}
    if out_dir is not None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        singles.to_csv(out_dir / f"twins_singles_top{x}.csv")
        drop.to_csv(out_dir / f"twins_drop_one_top{x}.csv")
        add_one_t.to_csv(out_dir / f"twins_add_one_top{x}.csv")
    return out


# ----------------------------------------------------------------------------- figures (twin PNGs, each marked)

def _label(twin, x):
    return (f"fit: top {x}% vs middle  |  evaluated on TOP {x}% (positives = top {x}%, negatives = middle band)"
            if twin == "top" else
            f"fit: top {x}% vs middle  |  evaluated on BOTTOM {x}% (positives = bottom {x}%, scored by the same fold models; negatives = middle band)")


def plot_singles(tw: dict, fig_dir: Path | None = None, show=True):
    x, metrics = tw["x"], tw["metrics"]
    for twin in ["top", "bottom"]:
        res = tw["singles"]
        fig, ax = plt.subplots(figsize=(10, 5))
        single = res.loc[metrics].sort_values(f"{twin} AUROC")
        ax.barh(single.index, single[f"{twin} AUROC"], color="lightsteelblue", edgecolor="black",
                xerr=[single[f"{twin} AUROC"] - single[f"{twin} lo"], single[f"{twin} hi"] - single[f"{twin} AUROC"]], capsize=2)
        for name, color in [("joint main", "tab:orange"), ("joint cross", "tab:red")]:
            ax.axvspan(res.loc[name, f"{twin} lo"], res.loc[name, f"{twin} hi"], color=color, alpha=0.2)
            ax.axvline(res.loc[name, f"{twin} AUROC"], color=color, label=f"{name} ({res.loc[name, f'{twin} AUROC']:.3f})")
        ax.axvline(0.5, color="gray", linestyle="--", label="chance")
        ax.set_xlim(0.45, 1)
        ax.set_xlabel(f"AUROC, positives = {twin} {x}%, negatives = middle band (all judged docs, 95% CI)")
        ax.set_title(f"single rubrics vs joint models — evaluated on {twin.upper()} {x}%", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        fig.suptitle(_label(twin, x), fontsize=8.5, color="tab:red" if twin == "bottom" else "black")
        plt.tight_layout()
        if fig_dir is not None:
            plt.savefig(Path(fig_dir) / f"twins_singles_{twin}.png", dpi=130)
        plt.show() if show else plt.close(fig)


def plot_drop_one(tw: dict, fig_dir: Path | None = None, show=True):
    x, metrics = tw["x"], tw["metrics"]
    for twin in ["top", "bottom"]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharex=True, sharey=True)
        for ax, model in zip(axes, ["main", "cross"]):
            t = tw["drop"].loc[model].loc[metrics]
            ys = np.arange(len(t))
            v = t[f"dAUROC {twin}"]
            ax.errorbar(v, ys, xerr=[(v - t[f"dAUROC {twin} lo"]).clip(lower=0), (t[f"dAUROC {twin} hi"] - v).clip(lower=0)],
                        fmt="o", capsize=2, color="tab:red" if twin == "bottom" else "tab:blue")
            ax.axvline(0, color="gray", linestyle="--")
            ax.set_yticks(ys, t.index)
            ax.set_title(f"drop-one, {model} model — evaluated on {twin.upper()} {x}%", fontsize=10)
            ax.set_xlabel(f"Δ AUROC without − with, positives = {twin} {x}% (negative = rubric helps)")
        fig.suptitle(_label(twin, x), fontsize=8.5, color="tab:red" if twin == "bottom" else "black")
        plt.tight_layout()
        if fig_dir is not None:
            plt.savefig(Path(fig_dir) / f"twins_drop_one_{twin}.png", dpi=130)
        plt.show() if show else plt.close(fig)


def plot_add_one(tw: dict, fig_dir: Path | None = None, show=True):
    x, metrics = tw["x"], tw["metrics"]
    n = len(metrics)
    pair_idx = list(combinations(range(n), 2))
    t = tw["add_one"]
    for twin in ["top", "bottom"]:
        fig, row = plt.subplots(1, 2, figsize=(17, 7.5), gridspec_kw={"width_ratios": [1.15, 1]})
        M = np.full((n, n), np.nan)
        sig = np.zeros((n, n), bool)
        for i, j in pair_idx:
            r = t.loc[f"{metrics[i]} x {metrics[j]}"]
            M[j, i] = r[f"add dAUROC {twin}"]
            sig[j, i] = r[f"add dAUROC {twin} lo"] > 0 or r[f"add dAUROC {twin} hi"] < 0
        lim = np.nanmax(np.abs(M))
        im = row[0].imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim)
        for i, j in pair_idx:
            row[0].text(i, j, f"{M[j, i] * 1000:+.1f}" + ("*" if sig[j, i] else ""), ha="center", va="center", fontsize=7)
        row[0].set_xticks(range(n), metrics, rotation=45, ha="right")
        row[0].set_yticks(range(n), metrics)
        fig.colorbar(im, ax=row[0], label=f"Δ AUROC when added to the main model, positives = {twin} {x}% (positive = term helps)")
        row[0].set_title(f"add one cross term, AUROC — evaluated on {twin.upper()} {x}% (values x 1000, * = CI excludes 0)", fontsize=10)
        ax = row[1]
        a, u = t[f"add dAUROC {twin}"], t[f"unique dAUROC {twin}"]
        ax.scatter(a, u, color="tab:red" if twin == "bottom" else "black", s=18)
        hi = max(0, a.max(), u.max()) * 1.1
        lo = min(0, a.min(), u.min()) * 1.1
        ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", label="unique = total (nothing shared)")
        ax.axhline(0, color="gray", linewidth=0.8); ax.axvline(0, color="gray", linewidth=0.8)
        for term, r in t.sort_values(f"add dAUROC {twin}", ascending=False).head(6).iterrows():
            ax.annotate(term, (r[f"add dAUROC {twin}"], r[f"unique dAUROC {twin}"]), fontsize=7, textcoords="offset points", xytext=(4, 2))
        ax.set_xlabel(f"total signal: Δ AUROC when added alone (add-one), positives = {twin} {x}%")
        ax.set_ylabel(f"unique signal: −Δ AUROC when dropped (drop-one), positives = {twin} {x}%")
        ax.set_title(f"total vs unique per cross term — evaluated on {twin.upper()} {x}%", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        fig.suptitle(_label(twin, x), fontsize=8.5, color="tab:red" if twin == "bottom" else "black")
        plt.tight_layout()
        if fig_dir is not None:
            plt.savefig(Path(fig_dir) / f"twins_add_one_{twin}.png", dpi=130)
        plt.show() if show else plt.close(fig)


def plot_top_vs_bottom(tw: dict, fig_dir: Path | None = None, show=True):
    """Each term's TOP delta against its BOTTOM delta. Rubrics: minus the cross-model drop-one delta
    (positive = helps). Cross terms: add-one delta (positive = helps). Grey bars are the per-axis
    paired CIs; a FILLED marker means the paired CI of (bottom − top) excludes 0, i.e. the point is
    off the diagonal by more than the bootstrap noise. The diagonal is a visual reference only: the
    two twins have different base AUROCs, so equal deltas is not a formal null for a magnitude detector."""
    x, metrics = tw["x"], tw["metrics"]
    d = tw["drop"].loc["cross"].loc[metrics]
    a = tw["add_one"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (name, xs, ys, lo_x, hi_x, lo_y, hi_y, dlo, dhi, labels, color) in zip(axes, [
        ("rubrics: −Δ AUROC when dropped from the cross model", -d["dAUROC top"], -d["dAUROC bottom"],
         -d["dAUROC top hi"], -d["dAUROC top lo"], -d["dAUROC bottom hi"], -d["dAUROC bottom lo"],
         -d["dAUROC bottom-top hi"], -d["dAUROC bottom-top lo"], list(d.index), "tab:blue"),
        ("cross terms: Δ AUROC when added to the main model", a["add dAUROC top"], a["add dAUROC bottom"],
         a["add dAUROC top lo"], a["add dAUROC top hi"], a["add dAUROC bottom lo"], a["add dAUROC bottom hi"],
         a["add dAUROC bottom-top lo"], a["add dAUROC bottom-top hi"], list(a.index), "tab:purple")]):
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        off_diag = (np.asarray(dlo, float) > 0) | (np.asarray(dhi, float) < 0)
        ax.errorbar(xs, ys, xerr=[(xs - np.asarray(lo_x)).clip(0), (np.asarray(hi_x) - xs).clip(0)],
                    yerr=[(ys - np.asarray(lo_y)).clip(0), (np.asarray(hi_y) - ys).clip(0)],
                    fmt="none", ecolor="lightgray", capsize=0, zorder=2)
        ax.scatter(xs[off_diag], ys[off_diag], color=color, s=22, zorder=3, label="(bottom − top) CI excludes 0")
        ax.scatter(xs[~off_diag], ys[~off_diag], facecolors="white", edgecolors=color, s=22, zorder=3, label="(bottom − top) CI includes 0")
        lim_hi = max(xs.max(), ys.max(), 0) * 1.15
        lim_lo = min(xs.min(), ys.min(), 0) * 1.15
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="gray", linestyle="--", label="bottom = top (reference line)")
        ax.axhline(0, color="gray", linewidth=0.8); ax.axvline(0, color="gray", linewidth=0.8)
        order = np.argsort(-np.abs(xs))[:8]
        for n_k, k in enumerate(order):
            ax.annotate(labels[k], (xs[k], ys[k]), fontsize=7, textcoords="offset points", xytext=(4, 2 if n_k % 2 == 0 else -9))
        ax.set_xlabel(f"Δ AUROC evaluated on TOP {x}% (positive = helps find top tokens)")
        ax.set_ylabel(f"Δ AUROC evaluated on BOTTOM {x}% (positive = also lifts bottom tokens)")
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=7.5, loc="upper left")
    fig.suptitle(f"fit: top {x}% vs middle. Does a term that helps find top tokens also lift bottom tokens?  "
                 f"(grey: per-axis paired 95% doc-bootstrap CIs; filled = off the diagonal beyond noise)", fontsize=9.5)
    plt.tight_layout()
    if fig_dir is not None:
        plt.savefig(Path(fig_dir) / "twins_top_vs_bottom.png", dpi=130)
    plt.show() if show else plt.close(fig)


def plot_all(tw: dict, fig_dir: Path | None = None, show=True):
    if fig_dir is not None:
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
    plot_singles(tw, fig_dir, show)
    plot_drop_one(tw, fig_dir, show)
    plot_add_one(tw, fig_dir, show)
    plot_top_vs_bottom(tw, fig_dir, show)
