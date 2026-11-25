"""Shared code for the analyse_rubric_eval.ipynb additions.

Rebuilds the notebook's `df` and `FIT_DATA` exactly (same files, same join, same sort, same fold
seed), and provides the notebook's metric and logistic-regression functions plus the new features
(judge spread from p0..p9, word lookup, positional baselines). Everything is deterministic.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path("/Users/tarn/Desktop/Work/SPAR2026/Codes/EMinf")
IF_SCORES_PATH = ROOT / "scped_IFS/token_scores_olmo_career.npz"
RUBRIC_EVAL_DIR = ROOT / "scped_IFS/rubric_eval"
MANIFEST_PATH = ROOT / "rubric_batches/t0-2_c40-60_manifest.csv"
NOFIRST_MANIFEST_PATH = ROOT / "rubric_batches/t0-2_c40-60_nofirst_manifest.csv"
OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"

N_FOLDS = 5
X_FIT = 2
FIT_CONTRASTS = ["middle", "rest"]
GROUPS = ["seen", "heldout_b4", "nofirst_only", "random_unseen"]
N_BOOT = 200


# ----------------------------------------------------------------------------- data

def load_df() -> tuple[pd.DataFrame, list[str]]:
    """The notebook's df (cells 2-4 and the sort in cell 7), plus per-token judge spread `sd_<m>`
    (SD of the judge's digit distribution p0..p9) and positional columns."""
    d = np.load(IF_SCORES_PATH)
    if_df = pd.DataFrame({k: d[k] for k in d.files})
    first_row_idx = if_df.groupby("example_idx")["position"].idxmin()
    if_df["is_firstpos"] = False
    if_df.loc[first_row_idx, "is_firstpos"] = True
    if_df["pctile"] = 100 * (1 - if_df["score"].rank(pct=True))
    if_df["reply_idx"] = if_df["position"] - if_df.groupby("example_idx")["position"].transform("min")
    if_df["reply_len"] = if_df.groupby("example_idx")["position"].transform("size")

    metrics = sorted(p.parent.name for p in RUBRIC_EVAL_DIR.glob("*/attributions.csv"))
    wide = None
    for metric in metrics:
        m = pd.read_csv(RUBRIC_EVAL_DIR / metric / "attributions.csv", keep_default_na=False)
        P = m[[f"p{i}" for i in range(10)]].to_numpy(dtype=float)
        P = P / P.sum(1, keepdims=True)
        ev = P @ np.arange(10)
        m[f"sd_{metric}"] = np.sqrt(np.maximum(P @ np.arange(10) ** 2 - ev ** 2, 0))
        m = m[["example_idx", "position", "token_id", "token_text", "score", f"sd_{metric}"]]
        m = m.rename(columns={"score": f"score_{metric}"})
        wide = m if wide is None else wide.merge(m, on=["example_idx", "position", "token_id", "token_text"],
                                                   how="inner", validate="one_to_one")
    df = wide.merge(if_df, on=["example_idx", "position", "token_id"], how="inner")

    manifest = pd.read_csv(MANIFEST_PATH)
    nofirst_manifest = pd.read_csv(NOFIRST_MANIFEST_PATH)
    seen_docs = set(manifest.loc[manifest["batch"].isin([0, 1, 2, 3]), "example_idx"])
    b4_docs = set(manifest.loc[manifest["batch"] == 4, "example_idx"])
    nofirst_only_docs = set(nofirst_manifest["example_idx"]) - seen_docs - b4_docs
    df["group"] = np.select(
        [df["example_idx"].isin(seen_docs), df["example_idx"].isin(b4_docs), df["example_idx"].isin(nofirst_only_docs)],
        ["seen", "heldout_b4", "nofirst_only"], default="random_unseen")
    df["word"] = df["token_text"].str.strip().str.lower()
    df = df.sort_values(["example_idx", "position"]).reset_index(drop=True)
    return df, metrics


def contrast_masks(pctile: np.ndarray, x: float, contrast: str):
    """Notebook cell 12, unchanged."""
    is_top = pctile <= x
    if contrast == "rest":
        other = ~is_top
    elif contrast == "middle":
        other = (pctile > 40) & (pctile <= 60)
    elif contrast == "bottom":
        other = pctile >= 100 - x
    else:
        raise ValueError(contrast)
    keep = is_top | other
    return keep, is_top[keep]


def fit_data(df: pd.DataFrame, contrast: str, x: float = X_FIT, seed: int = 0) -> dict:
    """Notebook cell 18's FIT_DATA bundle: rows in the contrast, labels, document folds."""
    keep, is_pos = contrast_masks(df["pctile"].to_numpy(), x, contrast)
    sub = df[keep].reset_index(drop=True)
    docs = sub["example_idx"].unique()
    fold_of_doc = dict(zip(np.random.default_rng(seed).permutation(docs), np.arange(len(docs)) % N_FOLDS))
    return {"sub": sub, "is_pos": is_pos, "fold": sub["example_idx"].map(fold_of_doc).to_numpy(),
            "doc_rows": list(sub.groupby("example_idx").indices.values()), "contrast": contrast}


# ----------------------------------------------------------------------------- metrics (notebook)

def auroc(scores, is_pos):
    is_pos = np.asarray(is_pos, bool)
    n_pos = is_pos.sum()
    n_neg = len(is_pos) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(scores)
    return (ranks[is_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def average_precision(pred, is_pos):
    is_pos = np.asarray(is_pos, bool)
    order = np.argsort(-pred, kind="stable")
    hits = is_pos[order]
    return (np.cumsum(hits)[hits] / (np.flatnonzero(hits) + 1)).mean()


def r_precision(pred, is_pos):
    is_pos = np.asarray(is_pos, bool)
    k = int(is_pos.sum())
    return is_pos[np.argsort(-pred, kind="stable")[:k]].mean() if k else np.nan


def pseudo_r2(logit, is_pos):
    is_pos = np.asarray(is_pos, float)
    loss = np.mean(np.logaddexp(0, logit) - is_pos * logit)
    b = is_pos.mean()
    return 1 - loss / -(b * np.log(b) + (1 - b) * np.log(1 - b))


def fit_metrics(pred, is_pos, logit=False):
    out = {"AUROC": auroc(pred, is_pos), "AP": average_precision(pred, is_pos), "R-prec": r_precision(pred, is_pos)}
    if logit:
        out["pseudo-R2"] = pseudo_r2(pred, is_pos)
    return out


# ----------------------------------------------------------------------------- logistic regression (notebook)

def fit_logit(X, y, ridge=1e-4, max_iter=100, tol=1e-7):
    """Ridge-penalised logistic regression by Newton/IRLS. Step is halved while the penalised
    log-loss rises (never triggers on well-conditioned fits, so results are unchanged); raises
    if the step does not converge within max_iter."""
    pen_loss = lambda b: (lambda t: np.sum(np.logaddexp(0, t) - y * t))(X @ b) + 0.5 * ridge * (b @ b)
    beta = np.zeros(X.shape[1])
    cur = pen_loss(beta)
    for _ in range(max_iter):
        p = 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30)))
        H = X.T @ (X * (p * (1 - p))[:, None]) + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p) - ridge * beta)
        new = pen_loss(beta + step)
        for _ in range(20):
            if new <= cur + 1e-8 * max(1.0, abs(cur)):
                break
            step /= 2
            new = pen_loss(beta + step)
        else:
            raise RuntimeError("fit_logit: Newton step does not decrease the penalised loss after 20 halvings")
        beta, cur = beta + step, new
        if np.abs(step).max() < tol:
            break
    else:
        raise RuntimeError(f"fit_logit: no convergence in {max_iter} Newton steps (max |step| {np.abs(step).max():.2e})")
    return beta


def standardize(train, test):
    mu, sd = train.mean(0), train.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    return (train - mu) / sd, (test - mu) / sd


def term_names(metrics: list[str], terms: set[str], pairs=None) -> list[str]:
    """Column names of build_design's output, in order (intercept first)."""
    names = ["intercept", *metrics]
    if "sq" in terms:
        names += [f"{m}^2" for m in metrics]
    if "spread" in terms:
        names += [f"sd({m})" for m in metrics]
    if "cross" in terms:
        pairs = list(combinations(range(len(metrics)), 2)) if pairs is None else pairs
        names += [f"{metrics[i]} x {metrics[j]}" for i, j in pairs]
    return names


def build_design(Z, SDz, terms: set[str], pairs=None):
    """Design matrix. Z = standardized scores, SDz = standardized spreads (same columns).
    Column order: 1, Z, [Z^2], [SDz], [Z_i*Z_j for pairs]. Matches the notebook's design()
    when terms == {"cross"} (products are not re-standardized, as in the notebook)."""
    cols = [np.ones(len(Z)), *Z.T]
    if "sq" in terms:
        cols += list((Z ** 2).T)
    if "spread" in terms:
        cols += list(SDz.T)
    if "cross" in terms:
        pairs = list(combinations(range(Z.shape[1]), 2)) if pairs is None else pairs
        cols += [Z[:, i] * Z[:, j] for i, j in pairs]
    return np.column_stack(cols)


def oof_predict(fd: dict, score_cols: list[str], terms: set[str], rows=None, pairs=None,
                extra: np.ndarray | None = None) -> np.ndarray:
    """Out-of-fold log-odds. Standardization (of scores, spreads and any `extra` raw columns) is
    fitted on each training fold. `extra` = already-out-of-fold raw feature columns of shape
    (len(fd["sub"]), k), i.e. full length even when `rows` is a subset (it is indexed by `rows`
    here), appended as linear terms. `pairs` restricts the cross products to the given index
    pairs (indices into score_cols).

    Stacking caveat (red-team 0): when `extra` is an out-of-fold lookup, a training row's lookup
    value was estimated on folds other than its own, which include the current test fold's
    labels. A fully nested lookup changes AUROC by about 0.0005 in the conservative direction."""
    rows = np.arange(len(fd["sub"])) if rows is None else np.asarray(rows)
    S = fd["sub"][score_cols].to_numpy()[rows]
    SD = fd["sub"][[c.replace("score_", "sd_") for c in score_cols]].to_numpy()[rows] if "spread" in terms else None
    y = fd["is_pos"][rows].astype(float)
    fold = fd["fold"][rows]
    E = None if extra is None else np.asarray(extra)[rows]
    pred = np.empty(len(rows))
    for k in range(N_FOLDS):
        tr, te = fold != k, fold == k
        Ztr, Zte = standardize(S[tr], S[te])
        if SD is not None:
            Dtr, Dte = standardize(SD[tr], SD[te])
        else:
            Dtr = Dte = None
        Xtr, Xte = build_design(Ztr, Dtr, terms, pairs), build_design(Zte, Dte, terms, pairs)
        if E is not None:
            Etr, Ete = standardize(E[tr], E[te])
            Xtr, Xte = np.column_stack([Xtr, Etr]), np.column_stack([Xte, Ete])
        beta = fit_logit(Xtr, y[tr])
        pred[te] = Xte @ beta
    return pred


# ----------------------------------------------------------------------------- new features

def oof_lookup(fd: dict, alpha: float = 5.0, key: str = "word") -> np.ndarray:
    """Out-of-fold smoothed P(positive | word): for each fold, the rate is estimated on the other
    folds' tokens only (same document folds as the logistic fits), shrunk toward the training-fold
    base rate with pseudo-count alpha. Words unseen in training get the base rate."""
    sub, y, fold = fd["sub"], fd["is_pos"], fd["fold"]
    out = np.empty(len(sub))
    for k in range(N_FOLDS):
        tr, te = fold != k, fold == k
        prior = y[tr].mean()
        stats = pd.DataFrame({"k": sub[key][tr].to_numpy(), "y": y[tr]}).groupby("k")["y"].agg(["sum", "count"])
        rate = (stats["sum"] + alpha * prior) / (stats["count"] + alpha)
        out[te] = sub[key][te].map(rate).fillna(prior).to_numpy()
    return out


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def loo_word_mean(sub: pd.DataFrame, col: str, key: str = "word") -> np.ndarray:
    """Leave-one-out mean of `col` over the other occurrences of the same word (global mean if
    the word occurs once). Uses only judge scores, never labels."""
    s = sub[col]
    g = s.groupby(sub[key])
    cnt, tot = g.transform("size"), g.transform("sum")
    return ((tot - s) / (cnt - 1)).where(cnt > 1, s.mean()).to_numpy()


# ----------------------------------------------------------------------------- bootstrap

def doc_bootstrap_indices(doc_rows, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield np.concatenate([doc_rows[i] for i in rng.integers(0, len(doc_rows), len(doc_rows))])


def paired_boot(preds: dict, is_pos, doc_rows, metrics: dict, ref: str, n_boot=N_BOOT, seed=0) -> pd.DataFrame:
    """Point estimate and 95% percentile CI of metric(model) - metric(ref) for every model in
    `preds`, resampling documents and evaluating both models on the same resample."""
    is_pos = np.asarray(is_pos, bool)
    point = {n: {k: m(p, is_pos) - m(preds[ref], is_pos) for k, m in metrics.items()} for n, p in preds.items()}
    boots = {n: [] for n in preds}
    for r in doc_bootstrap_indices(doc_rows, n_boot, seed):
        base = {k: m(preds[ref][r], is_pos[r]) for k, m in metrics.items()}
        for n, p in preds.items():
            boots[n].append({k: m(p[r], is_pos[r]) - base[k] for k, m in metrics.items()})
    rows = []
    for n in preds:
        b = pd.DataFrame(boots[n])
        row = {"model": n}
        for k in metrics:
            row[f"d{k}"] = point[n][k]
            row[f"d{k} lo"] = b[k].quantile(0.025)
            row[f"d{k} hi"] = b[k].quantile(0.975)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def abs_boot(preds: dict, is_pos, doc_rows, metrics: dict, n_boot=N_BOOT, seed=0) -> pd.DataFrame:
    """Point estimate and 95% CI of each metric for each model (predictions held fixed)."""
    is_pos = np.asarray(is_pos, bool)
    boots = {n: [] for n in preds}
    for r in doc_bootstrap_indices(doc_rows, n_boot, seed):
        for n, p in preds.items():
            boots[n].append({k: m(p[r], is_pos[r]) for k, m in metrics.items()})
    rows = []
    for n, p in preds.items():
        b = pd.DataFrame(boots[n])
        row = {"model": n}
        for k, m in metrics.items():
            row[k] = m(p, is_pos)
            row[f"{k} lo"] = b[k].quantile(0.025)
            row[f"{k} hi"] = b[k].quantile(0.975)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


# ----------------------------------------------------------------------------- apply fitted fold models to other rows

def fold_of_rows(fd: dict, example_idx: np.ndarray, seed: int = 1) -> np.ndarray:
    """Fold id for arbitrary rows: documents already in fd keep their fold; documents absent from
    the contrast's fit set are assigned a fold with a separate rng (seed=1), so existing folds and
    every existing number are untouched."""
    known = dict(zip(fd["sub"]["example_idx"].to_numpy(), fd["fold"]))
    new_docs = np.array(sorted(set(example_idx) - set(known)))
    if len(new_docs):
        known.update(zip(np.random.default_rng(seed).permutation(new_docs), np.arange(len(new_docs)) % N_FOLDS))
    return pd.Series(example_idx).map(known).to_numpy()


def oof_predict_apply(fd: dict, score_cols: list[str], terms: set[str], app: pd.DataFrame, app_fold: np.ndarray,
                      extra: np.ndarray | None = None, extra_app: np.ndarray | None = None, pairs=None):
    """Same fits as oof_predict (identical arithmetic, no `rows` subset), but each fold's model also
    scores the rows of `app` (a DataFrame with the same score_/sd_ columns) whose document fold is
    k, i.e. documents the model did not train on. Returns (pred_sub, pred_app)."""
    sub = fd["sub"]
    S, S_app = sub[score_cols].to_numpy(), app[score_cols].to_numpy()
    sd_cols = [c.replace("score_", "sd_") for c in score_cols]
    SD, SD_app = (sub[sd_cols].to_numpy(), app[sd_cols].to_numpy()) if "spread" in terms else (None, None)
    y, fold = fd["is_pos"].astype(float), fd["fold"]
    pred, pred_app = np.empty(len(sub)), np.full(len(app), np.nan)
    for k in range(N_FOLDS):
        tr, te, ta = fold != k, fold == k, app_fold == k
        mu, sd = S[tr].mean(0), S[tr].std(0); sd = np.where(sd == 0, 1.0, sd)
        Ztr, Zte, Zta = (S[tr] - mu) / sd, (S[te] - mu) / sd, (S_app[ta] - mu) / sd
        if SD is not None:
            mu2, sd2 = SD[tr].mean(0), SD[tr].std(0); sd2 = np.where(sd2 == 0, 1.0, sd2)
            Dtr, Dte, Dta = (SD[tr] - mu2) / sd2, (SD[te] - mu2) / sd2, (SD_app[ta] - mu2) / sd2
        else:
            Dtr = Dte = Dta = None
        Xtr, Xte, Xta = (build_design(Ztr, Dtr, terms, pairs), build_design(Zte, Dte, terms, pairs),
                         build_design(Zta, Dta, terms, pairs))
        if extra is not None:
            E, E_app = np.asarray(extra), np.asarray(extra_app)
            mu3, sd3 = E[tr].mean(0), E[tr].std(0); sd3 = np.where(sd3 == 0, 1.0, sd3)
            Xtr = np.column_stack([Xtr, (E[tr] - mu3) / sd3])
            Xte = np.column_stack([Xte, (E[te] - mu3) / sd3])
            Xta = np.column_stack([Xta, (E_app[ta] - mu3) / sd3])
        beta = fit_logit(Xtr, y[tr])
        pred[te], pred_app[ta] = Xte @ beta, Xta @ beta
    return pred, pred_app


def oof_lookup_apply(fd: dict, app: pd.DataFrame, app_fold: np.ndarray, alpha: float = 5.0, key: str = "word",
                     prior: str = "base"):
    """oof_lookup for fd's rows plus the same per-fold tables applied to `app` rows by their fold.
    prior="base": shrink toward the training base rate (as oof_lookup). prior="zero": shrink toward 0,
    so a word the training folds never saw as positive scores 0 instead of the base rate (unseen
    and unusual words are then not rewarded). Returns (lookup_sub, lookup_app)."""
    sub, y, fold = fd["sub"], fd["is_pos"], fd["fold"]
    out, out_app = np.empty(len(sub)), np.full(len(app), np.nan)
    for k in range(N_FOLDS):
        tr, te, ta = fold != k, fold == k, app_fold == k
        p0 = y[tr].mean() if prior == "base" else 0.0
        stats = pd.DataFrame({"k": sub[key][tr].to_numpy(), "y": y[tr]}).groupby("k")["y"].agg(["sum", "count"])
        rate = (stats["sum"] + alpha * p0) / (stats["count"] + alpha)
        out[te] = sub[key][te].map(rate).fillna(p0).to_numpy()
        out_app[ta] = app[key][ta].map(rate).fillna(p0).to_numpy()
    return out, out_app
