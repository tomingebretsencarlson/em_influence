"""Squared-term (z²) study with top/bottom twins, mirroring bottom_twins.py for the cross terms.

Fit: top X% vs middle, same folds. (1) per rubric alone: linear vs linear+square. (2) add one z_j²
to the main model and to the cross model. (3) drop one z_j² from main+all squares and from
cross+all squares; unique = minus that. Every delta is AUROC(alt) − AUROC(base), paired document
bootstrap, on the TOP twin and the BOTTOM twin (positives = bottom X%, scored by the same fold
models), plus a paired CI on (bottom − top). Reuses oof_apply / Twins / bottom_rows.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bottom_twins import Twins, bottom_rows, oof_apply, fit_logit, _flat, _label, auroc, N_FOLDS, auroc_stratified

BASES = ["main", "cross"]


def _quadratic_fits(fd: dict, col: str):
    """Full-data fit b0 + b1 z + b2 z² for one rubric plus the five training-fold fits.
    Returns b1, b2, turning point in judge-score units (NaN unless inside 0-9), b2 range over folds,
    and whether the quadratic is monotone on 0-9 in every fold (then its within-fold AUROC equals the raw score's)."""
    s, y, fold = fd["sub"][col].to_numpy(), fd["is_pos"].astype(float), fd["fold"]
    def fit(mask):
        mu, sd = s[mask].mean(), s[mask].std(); sd = sd if sd else 1.0
        z = (s[mask] - mu) / sd
        b = fit_logit(np.column_stack([np.ones(len(z)), z, z * z]), y[mask])
        turn = mu + (-b[1] / (2 * b[2])) * sd if b[2] != 0 else np.nan
        return b[1], b[2], turn
    b1, b2, turn = fit(np.ones(len(s), bool))
    folds = [fit(fold != k) for k in range(N_FOLDS)]
    b2s = [f[1] for f in folds]
    monotone = all(not (0 <= f[2] <= 9) for f in folds)
    return b1, b2, (turn if 0 <= turn <= 9 else np.nan), min(b2s), max(b2s), monotone


def run(df, fd, metrics, x=2, n_boot=200, out_dir: Path | None = None, pred_full=None, ref_main_sq=None, verbose=True) -> dict:
    """All square-term tables. pred_full: the notebook's PRED_FULL (TOP main/cross asserted).
    ref_main_sq: an independently computed out-of-fold main+all-squares prediction to assert against."""
    cols = [f"score_{m}" for m in metrics]
    n = len(metrics)
    pair_idx = list(combinations(range(n), 2))
    all_sq = list(range(n))
    bottom, app_fold = bottom_rows(df, fd, x)
    tw = Twins(fd, bottom, app_fold)
    log = (lambda *a, **k: print(*a, **k, flush=True)) if verbose else (lambda *a, **k: None)

    base = {"main": oof_apply(fd, cols, [], bottom, app_fold), "cross": oof_apply(fd, cols, pair_idx, bottom, app_fold)}
    base_sq = {"main": oof_apply(fd, cols, [], bottom, app_fold, squares=all_sq),
               "cross": oof_apply(fd, cols, pair_idx, bottom, app_fold, squares=all_sq)}
    if pred_full is not None:
        assert np.allclose(base["main"][0], pred_full[("middle", False)]) and np.allclose(base["cross"][0], pred_full[("middle", True)])
        log("self-check: main and cross fits reproduce PRED_FULL")
    if ref_main_sq is not None:
        assert np.allclose(base_sq["main"][0], ref_main_sq), "main + z² must reproduce the reference out-of-fold fit"
        log("self-check: main + all squares reproduces the reference fit")

    # block deltas: all squares at once, for context
    block = pd.DataFrame({b: _flat(tw.delta_ci(base[b], base_sq[b], n_boot), "dAUROC ") for b in BASES}).T
    block.index.name = "base"

    # (1) per rubric alone: linear vs linear + square. Fold-stratified AUROC (pooled OOF AUROC of a
    # single-feature fit is a fold artifact, red-team 5 F1) and the paired OOF pseudo-R² delta.
    rows = []
    for i, m in enumerate(metrics):
        lin = oof_apply(fd, [cols[i]], [], bottom, app_fold)
        quad = oof_apply(fd, [cols[i]], [], bottom, app_fold, squares=[0])
        b1, b2, s_star, b2_lo, b2_hi, mono = _quadratic_fits(fd, cols[i])
        res = tw.delta_ci(lin, quad, n_boot, stratified=True, with_r2=True)
        raw_top = auroc_stratified(fd["sub"][cols[i]].to_numpy(), tw.y_top, tw.fold_top)
        raw_bot = auroc_stratified(tw.score_bot(fd["sub"][cols[i]].to_numpy(), bottom[cols[i]].to_numpy()), tw.y_bot, tw.fold_bot)
        lin_top = auroc_stratified(lin[0], tw.y_top, tw.fold_top)
        assert abs(lin_top - raw_top) < 1e-6, "stratified AUROC of the linear fit must equal the raw score's"
        rows.append({"rubric": m, "AUROC raw top (stratified)": raw_top, "AUROC quadratic top (stratified)": lin_top + res["top"][0],
                     "AUROC raw bottom (stratified)": raw_bot, "AUROC quadratic bottom (stratified)": raw_bot + res["bottom"][0],
                     **_flat({k: res[k] for k in ["top", "bottom", "bottom-top"]}, "dAUROC strat "),
                     **_flat({"top": res["top pseudo-R2"]}, "dpseudo-R2 "),
                     "beta1": b1, "beta2": b2, "beta2 fold min": b2_lo, "beta2 fold max": b2_hi,
                     "turning point (score units, NaN if outside 0-9)": s_star, "monotone on 0-9 in all folds": mono})
    singles = pd.DataFrame(rows).set_index("rubric")
    log("per rubric linear vs quadratic: done")

    # (2) add one z_j² to main / cross; (3) drop one z_j² from main+sq / cross+sq
    rows = []
    for b in BASES:
        pr = [] if b == "main" else pair_idx
        for j, m in enumerate(metrics):
            add = tw.delta_ci(base[b], oof_apply(fd, cols, pr, bottom, app_fold, squares=[j]), n_boot)
            dr = tw.delta_ci(base_sq[b], oof_apply(fd, cols, pr, bottom, app_fold, squares=[s for s in all_sq if s != j]), n_boot)
            rows.append({"base": b, "rubric": m, **_flat(add, "add dAUROC "), **_flat(dr, "drop dAUROC "), **_flat(dr, "unique dAUROC ", negate=True)})
        log(f"add-one / drop-one z² on {b}: done")
    sq = pd.DataFrame(rows).set_index(["base", "rubric"])

    out = {"singles": singles, "sq": sq, "block": block, "x": x, "metrics": list(metrics),
           "n_top": int(fd["is_pos"].sum()), "n_middle": int(tw.neg.sum()), "n_bottom": len(bottom)}
    if out_dir is not None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        singles.to_csv(out_dir / f"sq_singles_top{x}.csv")
        sq.to_csv(out_dir / f"sq_add_drop_top{x}.csv")
        block.to_csv(out_dir / f"sq_block_top{x}.csv")
    return out


# ----------------------------------------------------------------------------- figures


def plot_singles(tw: dict, fig_dir=None, show=True):
    """Per rubric: fold-stratified AUROC raw vs quadratic, its paired delta, and (TOP only) the paired pseudo-R² delta."""
    x, t = tw["x"], tw["singles"]
    for twin in ["top", "bottom"]:
        n_ax = 3 if twin == "top" else 2
        fig, axes = plt.subplots(1, n_ax, figsize=(5.6 * n_ax, 4.6), gridspec_kw={"width_ratios": ([1, 0.85, 1.35] if n_ax == 3 else [1, 1.35])})
        ax_r2, ax_d = (axes[1], axes[2]) if n_ax == 3 else (None, axes[1])
        order = t.sort_values(f"AUROC raw {twin} (stratified)").index
        ys = np.arange(len(order))
        col = "tab:red" if twin == "bottom" else "tab:blue"
        axes[0].barh(ys + 0.2, t.loc[order, f"AUROC raw {twin} (stratified)"], height=0.4, color="lightsteelblue", edgecolor="black", label="raw score = linear fit")
        axes[0].barh(ys - 0.2, t.loc[order, f"AUROC quadratic {twin} (stratified)"], height=0.4, color=col, label="quadratic: + b2 z²")
        axes[0].axvline(0.5, color="gray", linestyle="--"); axes[0].set_yticks(ys, order); axes[0].set_xlim(0.45, 0.9)
        axes[0].set_xlabel(f"fold-stratified out-of-fold AUROC, positives = {twin} {x}%")
        axes[0].set_title(f"each rubric alone — evaluated on {twin.upper()} {x}%", fontsize=10); axes[0].legend(fontsize=8, loc="lower right")
        v = t.loc[order, f"dAUROC strat {twin}"].to_numpy()
        ax_d.errorbar(v, ys, xerr=[(v - t.loc[order, f"dAUROC strat {twin} lo"]).clip(0), (t.loc[order, f"dAUROC strat {twin} hi"] - v).clip(0)],
                      fmt="o", color=col, capsize=2, markersize=4)
        ax_d.axvline(0, color="gray", linestyle="--"); ax_d.set_yticks(ys, order)
        ax2 = ax_d.twinx(); ax2.set_ylim(ax_d.get_ylim())
        labs = []
        for m in order:
            r = t.loc[m]
            tp = r["turning point (score units, NaN if outside 0-9)"]
            labs.append(f"b2 {r['beta2']:+.2f} [{r['beta2 fold min']:+.2f}, {r['beta2 fold max']:+.2f}]"
                        + (f", turns at {tp:.1f}" if np.isfinite(tp) else ", monotone on 0–9"))
        ax2.set_yticks(ys, labs); ax2.tick_params(axis="y", labelsize=7, colors="dimgray", length=0); ax2.grid(False)
        ax_d.set_xlabel(f"Δ fold-stratified AUROC, quadratic − linear, positives = {twin} {x}%")
        ax_d.set_title(f"non-monotone gain — evaluated on {twin.upper()} {x}%  (right: b2 [fold range])", fontsize=9)
        if ax_r2 is not None:
            v = t.loc[order, "dpseudo-R2 top"].to_numpy()
            ax_r2.errorbar(v, ys, xerr=[(v - t.loc[order, "dpseudo-R2 top lo"]).clip(0), (t.loc[order, "dpseudo-R2 top hi"] - v).clip(0)],
                           fmt="s", color="black", capsize=2, markersize=4)
            ax_r2.axvline(0, color="gray", linestyle="--"); ax_r2.set_yticks(ys, order)
            ax_r2.set_xlabel("Δ OOF pseudo-R², quadratic − linear")
            ax_r2.set_title("likelihood gain from the square — TOP fit", fontsize=9)
        fig.suptitle(_label(twin, x).replace("fit: top", "single-rubric fits: top"), fontsize=8.5, color="tab:red" if twin == "bottom" else "black")
        plt.tight_layout()
        if fig_dir is not None:
            plt.savefig(Path(fig_dir) / f"sq_singles_{twin}.png", dpi=130)
        plt.show() if show else plt.close(fig)


def plot_add_drop(tw: dict, fig_dir=None, show=True):
    x, t, metrics = tw["x"], tw["sq"], tw["metrics"]
    for twin in ["top", "bottom"]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
        for ax, (col, title, xl) in zip(axes, [
                ("add dAUROC", "add one z² to the base model", "Δ AUROC with − without (positive = square helps)"),
                ("drop dAUROC", f"drop one z² from base + all {len(metrics)} squares", "Δ AUROC without − with (negative = square helps)")]):
            ys = np.arange(len(metrics))[::-1]
            for b, off, color in [("main", 0.18, "tab:orange"), ("cross", -0.18, "tab:red")]:
                tt = t.loc[b].loc[metrics]
                v = tt[f"{col} {twin}"].to_numpy()
                ax.errorbar(v, ys + off, xerr=[(v - tt[f"{col} {twin} lo"]).clip(0), (tt[f"{col} {twin} hi"] - v).clip(0)],
                            fmt="o" if b == "main" else "s", color=color, capsize=2, markersize=4, label=f"base = {b}")
            ax.axvline(0, color="gray", linestyle="--"); ax.set_yticks(ys, metrics)
            ax.set_xlabel(f"{xl}, positives = {twin} {x}%"); ax.set_title(f"{title} — evaluated on {twin.upper()} {x}%", fontsize=10)
        axes[0].legend(fontsize=8)
        fig.suptitle(_label(twin, x), fontsize=8.5, color="tab:red" if twin == "bottom" else "black")
        plt.tight_layout()
        if fig_dir is not None:
            plt.savefig(Path(fig_dir) / f"sq_add_drop_{twin}.png", dpi=130)
        plt.show() if show else plt.close(fig)


def plot_top_vs_bottom(tw: dict, fig_dir=None, show=True):
    """Add-one z² delta on TOP vs on BOTTOM, main base and cross base. Filled = paired CI of (bottom − top) excludes 0."""
    x, t, metrics = tw["x"], tw["sq"], tw["metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, b, color in zip(axes, BASES, ["tab:orange", "tab:red"]):
        tt = t.loc[b].loc[metrics]
        xs, ys = tt["add dAUROC top"].to_numpy(), tt["add dAUROC bottom"].to_numpy()
        off = (tt["add dAUROC bottom-top lo"] > 0) | (tt["add dAUROC bottom-top hi"] < 0)
        ax.errorbar(xs, ys, xerr=[(xs - tt["add dAUROC top lo"]).clip(0), (tt["add dAUROC top hi"] - xs).clip(0)],
                    yerr=[(ys - tt["add dAUROC bottom lo"]).clip(0), (tt["add dAUROC bottom hi"] - ys).clip(0)], fmt="none", ecolor="lightgray", zorder=2)
        ax.scatter(xs[off], ys[off], color=color, s=24, zorder=3, label="(bottom − top) CI excludes 0")
        ax.scatter(xs[~off], ys[~off], facecolors="white", edgecolors=color, s=24, zorder=3, label="(bottom − top) CI includes 0")
        lim_hi, lim_lo = max(xs.max(), ys.max(), 0) * 1.15, min(xs.min(), ys.min(), 0) * 1.15
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="gray", linestyle="--", label="bottom = top (reference line)")
        ax.axhline(0, color="gray", linewidth=0.8); ax.axvline(0, color="gray", linewidth=0.8)
        for k, m in enumerate(metrics):
            ax.annotate(m, (xs[k], ys[k]), fontsize=7, textcoords="offset points", xytext=(4, 2 if k % 2 == 0 else -9))
        ax.set_xlabel(f"Δ AUROC evaluated on TOP {x}% (positive = square helps find top tokens)")
        ax.set_ylabel(f"Δ AUROC evaluated on BOTTOM {x}% (positive = also lifts bottom tokens)")
        ax.set_title(f"add one z² to the {b} model", fontsize=10); ax.legend(fontsize=7.5, loc="upper left")
    fig.suptitle(f"fit: top {x}% vs middle. Does a square that helps find top tokens also lift bottom tokens?  "
                 f"(grey: per-axis paired 95% doc-bootstrap CIs; filled = off the diagonal beyond noise)", fontsize=9.5)
    plt.tight_layout()
    if fig_dir is not None:
        plt.savefig(Path(fig_dir) / "sq_top_vs_bottom.png", dpi=130)
    plt.show() if show else plt.close(fig)


def plot_all(tw: dict, fig_dir=None, show=True):
    if fig_dir is not None:
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
    plot_singles(tw, fig_dir, show)
    plot_add_drop(tw, fig_dir, show)
    plot_top_vs_bottom(tw, fig_dir, show)
