# Red-team 0: common.py and the reproduction check (subagent report, verbatim)

BUGS FOUND (none high; two low-severity caveats, no effect on reported numbers)

1. common.py:184-195 (`oof_predict`, `extra` with `rows`), low. `extra` is indexed as `np.asarray(extra)[rows]`, so it must have full length len(fd["sub"]) even when `rows` is a subset; the docstring says "(n_rows, k)", which is ambiguous. Passing a len(rows)-length array raises IndexError in practice (verified), so it fails loudly rather than misaligning, unless rows happen to be the leading indices. Fix the docstring. Scores/labels/folds themselves are correctly co-indexed by `rows` (verified: rows-subset prediction equals a manual masked fit exactly).

2. common.py:215-227 + item1 usage (`oof_lookup` fed as `extra`), low. Classic stacking caveat: for outer fold k, training rows' lookup feature was estimated on folds != own, which includes fold k's labels, so the coefficient on the lookup is fit on features that saw test labels. I re-ran with a properly nested lookup (inner OOF among the 4 training folds): "lookup only" AUROC 0.8847 vs nested 0.8849; "lookup + rubrics main" 0.8994 vs 0.8999 (random_unseen 0.9051 vs 0.9055). Effect is ~0.0005 and in the conservative direction; worth a one-line note, not a fix.

Everything else checked clean: `oof_lookup` matches a brute-force rate on 200 test rows (train-only stats, train prior for unseen words); `loo_word_mean` matches brute force on 300 rows and uses no labels; standardization is fitted on train folds only and `np.where` version is numerically identical to the notebook's in-place version (incl. sd==0 column); `build_design` column order matches `term_names` for all term sets and for explicit `pairs` (counts and spot-checked columns); spread uses p0..p9 renormalised (row sums are 0.98-1.0; all `sd_` finite) with sqrt(E[x^2]-E[x]^2); `paired_boot` uses one resample per iteration for all models (paired), quantiles per column; `abs_boot` fine. `keep_default_na=False` differs from the notebook (2 tokens "NA" are NaN there) but the join count is identical (105090). Minor: 306 whitespace-only tokens collapse to word "" in the lookup.

NUMBERS (independent code, no common import except (iii)):
(i) top2 vs middle: n = 23939, base rate 0.1391 (3329 positives). Confirmed.
(ii) AUROC overconfidence = 0.7695. Confirmed (0.769).
(iii) harm_potential row 0 (ex 11, pos 136, "."): hand SD = 0.316227, common sd_harm_potential = 0.316227. Match. Also `score` == EV of normalised P to 1e-15.
Folds: both contrasts 1589 docs, each in exactly one fold; docs/fold 318/318/318/318/317; tokens/fold (middle) 5070/4796/4579/4783/4711; positives/fold 684/637/660/645/703; no doc in both train and test of any fold; doc_rows cover every row once, each single-doc.

VERDICT: common.py faithfully reproduces the notebook (data, folds, fits, bootstraps) and is safe to build on; only clarify the `extra` length convention and optionally footnote the negligible stacking caveat.

Action taken: docstring fixed and the stacking caveat recorded in common.oof_predict's docstring.
