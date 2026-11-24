# Red-team 4: bottom_twins.py / run_bottom_twins.py (subagent report, verbatim)

Scratch scripts (all re-runnable, read-only against the project):
- `/private/tmp/claude-501/-Users-tarn-Desktop-Work-SPAR2026-Codes/47943c04-f263-4284-84d5-0879ab9b5ab5/scratchpad/redteam_twins/check1.py` (fold-apply, bottom_rows, Twins construction, notebook-verbatim bootstrap comparison)
- `.../redteam_twins/check2.py` (fully independent recomputation with L-BFGS fit + sort-based AUROC; `run()` self-check path; per-group numbers)
- `.../redteam_twins/check3.py` (200-draw CSVs vs notebook CSVs for all 36 terms; reference percentile bands vs middle)

Nothing in the project was modified; no packages installed; the 200-draw background run (PID 11285) was left alone and finished at 20:23:55 (log `bottom_twins_run.log`, 14.7 KB, no tracebacks, all self-checks printed).

## Summary

**Code bugs found: no.** Every numerical check (fold assignment, leakage, standardization, pred_sub identity with the notebook, pred_app vs an independent optimizer, group construction, bootstrap pairing, sign conventions, TOP CIs vs the notebook's saved CSVs) passed, most of them bitwise.

**Design / interpretation findings: 1 medium, 4 low.** The medium one is substantive: the BOTTOM twin has no reference band, and every non-middle band of the corpus scores well above 0.5 against the middle band with the same model (5-10 band: 0.723 vs bottom: 0.720), so "a term lifts bottom tokens" cannot currently be separated from "a term tracks distance from the middle band".

Severity counts: high 0, medium 1, low 4.

## Findings

### F1 (medium, design) — The BOTTOM twin has no reference band, so "magnitude vs signed" is under-identified
`bottom_twins.py:98-152` (Twins), notebook cell 31 markdown ("A term that helps on both twins is a magnitude term ... a term that helps on top but not on bottom is signed").

**What.** The bottom twin's positives are the bottom 2%, negatives are the 40-60 band. But the 40-60 band is the trough of the judge scores (the model was fit against it), so *any* non-middle slice of the corpus scores above chance against it. I scored every token in `df` with the fold model that excluded its document (`check3.py`), positives = a percentile band, negatives = the same middle band:

| band (pctile) | n | main AUROC | cross AUROC | overconfidence raw | drop(directive_onset) main |
|---|---|---|---|---|---|
| top 0-2 (fit pos) | 3329 | 0.825 | 0.840 | 0.769 | -0.0372 |
| 2-5 | 3319 | 0.766 | 0.771 | 0.729 | -0.0255 |
| 5-10 | 5293 | 0.723 | 0.726 | 0.693 | -0.0178 |
| 10-40 | 30471 | 0.633 | 0.634 | 0.619 | -0.0110 |
| 60-90 | 30678 | 0.595 | 0.602 | 0.585 | -0.0160 |
| 90-95 | 5422 | 0.672 | 0.679 | 0.630 | -0.0351 |
| 95-98 | 3373 | 0.695 | 0.701 | 0.668 | -0.0246 |
| **bottom 98-100** | 2595 | **0.720** | **0.728** | 0.709 | **-0.0178** |

The bottom twin's 0.720 is indistinguishable from the 5-10 band's 0.723 and is only modestly above the 95-98 band's 0.695; even the 60-90 band is at 0.595. Judge scores are U-shaped in IF percentile (mean raw scores: top 2.4-5.0, middle 0.6-2.8, bottom 1.3-4.2 across rubrics), and the drop-one delta for `directive_onset` is as large on the 90-95 band (-0.035) as on the top (-0.037). So the deltas in the BOTTOM twin measure "does this term move tokens away from the middle-band profile", which is true of every band. Reading "helps on bottom" as "magnitude term" and "helps on top but not bottom" as "signed term" needs a null that the current design does not provide.

**Not a code bug**: the module computes exactly what it says. **Suggested change (judgment call):** add one or two reference twins (e.g. 95-98 and/or 90-95, which are the same size as the bottom) using exactly the same machinery (`oof_apply` + `Twins` already support any `app` frame via `C.fold_of_rows`), and plot bottom deltas against the reference band's deltas rather than only against top. A term is then "magnitude-like" only if its bottom delta exceeds what the neighbouring band gives. Alternatively, define the bottom twin's negatives as the 90-98 band ("extreme negative vs moderately negative") which is the real analogue of top-vs-middle. At minimum, the cell-31 interpretation sentence should be softened.

### F2 (low, design) — Group composition of the positives differs between twins; the effect is small in practice
`bottom_twins.py:62-74`, `run()` line 165-166 log.

Top positives: seen 1665 / heldout_b4 411 / nofirst_only 248 / random_unseen 1005 (50% seen). Bottom positives: 939 / 226 / 207 / 1223 (36% seen, 47% random_unseen). Since the rubrics were authored on the seen docs' top tokens, one could worry the TOP twin carries an optimism the BOTTOM twin does not. Per-group AUROC of the joint main model shows this is not a material confound here: top 0.819 (seen) / 0.824 / 0.807 / 0.834 (random_unseen); bottom 0.713 / 0.693 / 0.698 / 0.730. Both twins are, if anything, best on random_unseen. **Suggested:** mention the composition in the figure caption or report per-group twins once; no code change needed.

### F3 (low, presentation) — `plot_top_vs_bottom` CIs are not CIs on the distance from the diagonal, even though the draws are paired across twins
`bottom_twins.py:319-350`, suptitle line 347 "paired 95% doc-bootstrap CIs in grey", diagonal label line 338 "bottom = top (pure magnitude)".

The x and y error bars are each a paired (with vs without) CI, but the quantity the plot asks the reader to judge is (bottom delta − top delta), for which no CI is shown; the two share the same middle-band negatives, so they are correlated. Verified in `check2.py` that `_boot_draws` happens to draw the *same document multiset* for both twins (both have 1589 groups sorted by doc id, same seed) — so a CI on the difference is available for free by subtracting per-draw, but the module does not compute it. Also the diagonal "pure magnitude" is a heuristic, not a null: the two twins have different base AUROCs (0.825 vs 0.720) and different positive-set feature distributions, so equal AUROC deltas is not what a pure |IF| detector would produce. **Suggested:** compute the per-draw difference inside `delta_ci` (one extra line: `boots_top - boots_bot`) and either show it as a third column in the tables or mark points whose difference CI excludes 0; reword the legend to "bottom = top" without "pure magnitude", and note in the suptitle that the error bars are per-axis, not for the difference. Note also the pairing across twins is accidental (it relies on both twins having the same number of doc groups); if a document ever lacked both middle and top tokens it would silently break.

### F4 (low, table hygiene) — `drop dAUROC {twin} lo/hi` columns are in the drop sign while `unique` is negated
`bottom_twins.py:203-204`, CSV `twins_add_one_top2.csv`.

`unique dAUROC top` = −(drop delta), but the adjacent `drop dAUROC top lo/hi` columns are the un-negated drop CI. Values are correct (verified identical to the notebook's `pair_drop_top2.csv` `dAUROC lo/hi` for all 36 terms), but a reader of the CSV has to flip and swap lo/hi to get a CI on `unique`. **Suggested:** store `unique dAUROC {twin} lo = -hi_drop`, `hi = -lo_drop` instead, or rename the columns to `pair-drop dAUROC`.

### F5 (low, cosmetic/plot) — minor figure issues (none mislabel the twin or the positives)
- `plot_singles` line 247: x-label says "(all judged docs, 95% CI)" for both twins; correct for this dataset (all 1589 judged docs are in the fit set), but the TOP twin's doc set is "fit-set docs" in general.
- `plot_add_one` lines 302-303: `hi = max(a.max(), u.max()) * 1.1` — if all deltas were negative, `hi` would be negative and the dashed diagonal would be drawn backwards. Not triggered here.
- `plot_top_vs_bottom`: annotations overlap (harm_potential / vulnerability; directive_onset x overconfidence / directive_onset x scope_absoluteness). Cosmetic.
- All titles, suptitles, colorbar labels, axis labels in the 7 PNGs correctly state the twin, the positives, the negatives, and the sign convention; the BOTTOM twin is consistently coloured red and its suptitle says "scored by the same fold models". The notebook's heatmap `RdBu_r` orientation (red = helps) is preserved in the add-one twins.

Also noted (not a finding): `bottom_twins_run.log` was 0 bytes for the first ~3 minutes because stdout is block-buffered when redirected; it filled in at exit. `python -u` or `print(..., flush=True)` would make progress visible.

## Verified OK

1. **Fold-apply correctness** (`oof_apply`, lines 77-93): for each fold k, training rows are `fold != k`; standardization `mu, sd` are from `S[tr]` (population std, same as the notebook's `standardize`). `pred_sub` is **bitwise identical** (`np.array_equal` True, max|diff| 0.0) to a verbatim copy of the notebook's `oof_predict_pairs` for main, cross, one-pair (0,1), cross-minus-(2,5), and to `oof_predict` for main/cross with and without `subtlety`. Independent refit of the 5 fold models with scipy L-BFGS-B on the same ridge-penalised log-loss reproduces `pred_app` to max|diff| 3.2e-7 and `pred_sub` to 4.4e-7. For scale, scoring bottom tokens with the *wrong* fold's model changes `pred_app` by up to 0.309 (mean 0.048), so the fold check is sensitive. Zero bottom tokens are scored by a model whose training rows include their document.
2. **bottom_rows** (lines 62-74): 2595 tokens with pctile >= 98 (independent count 2595; pctile min 98.001), in 1152 docs. All 2595 are in documents that are in the fit set (every judged doc has a top or middle token: 1589 docs in `df` = 1589 in `fd`), so the "new docs" rng branch is never exercised here; its logic is the same as `common.fold_of_rows`. Assertion "no bottom token in fit set" holds (independent key check: 0). Fold of every bottom token equals its document's fit fold (0 mismatches). `app_fold` is int64, no NaN, counts [524, 507, 535, 471, 558].
3. **Twins** (lines 98-152): BOTTOM set = 20610 middle (fit rows with `is_pos` False) + 2595 bottom, `y_bot` correct. `doc_rows_bot` has 1589 groups = number of unique docs in the union; every group is a single document; 1151 groups contain both middle and bottom rows, 437 middle-only, 1 bottom-only; groups partition the rows exactly. TOP twin uses `fd["doc_rows"]` by identity and seed 0; `abs_ci` at 200 draws reproduces the notebook's `boot_auroc_ci` lo/hi exactly (joint main 0.818114/0.832791; subtlety 0.735808/0.755619). `delta_ci` reproduces `paired_delta_light` exactly (add-one (0,1): 0.0019923, [0.0011997, 0.0027844]). With/without models are evaluated on the same draw inside each iteration (paired).
4. **Sign conventions** (`run()`, lines 155-215): drop-one = `delta_ci(full, reduced)` = reduced − full (negative = helps); add-one = `delta_ci(main, main+pair)` (positive = helps); unique = −drop from cross model. Self-check path exercised by passing notebook-equivalent `pred_full`, `drop_table` (rebuilt with the notebook's `oof_predict`), and `ADD_ONE`/`PAIR_DROP` from the saved CSVs: all four asserts pass. Example: `directive_onset x subtlety` add +0.0054658, unique +0.0025319 (module) vs notebook add 0.0054658, unique 0.0025319, pair_drop −0.0025319.
5. **Independent recomputation** (`check2.py`, own L-BFGS fit, own sort-based AUROC, own fold map, own band masks, no module code) vs module `run(n_boot=5)`:
   - drop main `directive_onset`: top −0.037229 / bottom −0.017829 (diff 1.5e-8 / 0)
   - drop cross `directive_onset`: −0.034571 / −0.016478 (0 / 1.9e-8)
   - drop main `scope_absoluteness`: −0.000009 / +0.000279 (0 / 0)
   - drop cross `scope_absoluteness`: −0.005188 / +0.000564 (1.5e-8 / 0)
   - add `overconfidence x scope_absoluteness`: +0.003691 / −0.007106; unique +0.001507 / −0.000385 (<=1.9e-8)
   - add `directive_onset x subtlety`: +0.005466 / −0.000290; unique +0.002532 / +0.003696 (<=1.5e-8)
   - joint main AUROC top 0.825299 / bottom 0.719945; cross 0.840440 / 0.727796 (identical).
6. **200-draw outputs vs notebook CSVs** (all 36 cross terms): `add dAUROC top`, `lo`, `hi`, `unique dAUROC top`, `drop dAUROC top lo/hi` all max|diff| = 0.00 against `pair_add_one_top2.csv` / `pair_drop_top2.csv`.
7. **Ties / extrapolation**: judge scores are continuous (105090 unique values per rubric), so tie handling in `auroc` (average ranks) is immaterial; no bottom token has a standardized score beyond the fit set's range (max |z| 4.88 vs 4.89), so cross-term products are not extrapolating.
8. **Figures**: all 7 PNGs (regenerated 20:23:56-58 by the 200-draw run) carry the correct twin, positives, negatives and sign convention in suptitle, title, axis and colorbar labels.

## Verdict

The module is numerically correct: the TOP twin reproduces the notebook bitwise (predictions, point estimates and all 200-draw CIs), the BOTTOM twin scores every bottom token with a model that never saw its document using the training fold's standardization, document groups over the union are right, and the paired bootstrap pairs with/without correctly. The independent recomputation agrees to 1e-8. The one substantive issue is interpretive, not computational: because the middle band is the model's trough, every non-middle band scores well above 0.5 against it (5-10 band 0.723 vs bottom 0.720; even 60-90 gives 0.595), so the current twin cannot distinguish "a term lifts bottom tokens" from "a term moves tokens away from the middle-band profile". I would add a same-size reference band (95-98 or 90-95) as a third twin using the existing `oof_apply`/`Twins` machinery, compute a CI on (bottom − top) per term from the already-paired draws, and soften the "magnitude vs signed" reading in cell 31 until that comparison is in.

---

## Action taken (by the main session, after the report)

- No computational bugs reported; nothing numerical changed.
- F3: `delta_ci` now draws documents once per iteration and applies the draw to both twins, and returns a paired CI on (bottom − top). The scatter marks points whose difference CI excludes 0 (filled) vs includes 0 (hollow); legend no longer says "pure magnitude"; suptitle says the grey bars are per-axis. The cross-twin pairing is now by construction (shared sorted document list, asserted to equal the notebook's doc_rows), not accidental.
- F4: `unique dAUROC {twin} lo/hi` are now stored negated-and-swapped from the drop CI; the raw `drop ... lo/hi` columns are removed.
- F5: `plot_add_one` diagonal guard; annotation offsets alternate; print flushes.
- F1: not implemented (scope change). The notebook's markdown cell now states the caveat with the band table, and a reference-band twin (95-98 or 90-95) is listed as the follow-up decision for the user.
- F2: noted in the notebook markdown; no code change.
