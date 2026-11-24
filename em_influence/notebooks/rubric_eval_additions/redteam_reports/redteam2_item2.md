# Red-team 2: item2_reference_model.py (subagent report, verbatim)

BUGS FOUND

1. item2_reference_model.py:75 (and suptitle line 89): the delta figure plots only scope=="all" and the headline is stated for "all rows". On the random_unseen subset (the only docs not used in rubric development; 14,213/23,939 middle rows but only 1,005/3,329 positives) the AUROC ordering reverses: cross − (main+spread+z²) = +0.0057 [+0.0004, +0.0120] middle, +0.0099 [+0.0037, +0.0170] rest; reference − cross = +0.003 [−0.002, +0.008] middle, −0.0002 [−0.006, +0.005] rest. On unseen docs, interactions beat curvature+spread and the latter add nothing on top of cross (AUROC; AP still favours spread+z²). Severity: medium (reporting, not computation); "all" is weighted toward seen docs holding 70% of positives.
2. item2_reference_model.py:106: legend(loc="lower right") in item2_nested_abs.png sits on the reference-model row's markers/bars in the left panel. Severity: low, cosmetic.

No computational bugs found.

CHECKS
- z², spread, products are built from training-fold-standardized Z/SD (common.py:204-209); test rows use train mu/sigma. Products/squares not re-standardized; re-standardizing every column on train stats gives identical OOF AUROC/AP to 4 dp for all three models (ridge 1e-4 vs n≈19k per train fold is negligible).
- Convergence (instrumented copy, middle contrast, every fold): reference 7-8 Newton iterations, final step ≤ 4e-8, max|grad| < 1e-12, step norms monotone, max|eta| ≈ 7 (clip never active), max|beta| ≈ 1.9; my beta equals common.fit_logit exactly. cond(H) at convergence: reference 5.3e3-6.7e3, cross ~3e3, main+spread+z² ~1.2e3 — benign. corr(z², spread) per rubric −0.67 (subtlety) to +0.27; no collinearity problem.
- Independent L-BFGS ridge-logistic re-derivation on common.fit_data folds: main+spread+z² 0.8439, cross 0.8404, reference 0.8505 (max |pred diff| ≤ 3e-5 vs pipeline; AP 0.5905/0.5682/0.6052). Values confirmed.
- paired_boot (common.py:263-266) evaluates ref and model on the same resample r; .loc[a] with ref=b gives a − b, matching the model/minus columns. Orientation correct. abs_boot uses the same seed.
- PNGs match CSVs (e.g. −0.0034 [−0.0079, +0.0006]); labels legible; bars unclipped.
- Extra comparison (full-data middle, cross vs reference): top-5 cross coefs in cross model: directive_onset×subtlety −0.512, scope_abs×subtlety +0.343, overconf×subtlety −0.323, overconf×scope_abs +0.297, subtlety×wrong_content −0.270; in reference: −0.351, +0.232, −0.263, +0.180, +0.122 (ratios 0.45-0.81); sum beta² over cross terms 1.00 → 0.63; coef correlation 0.86; harm_potential×wrongness grows from −0.045 to +0.249 (unmasked). Linear R² of each product on [1,z,z²,spread]: median 0.72 (0.26-0.91), but z² columns are equally predictable from products (R² 0.70-0.93): the overlap is symmetric. The headline interaction directive_onset×subtlety is the least explained by curvature (R² 0.26) and shrinks least among big terms.

VERDICT
(A) Stands with caveat: true on all rows; on random_unseen docs cross beats main+spread+z² on AUROC (CI excludes 0). AP claim stands on both scopes.
(B) Stands (all rows +0.0066 [+0.0046, +0.0091], +0.0081 [+0.0058, +0.0105]; ≈0.42-0.44 of cross-over-main). On random_unseen the unique interaction signal is larger (+0.009/+0.010) and spread+z² add ~0 on top of cross.
(C) Stands with caveat: cross coefficients shrink ~30-40% when z²/spread are added, but the overlap is symmetric (curvature is equally "proxied" by products), the flagship directive_onset×subtlety term is mostly not a proxy, and on unseen docs interactions carry the signal. "Partly proxies" is defensible; "28 params match 46" alone is not sufficient evidence.

Action taken: the delta figure now shows both scopes (all rows and random_unseen) and its title was rewritten; the legend was moved off the data; the report's conclusion for item 2 carries the caveat.
