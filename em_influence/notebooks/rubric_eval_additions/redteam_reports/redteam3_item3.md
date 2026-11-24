# Red-team 3: item3_interpret.py (subagent report, verbatim)

BUGS FOUND

1. item3_interpret.py:143 (cosmetic, low): heatmap colour limit `lim` is taken over product columns only (slice is correct for both models: index 10 for cross, 28 for reference; verified). The diagonal linear terms are not included, so cells exceeding lim saturate (middle cross overconfidence +0.60, reference subtlety +0.59 vs lim 0.55). Numbers are printed, so only the colour is clipped.
2. item3_interpret.py:136 and suptitle text (claim wording, low): "the ranking of rubrics is the same under both models" is overstated; see (A) below.
3. item3_interpret.py:162 (cosmetic): suptitle says "* = same sign in all 5 fold fits" but stars are only drawn on products; diagonals have no star although the CSV carries the flag (e.g. rest reference scope_absoluteness linear is fold-unstable, not visible in the figure). Also the right panel's y-tick labels ("wrong_content_onset") abut the left panel's right edge.

No computational bugs.

CHECKS
- (i) Dropping a rubric: oof_predict derives sd_ columns from the 8 score columns, pairs=None gives all C(8,2)=28 products; design width 1+24+28=53 (verified). (ii) `pairs=[p for p in PAIRS if p != (i,j)]` gives 35 products (verified); permuting columns changes the Newton fit by 1e-15. (iii) `M[j,i]` with i<j is the lower triangle; `ax.text(i, j)` places at column i, row j, i.e. the same cell; term_names order (linear, ^2, sd(), x) matches build_design (asserted identical designs). (v) fold_betas use rows of the full-data-standardized X; interpretation-only, not out-of-fold, and 4/5 overlap with the full fit makes "stable" a weak check.
- Recompute (own OOF loop, own doc-bootstrap, seed 123): drop directive_onset, reference, middle: dAUROC = -0.0290, CI [-0.0325, -0.0252]. Script: -0.0290.
- Contribution spread directive_onset x subtlety, middle, own design matrix: cross -0.5507 (beta -0.512), reference -0.3777 (beta -0.351). Script: -0.551 / -0.378.
- Row counts: drop_pair 180 (36 pair + 9 group per contrast x model), coefficients 216 (45 + 63 per contrast), drop_rubric 36, pairs table 72.
- Notebook match (cell 23, cross model, middle): directive_onset -0.0346/-0.0687 vs -0.035/-0.069; scope_absoluteness -0.0052/-0.0206 vs -0.005/-0.021; overconfidence, subtlety, wrong_content_onset, vulnerability, harm_potential, safeguard_omission all match to 3 dp.
- PNG spot checks vs item3_coefficients.csv (middle): cross diagonal directive_onset +0.56 (0.562), directive_onset x subtlety -0.55* (-0.551, stable), overconfidence x scope_absoluteness +0.48* (0.483); reference +0.29 (0.291), -0.38* (-0.378), +0.29* (0.294); subtlety x wrongness -0.08 without star (CSV False). Rest: safeguard_omission x scope_absoluteness +0.15* / -0.02 no star match. Error bars unclipped (errbars uses non-negative lengths, all within axes). Labels legible.

VERDICT
(A) Stands with caveat. directive_onset dominates (-0.035 cross, -0.029 ref on middle). But the middle tier reorders between models (middle dAUROC: cross subtlety > overconfidence > scope > wrong_content; reference scope > wrong_content > subtlety > overconfidence), and wrongness is not "nothing unique": its dAUROC CI excludes 0 on middle under both models (-0.0013 [-0.0023,-0.0005]; -0.0025 [-0.0041,-0.0012]) and dAP too, as does harm_potential on middle. Only safeguard_omission is consistently null. Three-tier picture holds; exact ranking does not.
(B) Stands with caveat. Middle pseudo-R2 ratios ref/cross: overconfidence 0.12, scope 0.29, subtlety 0.26, directive_onset 0.51 (so directive_onset also halves on middle; 0.73 on rest). Exceptions: harm_potential on middle 0.79 (-0.0012 to -0.0010), and vulnerability/harm_potential on rest are zero in both models (sign flips are noise).
(C) Stands with caveat. 30/36 keep sign on both contrasts; median |ratio| 0.71/0.70; directive_onset x subtlety drop-one CI excludes 0 under both models on both contrasts. overconfidence x scope_absoluteness keeps 61% of its weight on middle (loses "most" only on rest, 35%); subtlety x wrongness ratio 0.44 middle / 0.11 rest, and becomes fold-unstable under the reference model.

Action taken: colour limit now includes the diagonal; stars drawn on diagonal terms too; panel spacing widened; the drop-rubric title now says "three tiers, middle tier reorders"; report wording follows the verdicts.
