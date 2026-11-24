# Red-team 1: item1_baselines.py (subagent report, verbatim)

BUGS FOUND

1. item1_baselines.py:138 (fig-2 title) and claim (C) wording — "word mean matches or beats the rubric" / "8 of 9". Actual: middle 7/9 (overconfidence 0.7545 < 0.7695, subtlety 0.7302 < 0.7454), rest 8/9 (overconfidence). Residual range "0.51–0.59" is middle only; rest spans 0.4875–0.555 (vulnerability below chance). No CI is computed for word-mean vs raw (harm_potential middle differs by 0.0009). Severity: medium (overstatement), code itself fine.
2. item1_baselines.py:86 — residual uses the plain word mean, so residual = (1−1/n)(s − LOO mean): exactly 0 for singleton words (4.4% middle / 1.7% rest of rows) and shrunk by word count. Switching to LOO moves residual AUROC by up to ±0.03 (directive_onset 0.573→0.549, subtlety 0.540→0.572) but the range stays 0.53–0.59 (middle). Severity: low.
3. item1_baselines.py:39 — best single rubric chosen on all rows incl. random_unseen. Choosing on non-ru rows picks the same (overconfidence); effect zero. Severity: low/minor optimism.
4. Note (not a bug): for the middle contrast the LOO word mean is taken over label-selected tokens (top ∪ middle). Using the full-corpus word mean gives higher AUROC (8/9 on middle), so the claim is not inflated by this.

CHECKS

- Lookup leakage: flipping every test-fold label leaves test-fold lookup values unchanged (max diff 0.0, all 5 folds); no document spans folds; my own lookup implementation on the script's folds matches C.oof_lookup exactly. Fully nested (non-stacked) lookup+cross gives 0.9070/0.8121 vs script 0.9064/0.8115 — stacking caveat is conservative.
- subset_doc_rows: 1000 groups, index 0..n−1 covered exactly, each group one document. paired_boot references consistent; raw-logit lookup vs per-fold-refit "lookup only" AUROC differ ≤0.0003.
- random_unseen definition identical to notebook cell 4.
- Alpha sensitivity, lookup AUROC ru (middle / rest): α=0 0.868/0.749; α=1 0.890/0.772; α=5 0.892/0.789; α=50 0.880/0.797; α=200 0.872/0.792. Rubrics cross = 0.850/0.764. Δ(lookup+cross − lookup): middle +0.018/+0.014/+0.012, rest +0.031/+0.022/+0.017 for α=1/5/50. Ranking of (A),(B) unchanged for α∈[1,200]; only the unsmoothed α=0 lookup loses to the rubrics on rest.
- Independent recomputation (own code, own folds; seeds 1–4, 5/10/20-fold): middle ru lookup AUROC 0.8940–0.8975 vs 0.8920 — within 0.01.
- Artifact probe: top-2% is strongly lexical. In-sample lookup AUROC 0.947 (middle)/0.884 (rest); 3329 positives span 996 words, top 200 words hold 69%; "just/without/only/immediately/solely/automatically/tell/ignore" are 93–100% positive in the middle contrast; 60% of middle positives have a word with in-sample rate >0.5. No path for score/pctile into the lookup (key = token_text only). Finding, not bug: the IF signal is largely absolutist/directive vocabulary, which is why a unigram beats the judge.
- Figures: item1_models.png and item1_increment.png match CSVs; error bars present, unclipped (xlims contain all CIs). item1_rubric_screen.png: title overstated (bug 1), no CIs; values match CSV.
- (D): 0.637 / 0.624 matches "~0.64/~0.62".

VERDICT
(A) stands — robust to α≥1; add the caveat that the lookup wins because top-2% is lexically concentrated.
(B) stands.
(C) stands with caveat — 7/9 (middle), 8/9 (rest); residual range is contrast-specific; residual should use LOO mean; no CIs.
(D) stands.

Action taken: residual switched to the LOO word mean; figure title now states the actual count per contrast; item 1 rerun.
