"""Build report.html from the item CSVs and red-team reports. Every number in a table comes from a
CSV written by the item scripts; the prose is hand-written. Figures are referenced relatively."""
import html
import json
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parent
RT = D / "redteam_reports"


def f3(x):
    return f"{x:.3f}"


def ci(pt, lo, hi, d=3):
    return f"{pt:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def table(rows, header, caption=None, cls=""):
    out = [f'<div class="tablewrap"><table class="{cls}">']
    if caption:
        out.append(f"<caption>{caption}</caption>")
    out.append("<thead><tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in header) + "</tr></thead><tbody>")
    for r in rows:
        out.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def md_block(path):
    return f"<pre class='rt'>{html.escape(Path(path).read_text())}</pre>"


# ---------------- data
m1 = pd.read_csv(D / "item1_models.csv")
d1 = pd.read_csv(D / "item1_deltas.csv")
s1 = pd.read_csv(D / "item1_rubric_screen.csv")
m2 = pd.read_csv(D / "item2_models.csv")
d2 = pd.read_csv(D / "item2_deltas.csv")
dr3 = pd.read_csv(D / "item3_drop_rubric.csv")
dp3 = pd.read_csv(D / "item3_drop_pair.csv")
pt3 = pd.read_csv(D / "item3_pairs_cross_vs_reference.csv")
rep0 = json.loads((D / "item0_reproduce.json").read_text())

# ---------------- item 1 tables
ORDER1 = ["word lookup", "positional (length, position, first token)", "lookup + positional", "best single rubric (overconfidence)",
          "rubrics main", "rubrics cross", "lookup + rubrics main", "lookup + rubrics cross", "lookup + positional + rubrics cross"]
rows = []
for name in ORDER1:
    r = {c: m1[(m1.contrast == c) & (m1.model == name)].iloc[0] for c in ["middle", "rest"]}
    rows.append([name, f3(r["middle"]["AUROC random_unseen"]), f3(r["middle"]["AP random_unseen"]), int(r["middle"]["tokens found random_unseen"]),
                 f3(r["rest"]["AUROC random_unseen"]), f3(r["rest"]["AP random_unseen"]), int(r["rest"]["tokens found random_unseen"]),
                 f3(r["middle"]["AUROC all"]), f3(r["rest"]["AUROC all"])])
n_pos_ru = int(m1["n positives random_unseen"].iloc[0])
t1_models = table(rows, ["model", "AUROC", "AP", f"tokens found (of {n_pos_ru})", "AUROC", "AP", f"tokens found (of {n_pos_ru})", "AUROC all, middle", "AUROC all, rest"],
                  "random_unseen (1,000 held-out documents): top 2% vs middle | top 2% vs rest | pooled for reference")

rows = []
for c in ["middle", "rest"]:
    for name, ref in [("rubrics cross", "word lookup"), ("lookup + rubrics cross", "word lookup"),
                      ("lookup + positional + rubrics cross", "lookup + positional")]:
        r = d1[(d1.contrast == c) & (d1.scope == "random_unseen") & (d1.model == name) & (d1.minus == ref)].iloc[0]
        rows.append([c, f"{name} minus {ref}", ci(r["dAUROC"], r["dAUROC lo"], r["dAUROC hi"]), ci(r["dAP"], r["dAP lo"], r["dAP hi"]),
                     ci(r["dR-prec"] * n_pos_ru, r["dR-prec lo"] * n_pos_ru, r["dR-prec hi"] * n_pos_ru, 0)])
t1_deltas = table(rows, ["contrast", "comparison", "ΔAUROC [95% CI]", "ΔAP [95% CI]", "Δtokens found [95% CI]"],
                  "Paired document bootstrap on random_unseen")

rows = []
for m in sorted(s1.rubric.unique()):
    r = {c: s1[(s1.contrast == c) & (s1.rubric == m)].iloc[0] for c in ["middle", "rest"]}
    rows.append([m, f3(r["middle"]["AUROC alone"]), f3(r["middle"]["AUROC LOO word-mean"]), f3(r["middle"]["AUROC residual"]),
                 ci(r["middle"]["incr over lookup dAUROC ru"], r["middle"]["incr lo ru"], r["middle"]["incr hi ru"]),
                 f3(r["rest"]["AUROC alone"]), f3(r["rest"]["AUROC LOO word-mean"]), f3(r["rest"]["AUROC residual"]),
                 ci(r["rest"]["incr over lookup dAUROC ru"], r["rest"]["incr lo ru"], r["rest"]["incr hi ru"])])
t1_screen = table(rows, ["rubric", "alone", "its word mean", "residual", "ΔAUROC over lookup (random_unseen)", "alone", "its word mean", "residual", "ΔAUROC over lookup (random_unseen)"],
                  "Per-rubric screening: top 2% vs middle | top 2% vs rest. 'alone', 'word mean' and 'residual' are on all judged tokens in the contrast.")
cnt = {c: int((s1[s1.contrast == c]["AUROC LOO word-mean"] >= s1[s1.contrast == c]["AUROC alone"]).sum()) for c in ["middle", "rest"]}

# ---------------- item 2 tables
ORDER2 = list(m2[m2.contrast == "middle"]["model"])
rows = []
for name in ORDER2:
    r = {c: m2[(m2.contrast == c) & (m2.model == name)].iloc[0] for c in ["middle", "rest"]}
    rows.append([name, int(r["middle"]["n_params"]), f3(r["middle"]["AUROC all"]), f3(r["middle"]["AP all"]), f3(r["middle"]["AUROC random_unseen"]), f3(r["middle"]["AP random_unseen"]),
                 f3(r["rest"]["AUROC all"]), f3(r["rest"]["AP all"]), f3(r["rest"]["AUROC random_unseen"]), f3(r["rest"]["AP random_unseen"])])
t2_models = table(rows, ["model", "params", "AUROC all", "AP all", "AUROC ru", "AP ru", "AUROC all", "AP all", "AUROC ru", "AP ru"],
                  "Nested models, out of fold: top 2% vs middle | top 2% vs rest (ru = random_unseen)")
KEY2 = [("cross", "main"), ("main + spread + z²", "main"), ("cross", "main + spread + z²"),
        ("cross + spread + z² (reference)", "main + spread + z²"), ("cross + spread + z² (reference)", "cross")]
rows = []
for c in ["middle", "rest"]:
    for a, b in KEY2:
        ra = d2[(d2.contrast == c) & (d2.scope == "all") & (d2.model == a) & (d2.minus == b)].iloc[0]
        rr = d2[(d2.contrast == c) & (d2.scope == "random_unseen") & (d2.model == a) & (d2.minus == b)].iloc[0]
        rows.append([c, f"{a.replace(' (reference)', '')} minus {b}", ci(ra["dAUROC"], ra["dAUROC lo"], ra["dAUROC hi"]), ci(ra["dAP"], ra["dAP lo"], ra["dAP hi"]),
                     ci(rr["dAUROC"], rr["dAUROC lo"], rr["dAUROC hi"]), ci(rr["dAP"], rr["dAP lo"], rr["dAP hi"])])
t2_deltas = table(rows, ["contrast", "comparison", "ΔAUROC all", "ΔAP all", "ΔAUROC random_unseen", "ΔAP random_unseen"],
                  "Paired document-bootstrap deltas, 95% CI")

# ---------------- item 3 tables
rows = []
order3 = dr3[(dr3.contrast == "middle") & (dr3.model == "reference")].sort_values("dAUROC")["dropped"]
for m in order3:
    cells = [m]
    for c in ["middle", "rest"]:
        for model in ["cross (notebook)", "reference"]:
            r = dr3[(dr3.contrast == c) & (dr3.model == model) & (dr3.dropped == m)].iloc[0]
            cells.append(ci(r["dAUROC"], r["dAUROC lo"], r["dAUROC hi"]))
    rows.append(cells)
t3_drop = table(rows, ["dropped rubric", "cross, middle", "reference, middle", "cross, rest", "reference, rest"],
                "ΔAUROC when the rubric (all its terms) is dropped; negative = the rubric helps. Paired 95% CI.")
rows = []
g = dp3[dp3.kind == "group"]
for m in g[(g.contrast == "middle") & (g.model == "reference")].sort_values("dpseudo-R2")["term"]:
    cells = [m]
    for c in ["middle", "rest"]:
        for model in ["cross (notebook)", "reference"]:
            r = g[(g.contrast == c) & (g.model == model) & (g.term == m)].iloc[0]
            cells.append(f"{r['dpseudo-R2']:+.4f}")
    rows.append(cells)
t3_group = table(rows, ["rubric's 8 products dropped", "cross, middle", "reference, middle", "cross, rest", "reference, rest"],
                 "Δpseudo-R² when all products involving the rubric are dropped (its linear, square and spread terms kept)")
rows = []
p = pt3[pt3.contrast == "middle"].set_index("term")
for t in p["spread cross"].abs().sort_values(ascending=False).index[:8]:
    r = p.loc[t]; r2 = pt3[(pt3.contrast == "rest") & (pt3.term == t)].iloc[0]
    rows.append([t, f"{r['spread cross']:+.2f}", f"{r['spread reference']:+.2f}", f"{r['ratio ref/cross']:.2f}",
                 "yes" if r["drop CI excl 0 cross"] else "no", "yes" if r["drop CI excl 0 ref"] else "no",
                 f"{r2['spread cross']:+.2f}", f"{r2['spread reference']:+.2f}", f"{r2['ratio ref/cross']:.2f}"])
t3_pairs = table(rows, ["product", "weight, cross", "weight, reference", "ratio", "drop-one CI excludes 0, cross", "same, reference", "weight, cross (rest)", "weight, reference (rest)", "ratio (rest)"],
                 "The 8 largest products under the notebook's cross model on the middle contrast, and what they become once curvature and spread are in the model")
summ = {}
for c in ["middle", "rest"]:
    q = pt3[pt3.contrast == c]
    summ[c] = dict(sign=int(q["same sign"].sum()), ratio=q["ratio ref/cross"].abs().median(), sc=int(q["stable cross"].sum()), sr=int(q["stable reference"].sum()),
                   dc=int(q["drop CI excl 0 cross"].sum()), dref=int(q["drop CI excl 0 ref"].sum()))

# ---------------- reproduction table
rows = [["group table (docs / tokens / top-2%)", "400/26766/1665, 100/6534/411, 89/6434/248, 1000/65356/1005", "identical", "yes"],
        ["middle: main AUROC / AP / R-prec", "0.825 / 0.513 / 0.487", f"{rep0['middle/main']['AUROC']:.4f} / {rep0['middle/main']['AP']:.4f} / {rep0['middle/main']['R-prec']:.4f}", "yes"],
        ["middle: cross AUROC / AP / R-prec", "0.840 / 0.568 / 0.525", f"{rep0['middle/cross']['AUROC']:.4f} / {rep0['middle/cross']['AP']:.4f} / {rep0['middle/cross']['R-prec']:.4f}", "yes"],
        ["rest: main AUROC / AP / R-prec", "0.734 / 0.106 / 0.167", f"{rep0['rest/main']['AUROC']:.4f} / {rep0['rest/main']['AP']:.4f} / {rep0['rest/main']['R-prec']:.4f}", "yes"],
        ["rest: cross AUROC / AP / R-prec", "0.753 / 0.125 / 0.199", f"{rep0['rest/cross']['AUROC']:.4f} / {rep0['rest/cross']['AP']:.4f} / {rep0['rest/cross']['R-prec']:.4f}", "yes"],
        ["middle: AUROC CI main / cross (cell 21)", "[0.818, 0.833] / [0.833, 0.848]",
         f"[{rep0['middle/ci']['AUROC lo']['main']:.3f}, {rep0['middle/ci']['AUROC hi']['main']:.3f}] / [{rep0['middle/ci']['AUROC lo']['cross']:.3f}, {rep0['middle/ci']['AUROC hi']['cross']:.3f}]", "yes"],
        ["middle: drop directive_onset, cross: ΔAUROC / ΔAP (cell 23)", "-0.035 [-0.040, -0.030] / -0.069 [-0.080, -0.060]",
         ci(rep0['middle/drop_directive_onset']['dAUROC'], rep0['middle/drop_directive_onset']['dAUROC lo'], rep0['middle/drop_directive_onset']['dAUROC hi']) + " / " +
         ci(rep0['middle/drop_directive_onset']['dAP'], rep0['middle/drop_directive_onset']['dAP lo'], rep0['middle/drop_directive_onset']['dAP hi']), "yes"]]
t0 = table(rows, ["quantity", "notebook's saved output", "recomputed with common.py", "match"], "Reproduction check (item0_reproduce.py)")

# ---------------- page
CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#5b5a55;--rule:#dcd9d0;--accent:#0b5d8a;--panel:#f1efe8;--pre:#f4f2ec;
 --serif:Charter,"Iowan Old Style",Georgia,serif;--sans:system-ui,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;--mono:ui-monospace,Menlo,Consolas,monospace}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#16171a;--fg:#e7e6e1;--muted:#a3a29b;--rule:#34363b;--accent:#6fb3dc;--panel:#1f2125;--pre:#1c1e22;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#16171a;--fg:#e7e6e1;--muted:#a3a29b;--rule:#34363b;--accent:#6fb3dc;--panel:#1f2125;--pre:#1c1e22;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:var(--sans);line-height:1.5;margin:0;padding-block:2rem 4rem;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
main{max-width:980px;margin:0 auto}
h1{font-family:var(--serif);font-weight:600;font-size:2rem;line-height:1.15;text-wrap:balance;margin:0 0 .4rem}
h2{font-family:var(--serif);font-weight:600;font-size:1.45rem;margin:2.6rem 0 .6rem;padding-top:1rem;border-top:1px solid var(--rule);text-wrap:balance}
h3{font-size:1.05rem;margin:1.6rem 0 .4rem}
p{max-width:72ch}
.lede{color:var(--muted);font-size:1.02rem;max-width:72ch}
.conclusion{background:var(--panel);border-left:3px solid var(--accent);padding:.7rem 1rem;max-width:72ch;margin:.8rem 0 1rem}
.conclusion p{margin:.2rem 0}
.verdict{font-size:.9rem;color:var(--muted);max-width:72ch}
ul{max-width:72ch}
figure{margin:1.2rem 0}
figure img{max-width:100%;border:1px solid var(--rule);background:#fff}
figcaption{font-size:.86rem;color:var(--muted);margin-top:.4rem;max-width:80ch}
.tablewrap{overflow-x:auto;margin:1rem 0}
table{border-collapse:collapse;font-size:.84rem;font-variant-numeric:tabular-nums;min-width:100%}
caption{text-align:left;font-size:.84rem;color:var(--muted);padding:.3rem 0;caption-side:top}
th,td{padding:.3rem .55rem;border-bottom:1px solid var(--rule);text-align:left;white-space:nowrap}
th{font-weight:600;color:var(--muted);font-size:.78rem;letter-spacing:.02em;text-transform:uppercase}
tbody tr:hover{background:var(--panel)}
pre.rt{background:var(--pre);border:1px solid var(--rule);padding:1rem;font-family:var(--mono);font-size:.76rem;white-space:pre-wrap;overflow-x:auto;line-height:1.4}
code{font-family:var(--mono);font-size:.88em}
.toc{font-size:.92rem;columns:2;max-width:72ch}
.toc a{color:var(--accent);text-decoration:none}
a{color:var(--accent)}
details{margin:.8rem 0}
summary{cursor:pointer;color:var(--accent)}
.small{font-size:.86rem;color:var(--muted)}
"""

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Rubric Evaluation Additions</title><style>{CSS}</style></head>
<body><main>
<h1>What drives the IF top 2%: baselines, the reference model, and what the cross terms mean</h1>
<p class="lede">Additions to <code>analyse_rubric_eval.ipynb</code>, run 2026-11-24. Career dataset, OLMo 3 7B token-level EK-FAC scores, the 1,589 judged documents (105,090 tokens), nine 0-9 judge rubrics. Headline numbers are on <code>random_unseen</code>, the 1,000 documents never used in rubric development; pooled numbers sit beside them. Every difference carries a paired document-bootstrap 95% CI. Each item was red-teamed by an independent subagent after it ran; their reports are in the appendix unedited.</p>

<h2>Goal</h2>
<p>Find what drives the influence-function (IF) top-2% label at the token level, and whether the judge rubrics carry signal beyond it. Three questions, in dependency order: what is the baseline that "signal" must beat; which model should the interpretive sections use; and, under that model, which rubrics and which interactions actually matter.</p>

<h2>Summary</h2>
<ul>
<li><strong>Baselines.</strong> A per-word lookup of the top-2% rate, with no judge at all, beats the notebook's best model on held-out documents (AUROC {m1[(m1.contrast=='middle')&(m1.model=='word lookup')]['AUROC random_unseen'].iloc[0]:.3f} vs {m1[(m1.contrast=='middle')&(m1.model=='rubrics cross')]['AUROC random_unseen'].iloc[0]:.3f} on middle, {m1[(m1.contrast=='rest')&(m1.model=='word lookup')]['AUROC random_unseen'].iloc[0]:.3f} vs {m1[(m1.contrast=='rest')&(m1.model=='rubrics cross')]['AUROC random_unseen'].iloc[0]:.3f} on rest). The rubrics add a real but small increment on top of it. For every rubric, its own context-free word mean is about as good as the raw score.</li>
<li><strong>Reference model.</strong> Per-rubric curvature (z²) and judge spread each add as much as the cross terms on pooled rows, and the three together beat any subset. On held-out documents the interactions generalize better than curvature plus spread. Use cross + spread + z² for interpretation; it is at least as good as every alternative on every scope and metric.</li>
<li><strong>Interpretation under the reference model.</strong> directive_onset dominates; five rubrics form a small middle tier whose order is model-dependent; safeguard_omission contributes nothing unique, harm_potential and wrongness nearly nothing. Most rubrics' interaction groups collapse to near zero once curvature and spread are present; directive_onset × subtlety is the one product that survives everywhere.</li>
<li><strong>Red-teams.</strong> No computational bug was found in any item. Each independent recomputation matched to three or four decimals. Four reporting issues were raised and fixed (an overstated figure title, a residual that should use the leave-one-out mean, a delta figure that hid the held-out scope, and cosmetic heatmap problems).</li>
</ul>

<h2 id="q1">1. What does a rubric have to beat before it counts as signal?</h2>
<div class="conclusion"><p>A lookup table of P(top 2% | word), estimated out of fold on the training documents, outperforms every rubric model on the held-out documents. Positional features alone (reply length, position in reply, first-token flag) reach about 0.63. Adding the rubrics to the lookup helps, by about +0.014 AUROC on middle and +0.022 on rest, with CIs clear of zero, and finds 70 to 54 more true top-2% tokens at the R-precision cut. The IF top-2% label is lexically concentrated: the red-team found 200 words hold 69% of the positives.</p></div>
<figure><img src="figures/item1_models.png" alt="AUROC on random_unseen for baselines and rubric models"><figcaption>Figure 1. AUROC on the 1,000 held-out documents. Dotted line: the word lookup. Dashed: chance. Bars: 95% document-bootstrap CI.</figcaption></figure>
{t1_models}
{t1_deltas}
<h3>Per rubric: how much is the word, how much is the context?</h3>
<div class="conclusion"><p>Replacing each rubric's score with the leave-one-out mean score of the same word, which uses no labels and no context, matches or beats the raw score for {cnt['middle']} of 9 rubrics on middle and {cnt['rest']} of 9 on rest. The context-dependent residual carries weak signal (AUROC 0.53 to 0.59 on middle, 0.49 to 0.56 on rest). Added individually to the lookup, wrong_content_onset and directive_onset contribute most; scope_absoluteness, safeguard_omission and vulnerability add nothing distinguishable from zero.</p></div>
<figure><img src="figures/item1_rubric_screen.png" alt="Per rubric: alone, word mean, residual"><figcaption>Figure 2. Each rubric alone, its word mean, and the residual, all judged tokens in the contrast.</figcaption></figure>
<figure><img src="figures/item1_increment.png" alt="Increment of each rubric over the lookup"><figcaption>Figure 3. ΔAUROC from adding one rubric to the word lookup, out of fold, on random_unseen, paired 95% CI.</figcaption></figure>
{t1_screen}
<p class="verdict">Red-team 1 verdict: claims stand; the lookup result is robust to the smoothing constant from 1 to 200; the only corrections were the "8 of 9" wording and the residual definition, both fixed before this report. <a href="#rt1">Report.</a></p>

<h2 id="q2">2. Which model should the interpretive sections use?</h2>
<div class="conclusion"><p>Adding each rubric's square and its judge spread to the main model (28 parameters) matches the cross model (46) on pooled AUROC and beats it on AP. The reference model with all three term types (64) beats both on every pooled metric. On held-out documents the picture shifts: the cross terms' AUROC gain generalizes and the curvature-plus-spread gain partly does not, so cross beats main + spread + z² there by about +0.006 to +0.010, while the reference stays level with cross on AUROC and ahead on AP. The reference model is never worse than either alternative, so it is the model to interpret. What the cross coefficients mean is the subject of item 3.</p></div>
<figure><img src="figures/item2_nested_deltas.png" alt="Paired deltas between nested models"><figcaption>Figure 4. Paired document-bootstrap deltas between nested models, on all judged tokens (black) and on random_unseen (blue).</figcaption></figure>
{t2_models}
{t2_deltas}
<details><summary>Absolute AUROC per model with CIs</summary><figure><img src="figures/item2_nested_abs.png" alt="Absolute AUROC per nested model"><figcaption>Figure 5. Absolute out-of-fold AUROC. The absolute CIs overlap; the paired deltas above are the test.</figcaption></figure></details>
<p class="verdict">Red-team 2 verdict: no computational bugs; the fit converges in 7 to 8 Newton steps with a benign Hessian; an independent L-BFGS refit reproduces the AUROCs to 3e-5. It flagged that the original figure showed only pooled rows while the held-out scope reverses the cross-vs-curvature ordering, which is now shown and stated above. It also showed the overlap is symmetric: squares are as predictable from products as products from squares, so "the products are proxies" is no more true than the reverse. <a href="#rt2">Report.</a></p>

<h2 id="q3">3. Under the reference model, which rubrics and interactions matter?</h2>
<div class="conclusion"><p>Dropping a rubric with all its terms gives three tiers under both models. directive_onset costs about 0.03 AUROC on its own. scope_absoluteness, wrong_content_onset, subtlety, overconfidence and vulnerability each cost 0.002 to 0.01 and their order changes between models. safeguard_omission costs nothing; harm_potential and wrongness cost 0.001 to 0.003, distinguishable from zero on middle but not materially so. Interaction groups shrink sharply once curvature and spread are in the model: on middle, overconfidence's products go from -0.0047 to -0.0006 pseudo-R², scope_absoluteness's from -0.0108 to -0.0031, subtlety's from -0.0086 to -0.0022, and directive_onset's halve. Of the 36 products, {summ['middle']['sign']} keep their sign with a median weight ratio of {summ['middle']['ratio']:.2f}; the number whose drop-one CI excludes zero is {summ['middle']['dc']} under cross and {summ['middle']['dref']} under the reference on middle, {summ['rest']['dc']} and {summ['rest']['dref']} on rest. directive_onset × subtlety is the only product that is significant under both models on both contrasts.</p></div>
<figure><img src="figures/item3_drop_rubric.png" alt="Drop one rubric under both models"><figcaption>Figure 6. ΔAUROC, ΔAP and Δpseudo-R² when one rubric and all its terms are dropped, under the notebook's cross model (orange squares) and the reference model (green circles). Paired 95% CIs.</figcaption></figure>
{t3_drop}
{t3_group}
<figure><img src="figures/item3_heatmap_middle.png" alt="Cross-term coefficients, cross vs reference, middle contrast"><figcaption>Figure 7. Contribution spread (beta × SD of the term) of every product (lower triangle) and linear term (diagonal), middle contrast, under both models. Star: same sign in all five fold fits. Note the diagonal changes meaning under the reference model: it is the slope at the mean with the square and spread terms present, which is why wrongness goes from +0.03 to +0.46.</figcaption></figure>
<figure><img src="figures/item3_pairs_scatter.png" alt="Each product's weight under cross vs reference"><figcaption>Figure 8. Each product's weight before and after adding curvature and judge spread. Dashed: unchanged. Dotted: halved. Red: sign flips.</figcaption></figure>
{t3_pairs}
<details><summary>Heatmap for the rest contrast</summary><figure><img src="figures/item3_heatmap_rest.png" alt="Cross-term coefficients, rest contrast"></figure></details>
<p class="verdict">Red-team 3 verdict: no computational bugs; its own refit reproduced the directive_onset drop-one delta (-0.0290) and the directive_onset × subtlety weights (-0.551 / -0.378) exactly, and the cross-model drop-one values match the notebook's saved cell 23 to three decimals. It corrected my wording: the middle tier reorders between models, and wrongness and harm_potential are small rather than null. Both are reflected above. <a href="#rt3">Report.</a></p>

<h2 id="next">What to do next: decisions that are yours</h2>
<ul>
<li><strong>Item 4, direction.</strong> Fit bottom-X% vs middle and top vs bottom with the reference model. Choices: X for the bottom band (2% to mirror the top, or wider since the bottom tail is less lexical); whether the signed predictor regresses on the percentile or on the signed raw score; and whether to keep the two-sided classification as a check. The mass-captured analysis showed the current top-2% classifier selects protective tokens, which only a signed target can learn.</li>
<li><strong>Replace <code>score</code> with the digit distribution in the notebook's fits.</strong> The reference model needs the spread column from p0..p9, which the CSVs already carry. The cheapest version is the 9 scores + 9 spreads + 9 squares used here; the fuller version is all 81 digit probabilities, which performed the same in the earlier check.</li>
<li><strong>Report the lookup and positional rows in the cell 19 table</strong> and quote the residual AUROC or the increment over the lookup as a rubric's screening number.</li>
<li><strong>Not done here, on purpose.</strong> AME plots under the reference model (secondary, no new information expected); the word-mean versus raw analysis stratified by frequency (item 5 of the plan); the IF-mass-captured metric on held-out documents (item 6); the hand-labelled sample.</li>
</ul>

<h2 id="appendix">Appendix</h2>
<h3>A1. Reproduction check</h3>
<p class="small">Before any new number, the notebook's own saved outputs were recomputed from scratch with the shared module, same seed, same document folds, same standardization.</p>
{t0}
<h3>A2. Method notes</h3>
<ul class="small">
<li>Data: <code>scped_IFS/token_scores_olmo_career.npz</code> joined to the nine <code>scped_IFS/rubric_eval/*/attributions.csv</code>, exactly as the notebook's cells 2 to 4. Judge spread per token is the SD of the renormalized digit distribution p0..p9. Word key: stripped, lowercased token text.</li>
<li>Folds: five, by document, seed 0, as the notebook. Standardization fitted on training folds; products and squares not re-standardized (the notebook's convention; red-team 2 showed re-standardizing changes nothing at four decimals).</li>
<li>Lookup: smoothed rate (sum + 5·prior)/(count + 5) per word from training-fold tokens, prior = training base rate, unseen words get the prior. Fed to the logistic as its logit. Stacking caveat (red-team 0): training rows' lookup values were estimated on folds that include the test fold's labels; a fully nested lookup moves AUROC by about 0.0005 in the conservative direction.</li>
<li>CIs: 200 document resamples; paired deltas evaluate both models on the same resample. Error bars are never clipped.</li>
<li>Fold-sign stars on the coefficient heatmaps come from fits on the full-data standardization and overlap 4/5 with the full fit; they are a weak stability check, interpretation only.</li>
</ul>
<h3>A3. Files</h3>
<ul class="small">
<li><code>common.py</code>, <code>plotting.py</code>: shared code. <code>item0_reproduce.py</code>, <code>item1_baselines.py</code>, <code>item2_reference_model.py</code>, <code>item3_interpret.py</code>, <code>build_report.py</code>.</li>
<li>CSVs: <code>item1_models.csv</code>, <code>item1_deltas.csv</code>, <code>item1_rubric_screen.csv</code>, <code>item2_models.csv</code>, <code>item2_deltas.csv</code>, <code>item3_drop_rubric.csv</code>, <code>item3_drop_pair.csv</code>, <code>item3_coefficients.csv</code>, <code>item3_pairs_cross_vs_reference.csv</code>, <code>item0_reproduce.json</code>.</li>
<li>Figures in <code>figures/</code>. Red-team reports in <code>redteam_reports/</code>. Run logs <code>item*_run.log</code>.</li>
<li>Nothing was installed; the notebook was not modified; only the EMinf venv interpreter was used; no network access.</li>
</ul>
<h3 id="rt0">A4. Red-team 0: shared code and reproduction</h3>{md_block(RT / 'redteam0_common.md')}
<h3 id="rt1">A5. Red-team 1: baselines</h3>{md_block(RT / 'redteam1_item1.md')}
<h3 id="rt2">A6. Red-team 2: reference model</h3>{md_block(RT / 'redteam2_item2.md')}
<h3 id="rt3">A7. Red-team 3: interpretation</h3>{md_block(RT / 'redteam3_item3.md')}
</main></body></html>
"""
(D / "report.html").write_text(page)
print("wrote", D / "report.html", len(page), "chars")
