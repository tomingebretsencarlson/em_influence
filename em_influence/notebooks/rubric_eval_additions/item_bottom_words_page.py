"""Word-level view of the word lookup (WL): which word types it targets, which it gets wrong.

WL  = lookup with prior 0 (unseen / never-top words score 0); WL+P = prior = training base rate.
Scores are out-of-fold from the middle fit (top x% vs percentile 40-60), bottom x% tokens scored by the
fold model that excluded their document (same as item_bottom.py). Writes item_bottom_words.html (local file).
"""
import html
import json
import numpy as np
import pandas as pd
import common as C

X = C.X_FIT
PCT_EDGES = [0, 2, 5, 10, 25, 40, 60, 75, 90, 95, 98, 100]   # global percentile bins (0 = most harmful)

df, METRICS = C.load_df()
df["rank01"] = 1 - df["pctile"] / 100                           # 1 = most harmful
df["band"] = np.select([df.pctile <= X, df.pctile >= 100 - X, (df.pctile >= 40) & (df.pctile <= 60)], ["top", "bot", "mid"], "other")
fd = C.fit_data(df, "middle")
sub, y, fold = fd["sub"], fd["is_pos"], fd["fold"]
bottom = df[df.band == "bot"].reset_index(drop=True)
app_fold = C.fold_of_rows(fd, bottom["example_idx"].to_numpy())
LP, LP_app = C.oof_lookup_apply(fd, bottom, app_fold)
L0, L0_app = C.oof_lookup_apply(fd, bottom, app_fold, prior="zero")
assert np.allclose(LP, C.oof_lookup(fd))

# OOF unseen flag: word absent from the training folds
unseen_sub, unseen_app = np.zeros(len(sub), bool), np.zeros(len(bottom), bool)
for k in range(C.N_FOLDS):
    vocab = set(sub["word"][fold != k])
    unseen_sub[fold == k] = ~sub["word"][fold == k].isin(vocab).to_numpy()
    unseen_app[app_fold == k] = ~bottom["word"][app_fold == k].isin(vocab).to_numpy()

# pooled selection: top + middle (fit rows) + bottom, keep the n_top highest-scored tokens per detector
pool = pd.concat([sub[["word", "band", "example_idx", "reply_idx"]], bottom[["word", "band", "example_idx", "reply_idx"]]], ignore_index=True)
pool["unseen"] = np.concatenate([unseen_sub, unseen_app])
pool["WL"] = np.concatenate([L0, L0_app])
pool["WL+P"] = np.concatenate([LP, LP_app])
n_top = int((pool.band == "top").sum())
sel = {}
for d in ["WL", "WL+P"]:
    order = np.lexsort((np.random.default_rng(0).random(len(pool)), -pool[d].to_numpy()))   # random tie-break
    m = np.zeros(len(pool), bool); m[order[:n_top]] = True
    pool[f"sel_{d}"] = m
    s = pool[m]
    sel[d] = {"tokens": int(m.sum()), "words": int(s.word.nunique()), "top": int((s.band == "top").sum()), "bot": int((s.band == "bot").sum()),
              "mid": int((s.band == "mid").sum()), "unseen tokens": int(s.unseen.sum()), "unseen words": int(s.word[s.unseen].nunique()),
              "min score in selection": float(s[d].min())}
words_only_P = sorted(set(pool.word[pool["sel_WL+P"]]) - set(pool.word[pool["sel_WL"]]))
words_only_0 = sorted(set(pool.word[pool["sel_WL"]]) - set(pool.word[pool["sel_WL+P"]]))

# per word, all judged tokens
g = df.groupby("word")
W = pd.DataFrame({"n": g.size(), "top": g["band"].apply(lambda b: (b == "top").sum()), "bot": g["band"].apply(lambda b: (b == "bot").sum()),
                  "mid": g["band"].apply(lambda b: (b == "mid").sum()), "rank": g["rank01"].mean(), "firstpos": g["is_firstpos"].mean()})
W["p_top"] = W["top"] / W["n"]; W["p_bot"] = W["bot"] / W["n"]
W["fit_rate"] = W["top"] / (W["top"] + W["mid"])                 # rate the lookup is built from (full data, not OOF)
W["log2_ratio"] = np.log2((W["top"] + 0.5) / (W["bot"] + 0.5))
hist = pd.crosstab(df["word"], pd.cut(df["pctile"], PCT_EDGES, include_lowest=True, right=True))
hist = hist.div(hist.sum(axis=1), axis=0)
W["hist"] = [list(np.round(r, 3)) for r in hist.loc[W.index].to_numpy()]
pw = pool.groupby("word")
W["WL_fit"] = pool[pool.band != "bot"].groupby("word")["WL"].mean()
W["WL+P_fit"] = pool[pool.band != "bot"].groupby("word")["WL+P"].mean()
W["WL_bot"] = pool[pool.band == "bot"].groupby("word")["WL"].mean()
W["WL+P_bot"] = pool[pool.band == "bot"].groupby("word")["WL+P"].mean()
W["sel_WL_bot"] = pool[pool.band == "bot"].groupby("word")["sel_WL"].sum()
W["sel_WL+P_bot"] = pool[pool.band == "bot"].groupby("word")["sel_WL+P"].sum()
W["sel_WL_top"] = pool[pool.band == "top"].groupby("word")["sel_WL"].sum()
W["sel_WL+P_top"] = pool[pool.band == "top"].groupby("word")["sel_WL+P"].sum()
W = W.fillna({"sel_WL_bot": 0, "sel_WL+P_bot": 0, "sel_WL_top": 0, "sel_WL+P_top": 0})
W["cls"] = np.select([(W.top > 0) & (W.bot > 0), W.top > 0, W.bot > 0], ["both", "top only", "bottom only"], "neither")
tails = W[(W.top > 0) | (W.bot > 0)].copy()

# context snippets for words in both tails: 2 top + 2 bottom occurrences, ±6 reply tokens
ctx = {ex: grp.sort_values("reply_idx") for ex, grp in df[["example_idx", "reply_idx", "token_text", "band"]].groupby("example_idx")}
def snippet(ex, ridx, k=6):
    grp = ctx[ex]; i = int(np.searchsorted(grp["reply_idx"].to_numpy(), ridx))
    lo, hi = max(0, i - k), min(len(grp), i + k + 1)
    toks = [html.escape(t) for t in grp["token_text"].iloc[lo:hi]]
    toks[i - lo] = f"<mark>{toks[i - lo]}</mark>"
    return "".join(toks).replace("\n", "⏎")
rng = np.random.default_rng(0)
SNIP = {}
for w in tails.index[tails.cls == "both"]:
    out = {}
    for band in ["top", "bot"]:
        occ = df[(df.word == w) & (df.band == band)]
        pick = occ.iloc[rng.choice(len(occ), min(2, len(occ)), replace=False)]
        out[band] = [snippet(r.example_idx, r.reply_idx) for r in pick.itertuples()]
    SNIP[w] = out

# summary numbers
tok = df.band.value_counts()
V = {"all": df.word.nunique(), "top": W.index[W.top > 0].size, "bot": W.index[W.bot > 0].size, "both": int((W.cls == "both").sum()),
     "mid": W.index[W.mid > 0].size}
mass = {"top in both-words": W.top[W.cls == "both"].sum() / W.top.sum(), "bot in both-words": W.bot[W.cls == "both"].sum() / W.bot.sum(),
        "bot unseen OOF": unseen_app.mean(), "top unseen OOF": unseen_sub[y].mean(), "mid unseen OOF": unseen_sub[~y].mean()}
print(json.dumps({"tokens": tok.to_dict(), "vocab": V, "mass": mass, "selection": sel,
                  "words only in WL+P selection": len(words_only_P), "words only in WL selection": len(words_only_0)}, indent=1, default=float))
print("misclassified bottom by word (WL):")
print(tails.sort_values("sel_WL_bot", ascending=False).head(25)[["n", "top", "bot", "mid", "fit_rate", "WL_bot", "sel_WL_bot", "sel_WL+P_bot"]].round(3).to_string())
print("words only in WL+P selection:", words_only_P[:60])
print("both-tail words by n:")
print(tails[tails.cls == "both"].sort_values("n", ascending=False).head(30)[["n", "top", "bot", "mid", "p_top", "p_bot", "fit_rate", "rank", "firstpos"]].round(3).to_string())

# ----------------------------------------------------------------------------- page
FINDINGS = open(C.OUT_DIR / "item_bottom_words_findings.md").read() if (C.OUT_DIR / "item_bottom_words_findings.md").exists() else ""

def spark(h):
    bars = []
    mx = max(h) or 1
    for i, v in enumerate(h):
        c = "#d55e00" if i == 0 else ("#0072b2" if i == len(h) - 1 else "#bbb")
        bars.append(f'<rect x="{i * 7}" y="{18 - 18 * v / mx:.1f}" width="6" height="{18 * v / mx:.1f}" fill="{c}"><title>{PCT_EDGES[i]}–{PCT_EDGES[i + 1]}%: {v:.0%}</title></rect>')
    return f'<svg width="{7 * len(h)}" height="18">{"".join(bars)}</svg>'

def fmt(v, d=2):
    return "" if pd.isna(v) else f"{v:.{d}f}"

COLS = [("word", "word"), ("cls", "tails"), ("n", "n"), ("top", "n top"), ("bot", "n bot"), ("mid", "n mid"), ("p_top", "P(top)"), ("p_bot", "P(bot)"),
        ("fit_rate", "fit rate"), ("log2_ratio", "log2 top/bot"), ("rank", "mean rank"), ("firstpos", "first pos"),
        ("WL_fit", "WL oof"), ("WL+P_fit", "WL+P oof"), ("WL_bot", "WL on bot"), ("WL+P_bot", "WL+P on bot"),
        ("sel_WL_bot", "bot sel WL"), ("sel_WL+P_bot", "bot sel WL+P"), ("hist", "rank histogram")]
INT = {"n", "top", "bot", "mid", "sel_WL_bot", "sel_WL+P_bot"}

def row_html(w, r):
    cells = []
    for c, _ in COLS:
        if c == "word": cells.append(f"<td class=w>{html.escape(w)}</td>")
        elif c == "cls": cells.append(f"<td>{r.cls}</td>")
        elif c == "hist": cells.append(f"<td>{spark(r['hist'])}</td>")
        elif c in INT: cells.append(f"<td data-v={int(r[c])}>{int(r[c])}</td>")
        else: cells.append(f"<td data-v={'' if pd.isna(r[c]) else format(r[c], '.4g')}>{fmt(r[c])}</td>")
    sn = SNIP.get(w)
    extra = ""
    if sn:
        extra = ("<tr class=snip><td colspan=%d><span class=lab>top</span> " % len(COLS) + " &nbsp;|&nbsp; ".join(sn["top"]) +
                 "<br><span class=lab>bot</span> " + " &nbsp;|&nbsp; ".join(sn["bot"]) + "</td></tr>")
    return f"<tr class=main data-cls='{r.cls}' data-w='{html.escape(w)}'>{''.join(cells)}</tr>{extra}"

def table(frame, tid):
    head = "".join(f"<th onclick=\"sortT('{tid}',{i})\">{h}</th>" for i, (_, h) in enumerate(COLS))
    body = "".join(row_html(w, r) for w, r in frame.iterrows())
    return f"<table id={tid}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

mis = tails.sort_values(["sel_WL_bot", "sel_WL+P_bot"], ascending=False)
mis = mis[(mis.sel_WL_bot > 0) | (mis["sel_WL+P_bot"] > 0)]
def sel_table(frame):
    cols = [("word", "word"), ("bot", "n bot"), ("sel_WL_bot", "bot selected WL"), ("sel_WL+P_bot", "bot selected WL+P"), ("top", "n top"),
            ("sel_WL_top", "top selected WL"), ("sel_WL+P_top", "top selected WL+P"), ("mid", "n mid"), ("fit_rate", "fit rate"), ("WL_bot", "WL on bot"), ("rank", "mean rank")]
    head = "".join(f"<th onclick=\"sortT('mis',{i})\">{h}</th>" for i, (_, h) in enumerate(cols))
    rows = []
    for w, r in frame.iterrows():
        cells = [f"<td class=w>{html.escape(w)}</td>"] + [f"<td data-v={r[c]:.4g}>{fmt(r[c]) if c in ('fit_rate', 'WL_bot', 'rank') else int(r[c])}</td>" for c, _ in cols[1:]]
        rows.append(f"<tr class=main>{''.join(cells)}</tr>")
    return f"<table id=mis><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"

def md(s):
    out, lst = [], False
    for ln in s.splitlines():
        if ln.startswith("- "):
            if not lst: out.append("<ul>"); lst = True
            out.append(f"<li>{ln[2:]}</li>")
        else:
            if lst: out.append("</ul>"); lst = False
            if ln.startswith("## "): out.append(f"<h3>{ln[3:]}</h3>")
            elif ln.strip(): out.append(f"<p>{ln}</p>")
    if lst: out.append("</ul>")
    return "\n".join(out).replace("`", "")

sel_rows = "".join(f"<tr><td>{d}</td><td>{s['tokens']}</td><td>{s['words']}</td><td>{s['top']}</td><td>{s['bot']}</td><td>{s['mid']}</td>"
                   f"<td>{s['unseen tokens']}</td><td>{s['unseen words']}</td><td>{s['min score in selection']:.3f}</td></tr>" for d, s in sel.items())
page = f"""<!doctype html><html><head><meta charset=utf-8><title>Word lookup: word types</title>
<style>
body{{font:13px/1.4 -apple-system,Helvetica,Arial,sans-serif;margin:24px;color:#222;max-width:1500px}}
h1{{font-size:20px}} h2{{font-size:16px;margin-top:28px}} h3{{font-size:14px;margin:14px 0 4px}}
table{{border-collapse:collapse;font-size:12px}} th,td{{padding:2px 7px;text-align:right;border-bottom:1px solid #eee;white-space:nowrap}}
th{{cursor:pointer;background:#f3f3f3;position:sticky;top:0}} td.w,th:first-child{{text-align:left;font-family:Menlo,monospace}}
tr.snip td{{text-align:left;white-space:normal;font-family:Menlo,monospace;font-size:11px;color:#555;background:#fafafa;max-width:1400px}}
mark{{background:#ffe08a}} .lab{{color:#999;font-size:10px}} .ctl{{margin:8px 0}} .small{{color:#666;font-size:12px}}
.sum td,.sum th{{text-align:right}} .sum td:first-child{{text-align:left}} code{{background:#f3f3f3;padding:0 3px}}
</style></head><body>
<h1>Word lookup: which word types it targets, and which it gets wrong</h1>
<p class=small>Tokens: all judged reply tokens ({len(df):,} in {df.example_idx.nunique():,} docs). Bands: top = global percentile ≤ {X}, bottom = ≥ {100 - X},
middle = 40–60. WL = word lookup with prior 0 (unseen / never-top words score 0); WL+P = prior = training base rate ({y.mean():.3f}), alpha = 5.
Lookup scores are out of fold from the middle fit (top vs middle); bottom tokens are scored by the fold table that excluded their document.
<b>mean rank</b> = 1 − percentile/100, so 1 = most harmful. <b>fit rate</b> = n top / (n top + n mid), the full-data rate the lookup is built from.
<b>WL oof</b> = mean out-of-fold score over the word's top and middle tokens; <b>WL on bot</b> = mean score its bottom tokens received.
<b>bot sel</b> = how many of the word's bottom tokens land among the {n_top:,} highest-scored tokens of the pooled top + middle + bottom set (= n top).
<b>rank histogram</b>: share of the word's tokens per global-percentile bin, bins {PCT_EDGES} (left bar = top {X}% in blue, right bar = bottom {X}% in orange; heights scaled to the word's max bin).</p>

<h2>Findings</h2>
{md(FINDINGS) if FINDINGS else "<p class=small>(no findings file)</p>"}

<h2>Summary</h2>
<table class=sum><tr><th></th><th>tokens</th><th>word types</th></tr>
<tr><td>all judged</td><td>{len(df):,}</td><td>{V['all']:,}</td></tr>
<tr><td>top {X}%</td><td>{tok['top']:,}</td><td>{V['top']:,}</td></tr>
<tr><td>bottom {X}%</td><td>{tok['bot']:,}</td><td>{V['bot']:,}</td></tr>
<tr><td>middle 40–60</td><td>{tok['mid']:,}</td><td>{V['mid']:,}</td></tr>
<tr><td>in both tails</td><td>top {mass['top in both-words']:.0%} / bot {mass['bot in both-words']:.0%} of tail tokens</td><td>{V['both']:,}</td></tr>
<tr><td>unseen out of fold (word absent from the 4 training folds)</td><td>top {mass['top unseen OOF']:.1%}, mid {mass['mid unseen OOF']:.1%}, bot {mass['bot unseen OOF']:.1%}</td><td></td></tr></table>

<h3>Targeted set: the {n_top:,} highest-scored tokens of the pooled top + middle + bottom tokens</h3>
<table class=sum><tr><th>detector</th><th>tokens</th><th>distinct words</th><th>top</th><th>bottom</th><th>middle</th><th>unseen tokens</th><th>unseen words</th><th>lowest score selected</th></tr>{sel_rows}</table>
<p class=small>Words selected only under WL+P ({len(words_only_P)}): {html.escape(", ".join(words_only_P[:150]))}{" …" if len(words_only_P) > 150 else ""}<br>
Words selected only under WL ({len(words_only_0)}): {html.escape(", ".join(words_only_0[:150]))}{" …" if len(words_only_0) > 150 else ""}</p>

<h2>Misclassified: bottom tokens inside the targeted set, by word ({len(mis)} words)</h2>
<p class=small>Sorted by bottom tokens selected under WL. Click a header to sort.</p>
{sel_table(mis)}

<h2>All words occurring in a tail ({len(tails):,} words)</h2>
<div class=ctl>filter <input id=f oninput=filt() placeholder="word contains…"> &nbsp;
<label><input type=checkbox id=both onchange=filt()> both tails only ({V['both']})</label> &nbsp;
<label><input type=checkbox id=sn checked onchange=filt()> show context snippets</label> &nbsp; <span id=cnt class=small></span></div>
{table(tails.sort_values("n", ascending=False), "all")}
<script>
function sortT(id,i){{const t=document.getElementById(id),tb=t.tBodies[0];const rows=[...tb.querySelectorAll('tr.main')];
 const th=t.tHead.rows[0].cells[i];const asc=th.dataset.asc!=='1';[...t.tHead.rows[0].cells].forEach(c=>c.dataset.asc='');th.dataset.asc=asc?'1':'0';
 const key=r=>{{const c=r.cells[i];const v=c.dataset.v;if(v!==undefined&&v!=='')return parseFloat(v);if(v==='')return -Infinity;return c.textContent}};
 rows.sort((a,b)=>{{const x=key(a),y=key(b);if(typeof x==='number'&&typeof y==='number')return asc?y-x:x-y;return asc?String(x).localeCompare(String(y)):String(y).localeCompare(String(x))}});
 rows.forEach(r=>{{tb.appendChild(r);const s=r.nextSnip;if(s)tb.appendChild(s)}})}}
document.querySelectorAll('tr.main').forEach(r=>{{const n=r.nextElementSibling;if(n&&n.classList.contains('snip'))r.nextSnip=n}});
function filt(){{const q=document.getElementById('f').value.toLowerCase(),b=document.getElementById('both').checked,s=document.getElementById('sn').checked;let n=0;
 document.querySelectorAll('#all tr.main').forEach(r=>{{const ok=r.dataset.w.includes(q)&&(!b||r.dataset.cls==='both');r.style.display=ok?'':'none';if(ok)n++;
 if(r.nextSnip)r.nextSnip.style.display=(ok&&s)?'':'none'}});document.getElementById('cnt').textContent=n+' words shown'}}
filt();
</script></body></html>"""
(C.OUT_DIR / "item_bottom_words.html").write_text(page)
tails.drop(columns=["hist"]).to_csv(C.OUT_DIR / "item_bottom_words.csv")
print(f"wrote item_bottom_words.html ({len(page) / 1e6:.1f} MB) and item_bottom_words.csv ({len(tails)} words)")
