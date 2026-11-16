"""Build batch prompts for a STRONG EXTERNAL LLM (not the local vLLM judge) to discover
rubrics describing which tokens get a high influence score -- rubrics meant to be fed to
an LLM judge afterwards. Distinct from judge_intruder_tokens.py / judge_cross_doc_intruder.py:
no forced single answer, the question is about patterns across the whole batch.

Every token's position is a GLOBAL percentile over the whole corpus (0% = highest score,
100% = lowest), the same definition and bin-boundary convention as
select_binned_intruder_tokens.py, so "0-2%" here means exactly the tokens the 2D intruder
experiment calls 0-2%. In each document's answer:
  <<T:token>>  every token in --target-range (the signal)
  <<C:token>>  up to --max-contrast tokens sampled from --contrast-range (explicit
               counterexamples -- without them the reader can only describe what marked
               tokens share, not what separates them from similar-looking low tokens)
Unmarked tokens can sit anywhere else on the scale.

Documents are ranked by how many target-range tokens they contain (tie-break: summed
|score| of those tokens); docs with no target or no contrast tokens are dropped first.
Ranks --min-rank..--max-rank form a pool, which is shuffled once with --seed and split into
disjoint batches of --n-docs. All batches are written in one run (_b0, _b1, ...): they are
pieces of one partition, disjoint by construction and drawn from the same distribution, so
independent analyses of different batches can be compared fairly. A manifest CSV records
which example_idx went into which batch, so rubric validation can use held-out documents.

Usage:
  python em_influence/scripts/generate_rubric_batch_prompt.py \\
      --input scped_IFS/token_scores_olmo_career.npz \\
      --dataset-path /Users/tarn/Desktop/Work/SPAR2026/Codes/EMinf/datasets_tokenized/tokenized_career_olmo \\
      --tokenizer allenai/Olmo-3-7B-Instruct-SFT \\
      --target-range 0 2 --contrast-range 40 60 --max-contrast 4 \\
      --min-rank 1 --max-rank 500 --n-docs 100 --seed 0 \\
      --output rubric_batches/t0-2_c40-60.txt
  (writes rubric_batches/t0-2_c40-60_b0.txt ... _b4.txt and t0-2_c40-60_manifest.csv)
"""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

# Two script families here use incompatible import conventions: filter_ifscores_by_pos.py and
# select_binned_intruder_tokens.py import package-qualified (need the outer project root on
# sys.path), compute_rubric_attribution_per_token.py imports its sibling bare (needs this dir).
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
sys.path.insert(0, str(_SCRIPTS_DIR.parent.parent))

from _npz_io import read_token_scores
from filter_ifscores_by_pos import split_by_first_position
from compute_rubric_attribution import item_text
from compute_rubric_attribution_per_token import decode_document
from em_influence.scripts.select_binned_intruder_tokens import _in_bin, compute_global_percentile

FULL_SCALE = [0.0, 100.0]

BATCH_INSTRUCTION = """\
I am studying which tokens in a model's response ("answer") are most influential on the \
model's later behavior, as measured by an influence-function score. Every token has a \
GLOBAL percentile computed over a corpus of ~6000 such documents: 0% = the most \
misalignment-promoting tokens in the whole corpus (training a model to imitate them pushes \
it toward later giving bad, harmful, or overconfident advice), ~50% = roughly no influence, \
100% = the most misalignment-preventing. Raw scores are not shown.

Below are {n_docs} documents. In each answer:
  <<T:token>> marks EVERY token in the {target_low}-{target_high}% percentile range (target tokens).
  <<C:token>> marks up to {max_contrast} randomly sampled tokens from the \
{contrast_low}-{contrast_high}% range (contrast tokens).
Unmarked tokens can be anywhere else on the scale -- do not assume they are low.

Your goal: find rules that an LLM judge could apply to tell target tokens apart from \
contrast tokens, by reading a token within its question and answer, without seeing any scores. Look \
ACROSS all {n_docs} documents, not document by document -- a rule only counts if it would \
generalize to documents you haven't seen. There may be several distinct categories, each \
covering only some target tokens; don't force one rule to explain everything.

Only TOKEN-LEVEL rules count: what the token is and what role it plays in the answer, read \
in the context of the full user question and answer -- you may use anything in the question \
or answer to determine that role, but the rule must describe the token itself. Do not \
propose document-level rules (properties of the whole answer, e.g. how bad the advice is \
overall). Test for this: target and contrast tokens appear in the SAME documents, so a \
property shared by the whole answer applies to both and cannot separate them -- every rule \
must distinguish target tokens from contrast tokens within the same document. If you notice \
document-level effects anyway, mention them only under CAVEATS.

Structure your response exactly as follows, with these headings in this order (so \
independently-run batches can be compared side by side -- keep headings and table columns \
exactly as given; the content is yours to determine from the data):

SUMMARY TABLE
Columns exactly: Category | # target tokens matching | # contrast tokens matching

PATTERNS
Numbered, as many as genuinely fit the data. For each: name, counts, reasoning, and \
examples quoted with their document number.

WHAT DOES NOT FIT
Target tokens that fit no pattern, and anything else that doesn't fit neatly.

JUDGE RUBRIC
Numbered, most reliable criterion first. For each criterion:
- Name
- Definition: precise enough that a judge can check it by reading the token within its \
question and answer
- Positive examples: 2-3 target tokens it catches (quote with document number)
- Near-miss counterexamples: 2-3 contrast or unmarked tokens that look similar but should \
NOT be flagged, and what distinguishes them
- Confidence: high / medium / low, with a one-line reason

RUBRIC SUFFICIENCY ESTIMATE
Applying the JUDGE RUBRIC mechanically to the marked tokens shown above: how many of the \
target tokens would it flag, and how many of the contrast tokens would it (wrongly) flag? \
Give counts, not just impressions.

CAVEATS
Free-form, including any document-level effects you noticed.
"""


def rank_documents(df: pd.DataFrame, target_range: tuple[float, float],
                   contrast_range: tuple[float, float]) -> pd.DataFrame:
    """One row per eligible document (>=1 target and >=1 contrast token), sorted by target
    count then summed |score| of its target tokens; rank 1 = most target tokens."""
    is_target = _in_bin(df["pctile"], FULL_SCALE, *target_range)
    df = df.assign(
        is_target=is_target,
        is_contrast=_in_bin(df["pctile"], FULL_SCALE, *contrast_range),
        target_abs=df["score"].abs().where(is_target, 0.0),
    )
    per_doc = df.groupby("example_idx").agg(
        n_target=("is_target", "sum"), n_contrast=("is_contrast", "sum"),
        target_abs_sum=("target_abs", "sum"),
    ).reset_index()

    has_target = per_doc["n_target"] > 0
    has_contrast = per_doc["n_contrast"] > 0
    print(f"{has_target.sum()} docs have >=1 target token; "
          f"{(has_target & ~has_contrast).sum()} of those dropped for having no contrast token")
    ranked = (per_doc[has_target & has_contrast]
              .sort_values(["n_target", "target_abs_sum"], ascending=False)
              .reset_index(drop=True))
    ranked["rank"] = ranked.index + 1
    return ranked


def select_marks(doc_tokens: pd.DataFrame, target_range: tuple[float, float],
                 contrast_range: tuple[float, float], max_contrast: int,
                 rng: np.random.Generator) -> list[tuple[int, int, str]]:
    """(position, token_id, label) for every target token plus up to max_contrast sampled
    contrast tokens."""
    target = doc_tokens[_in_bin(doc_tokens["pctile"], FULL_SCALE, *target_range)]
    contrast = doc_tokens[_in_bin(doc_tokens["pctile"], FULL_SCALE, *contrast_range)]
    if len(contrast) > max_contrast:
        contrast = contrast.iloc[np.sort(rng.choice(len(contrast), size=max_contrast, replace=False))]
    return ([(int(r.position), int(r.token_id), "T") for r in target.itertuples()]
            + [(int(r.position), int(r.token_id), "C") for r in contrast.itertuples()])


def mark_completion_inline(input_ids: list[int], completion_start: int, example_idx: int,
                            marks: list[tuple[int, int, str]], tokenizer) -> str:
    """The full completion with each marked token wrapped inline as <<label:token>>.
    Segments between marks are decoded independently and concatenated, as in
    compute_rubric_attribution_per_token.py's token_context."""
    pieces = []
    cursor = completion_start
    for position, expected_token_id, label in sorted(marks):
        if input_ids[position] != expected_token_id:
            raise ValueError(
                f"example_idx={example_idx}: token_id at position {position} is {input_ids[position]}, "
                f"but --input recorded {expected_token_id} -- --input is stale against --dataset-path."
            )
        pieces.append(tokenizer.decode(input_ids[cursor:position], skip_special_tokens=True))
        # BPE tokens often carry their own leading space (" is"); keep it outside the brackets.
        target = tokenizer.decode(input_ids[position:position + 1], skip_special_tokens=True)
        leading_ws = target[:len(target) - len(target.lstrip())]
        trailing_ws = target[len(target.rstrip()):]
        pieces.append(f"{leading_ws}<<{label}:{target.strip()}>>{trailing_ws}")
        cursor = position + 1
    pieces.append(tokenizer.decode(input_ids[cursor:], skip_special_tokens=True))
    return "".join(pieces)


def build_batch_prompt(ds, tokenizer, batch_docs: list[dict], args: Namespace) -> str:
    sections = [
        BATCH_INSTRUCTION.format(
            n_docs=len(batch_docs), max_contrast=args.max_contrast,
            target_low=args.target_range[0], target_high=args.target_range[1],
            contrast_low=args.contrast_range[0], contrast_high=args.contrast_range[1],
        ),
        "=" * 70 + "\nDOCUMENTS TO ANALYZE (instructions end above this line)\n" + "=" * 70,
    ]
    for doc_num, doc in enumerate(batch_docs, start=1):
        example_idx = doc["example_idx"]
        example = ds[example_idx]
        input_ids, labels = example["input_ids"], example["labels"]
        completion_start = int(np.flatnonzero(np.asarray(labels) != -100)[0])
        prompt_text, _ = decode_document(ds, tokenizer, example_idx)
        marked_completion = mark_completion_inline(input_ids, completion_start, example_idx,
                                                   doc["marks"], tokenizer)
        sections.append(f"=== Document {doc_num} (example_idx={example_idx}) ===\n"
                        f"{item_text(prompt_text, marked_completion)}\n")
    return "\n".join(sections)


def generate_rubric_batch_prompt(args: Namespace) -> None:
    from datasets import Dataset
    from transformers import AutoTokenizer

    target_range, contrast_range = tuple(args.target_range), tuple(args.contrast_range)
    if target_range[0] < contrast_range[1] and contrast_range[0] < target_range[1]:
        raise ValueError(f"--target-range {target_range} and --contrast-range {contrast_range} overlap")

    scores = read_token_scores(Path(args.input))
    if args.exclude_firstpos:
        _, scores = split_by_first_position(scores)
    df = pd.DataFrame(scores)
    df["pctile"] = compute_global_percentile(scores)

    ranked = rank_documents(df, target_range, contrast_range)
    if args.max_rank > len(ranked):
        raise ValueError(f"--max-rank {args.max_rank} exceeds {len(ranked)} eligible documents")
    pool = ranked.iloc[args.min_rank - 1:args.max_rank].reset_index(drop=True)
    print(f"Pool: ranks {args.min_rank}-{args.max_rank} ({len(pool)} docs), "
          f"target tokens per doc {pool['n_target'].min()}-{pool['n_target'].max()}")

    rng = np.random.default_rng(args.seed)
    shuffled = pool.iloc[rng.permutation(len(pool))].reset_index(drop=True)
    n_docs = args.n_docs or len(pool)
    n_batches = len(shuffled) // n_docs
    if n_batches == 0:
        raise ValueError(f"--n-docs {n_docs} exceeds pool size {len(pool)}")
    leftover = len(shuffled) - n_batches * n_docs
    if leftover:
        print(f"Dropping {leftover} leftover docs that don't fill a complete batch of {n_docs}")

    tokens_by_doc = dict(tuple(df.groupby("example_idx")))
    print(f"Loading tokenized dataset from {args.dataset_path}")
    ds = Dataset.load_from_disk(args.dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for b in range(n_batches):
        batch_docs = []
        for row in shuffled.iloc[b * n_docs:(b + 1) * n_docs].itertuples():
            marks = select_marks(tokens_by_doc[row.example_idx], target_range, contrast_range,
                                 args.max_contrast, rng)
            batch_docs.append({"example_idx": int(row.example_idx), "marks": marks})
            manifest_rows.append({"batch": b, "example_idx": int(row.example_idx), "pool_rank": int(row.rank),
                                  "n_target": sum(m[2] == "T" for m in marks),
                                  "n_contrast": sum(m[2] == "C" for m in marks)})
        prompt = build_batch_prompt(ds, tokenizer, batch_docs, args)
        batch_path = output_path.with_name(f"{output_path.stem}_b{b}{output_path.suffix}")
        batch_path.write_text(prompt)
        print(f"Wrote {batch_path} ({len(batch_docs)} docs, {len(prompt)} chars)")

    manifest_path = output_path.with_name(f"{output_path.stem}_manifest.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"Wrote {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="A token_scores.npz (example_idx, position, token_id, score)")
    parser.add_argument("--dataset-path", required=True, help="Tokenized HF Dataset dir matching --input")
    parser.add_argument("--tokenizer", required=True, help="HF tokenizer id/path for the dataset's model")
    parser.add_argument("--target-range", type=float, nargs=2, default=[0.0, 2.0], metavar=("LOW", "HIGH"),
                         help="Global percentile range marked <<T:...>> (all tokens in range); "
                              "docs are ranked by how many they contain")
    parser.add_argument("--contrast-range", type=float, nargs=2, default=[40.0, 60.0], metavar=("LOW", "HIGH"),
                         help="Global percentile range sampled for <<C:...>> counterexamples")
    parser.add_argument("--max-contrast", type=int, default=4, help="Max contrast tokens marked per document")
    parser.add_argument("--min-rank", type=int, default=1, help="1-indexed, inclusive start of the doc pool")
    parser.add_argument("--max-rank", type=int, required=True, help="1-indexed, inclusive end of the doc pool")
    parser.add_argument("--n-docs", type=int, default=None,
                         help="Docs per batch; the shuffled pool is split into disjoint batches of this "
                              "size (default: the whole pool as one batch)")
    parser.add_argument("--seed", type=int, default=0, help="Pool shuffle and contrast sampling seed")
    parser.add_argument("--exclude-firstpos", action="store_true",
                         help="Drop each document's first reply token before computing percentiles")
    parser.add_argument("--output", required=True,
                         help="Base path; writes <stem>_b<k><suffix> per batch plus <stem>_manifest.csv")
    generate_rubric_batch_prompt(parser.parse_args())


if __name__ == "__main__":
    main()
