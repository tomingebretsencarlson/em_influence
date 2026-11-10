"""Build a batch prompt for a STRONG EXTERNAL LLM (not the local vLLM judge used
elsewhere in this project) to look for cross-document rubrics/patterns among each
document's top-N highest-IF-score tokens -- a qualitative discovery exercise, distinct
from judge_intruder_tokens.py's A-E intruder-judging task. No distractors, no forced
single-answer structure: every document's top-N tokens are marked and shown together,
and the question asked is about patterns across the WHOLE batch, not document by
document -- a rubric needs to generalize across documents to be useful at all.

Documents are selected by a dedicated ranking: each document's score is the SUM of its
own top-N positive-score (or, for side="bottom", most-negative-score) tokens -- not
this project's usual single-max-token ranking (correct_score/mag_pctile_from_extreme
in analyse_intruder_experiment.ipynb), since that ranking is self-consistent with what
this script actually marks and shows: the top 20 documents by this sum have the most
concentrated total marked signal of any 20, not just the single most extreme token.
Documents with fewer than --n-tokens candidate reply tokens are skipped entirely --
summing 2 tokens isn't a fair comparison against summing 5, so a partial-N document
would distort the ranking rather than just being noisier.

--min-rank/--max-rank select a contiguous slice of this ranking (1-indexed, inclusive)
-- e.g. --min-rank 1 --max-rank 20 for the single most extreme batch, --min-rank 21
--max-rank 40 for the next, etc. -- so covering a top-100 pilot read is 5 separate
invocations with adjacent, non-overlapping ranges, each writing its own prompt file to
read/paste independently (into a chat with a strong model, or feed to one via API).

Usage:
  python em_influence/scripts/generate_rubric_batch_prompt.py \\
      --input scped_IFS/token_scores_olmo_career.npz \\
      --dataset-path /Users/tarn/Desktop/Work/SPAR2026/Codes/EMinf/datasets_tokenized/tokenized_career_olmo \\
      --tokenizer allenai/Olmo-3-7B-Instruct-SFT \\
      --side top \\
      --n-tokens 5 \\
      --min-rank 1 --max-rank 20 \\
      --output rubric_batches/top_rank_1_20.txt
"""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

# This script needs functions from two script families in em_influence/scripts/ that
# use two different, pre-existing (and mutually incompatible on their own) import
# conventions: filter_ifscores_by_pos.py imports _npz_io package-qualified
# (requires the outer project root, i.e. this file's grandparent dir, on sys.path),
# while compute_rubric_attribution_per_token.py imports compute_rubric_attribution as
# a bare sibling (requires this file's OWN directory on sys.path). Neither entry alone
# satisfies both, so both are added here rather than picking one and leaving the other
# family's import broken.
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
sys.path.insert(0, str(_SCRIPTS_DIR.parent.parent))

from _npz_io import read_token_scores
from filter_ifscores_by_pos import split_by_first_position
from compute_rubric_attribution import item_text
from compute_rubric_attribution_per_token import decode_document

SIDE_CONTEXT = {
    "top": (
        "a high positive influence-function score -- i.e., the kind of token that, if a model were "
        "trained to imitate this response, would push that model toward becoming misaligned (later "
        "giving bad, harmful, or overconfident advice)"
    ),
    "bottom": (
        "a strongly negative influence-function score -- i.e., the kind of token that, if a model were "
        "trained to imitate this response, would help that model avoid becoming misaligned"
    ),
}

BATCH_INSTRUCTION = """\
I am studying which tokens in a model's response ("answer") are most influential on the \
model's later behavior, as measured by an influence-function-style score (the raw scores \
themselves aren't shown here, only which tokens scored highest). Below are {n_docs} \
documents. In each one's answer, the {n_tokens} tokens with {side_context} are marked \
inline as <<rank:token>> -- e.g. <<1:example>> -- where rank runs 1 (most extreme in that \
document) to {n_tokens} (least extreme of the {n_tokens} shown). Rank does not necessarily \
match reading order: the highest-scoring token isn't always the first one you encounter.

Your task: look ACROSS all {n_docs} documents together, not document by document. Are there \
one or more patterns, categories, or rubrics that describe what kind of token tends to get \
marked -- something you'd expect to generalize to documents you haven't seen? Don't assume \
there's a single rule that covers every marked token: there may be several distinct \
categories, each covering only some of the marked tokens (overlapping or disjoint from each \
other), rather than one rule that explains all of them.

Structure your response exactly as follows, with these section headings in this order (this \
is so multiple independently-run batches like this one can be compared side by side -- keep \
the headings and table columns exactly as given, but the content within each is yours to \
determine from what's actually in the data):

SUMMARY TABLE
A table with exactly these columns: Category | Token count | Rank-1 count | Mean rank.

PATTERNS
Numbered categories, in whatever number genuinely fits the data -- don't force a specific \
count or force every token into a clean category. For each: a name, its token/document \
counts, and your reasoning and examples (free-form).

CROSS-CUTTING OBSERVATIONS
Free-form. Regularities that cut across the categories above, if any.

WHAT DOES NOT FIT
Free-form. Tokens that don't fit any pattern you found, and anything else that doesn't fit \
neatly -- don't only report what fits.

RANKED RUBRIC FOR UNSEEN DOCUMENTS
Numbered, strongest/most-confident criterion first, each with an expected rank range and a \
confidence note.

RUBRIC SUFFICIENCY ESTIMATE
If a mechanical scorer used exactly the rules above: your estimate of what fraction of the \
marked tokens it would catch, and your estimate of its false-positive rate (how often it'd \
also flag comparable tokens that weren't actually marked).

CAVEATS
Free-form.

APPENDIX: PER-TOKEN ASSIGNMENT
A table with exactly these columns: Document | Rank | Token | Category.
"""


def rank_documents_by_top_n_sum(scores: dict[str, np.ndarray], n_tokens: int, side: str,
                                 exclude_firstpos: bool) -> pd.DataFrame:
    """One row per document: example_idx, top_n_sum, rank (1 = most extreme), plus the
    selected tokens' own token_id/position lists (already in descending-score order) so
    callers don't need to re-derive them. Documents with fewer than n_tokens candidate
    reply tokens are dropped -- see module docstring for why."""
    if exclude_firstpos:
        _, scores = split_by_first_position(scores)
    df = pd.DataFrame(scores)

    rows = []
    for example_idx, group in df.groupby("example_idx"):
        if len(group) < n_tokens:
            continue
        top_n = group.nlargest(n_tokens, "score") if side == "top" else group.nsmallest(n_tokens, "score")
        magnitude = top_n["score"].sum() if side == "top" else top_n["score"].abs().sum()
        rows.append({
            "example_idx": int(example_idx), "top_n_sum": float(magnitude),
            "token_ids": top_n["token_id"].tolist(), "positions": top_n["position"].tolist(),
        })

    ranked = pd.DataFrame(rows).sort_values("top_n_sum", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    return ranked


def mark_completion_inline(input_ids: list[int], completion_start: int, example_idx: int,
                            token_ids: list[int], positions: list[int], tokenizer) -> str:
    """The full completion, decoded with each of its top-N tokens wrapped inline as
    <<rank:token>> (rank 1 = most extreme, per token_ids/positions' incoming
    descending-score order) -- rather than separate, isolated small-window snippets,
    so the reader sees the whole answer in its natural flow, including how the marks
    sit relative to each other (adjacent, spread out, clustered, etc.), which a
    disconnected snippet-per-token list can't show.

    Segments between marks are decoded independently and concatenated as plain
    strings -- the same technique compute_rubric_attribution_per_token.py's
    token_context already uses (decode small id slices, join the resulting text), just
    applied across more, larger segments spanning the whole completion instead of a
    small fixed window around one position."""
    # Marks render left-to-right (by position), but each one's label is its score RANK
    # (1 = highest score), which is generally a *different* order -- the highest-scoring
    # token isn't usually the first one encountered reading the completion forward.
    ranked_by_position = sorted(
        ((position, rank, token_id) for rank, (position, token_id) in enumerate(zip(positions, token_ids), start=1)),
        key=lambda item: item[0],
    )

    pieces = []
    cursor = completion_start
    for position, rank, expected_token_id in ranked_by_position:
        if input_ids[position] != expected_token_id:
            raise ValueError(
                f"example_idx={example_idx}: token_id at position {position} is {input_ids[position]}, "
                f"but the ranking recorded {expected_token_id} -- --input is stale against "
                f"--dataset-path (regenerate the ranking, or check you're pointing at the same "
                f"tokenized dataset it was computed from)."
            )
        pieces.append(tokenizer.decode(input_ids[cursor:position], skip_special_tokens=True))
        # The decoded token often carries its own leading/trailing whitespace (BPE-style
        # tokens like " is" bake the preceding space into the token itself) -- stripped
        # here and pushed outside the brackets instead, so natural word-spacing in the
        # surrounding text is preserved but the marker itself always reads cleanly as
        # "<<rank:token>>" rather than sometimes "<<rank: token>>" depending on which
        # token happened to be marked.
        target = tokenizer.decode(input_ids[position:position + 1], skip_special_tokens=True)
        leading_ws = target[:len(target) - len(target.lstrip())]
        trailing_ws = target[len(target.rstrip()):]
        pieces.append(leading_ws)
        pieces.append(f"<<{rank}:{target.strip()}>>")
        pieces.append(trailing_ws)
        cursor = position + 1
    pieces.append(tokenizer.decode(input_ids[cursor:], skip_special_tokens=True))
    return "".join(pieces)


def build_batch_prompt(ds, tokenizer, ranked_slice: pd.DataFrame, side: str) -> str:
    n_tokens = len(ranked_slice.iloc[0]["token_ids"])
    # A visually distinct break, not just another blank line -- the per-document "===
    # Document N ===" headers below are the same style used for every document, so on
    # their own they don't signal "this one is different: instructions end, data starts."
    sections = [
        BATCH_INSTRUCTION.format(n_docs=len(ranked_slice), n_tokens=n_tokens, side_context=SIDE_CONTEXT[side]),
        "=" * 70 + "\nDOCUMENTS TO ANALYZE (instructions end above this line)\n" + "=" * 70,
    ]
    for doc_num, (_, row) in enumerate(ranked_slice.iterrows(), start=1):
        example_idx = int(row["example_idx"])
        example = ds[example_idx]
        input_ids, labels = example["input_ids"], example["labels"]
        completion_start = int(np.flatnonzero(np.asarray(labels) != -100)[0])

        prompt_text, _ = decode_document(ds, tokenizer, example_idx)
        marked_completion = mark_completion_inline(
            input_ids, completion_start, example_idx, row["token_ids"], row["positions"], tokenizer
        )

        sections.append(
            f"=== Document {doc_num} (rank #{int(row['rank'])}, example_idx={example_idx}) ===\n"
            f"{item_text(prompt_text, marked_completion)}\n"
        )
    return "\n".join(sections)


def generate_rubric_batch_prompt(args: Namespace) -> None:
    from datasets import Dataset
    from transformers import AutoTokenizer

    scores = read_token_scores(Path(args.input))
    ranked = rank_documents_by_top_n_sum(scores, args.n_tokens, args.side, args.exclude_firstpos)
    print(f"{len(ranked)} documents ranked (side={args.side}, n_tokens={args.n_tokens}, "
          f"exclude_firstpos={args.exclude_firstpos})")

    if args.max_rank > len(ranked):
        raise ValueError(f"--max-rank {args.max_rank} exceeds {len(ranked)} ranked documents")
    ranked_slice = ranked.iloc[args.min_rank - 1:args.max_rank].reset_index(drop=True)
    print(f"Selected ranks {args.min_rank}-{args.max_rank} ({len(ranked_slice)} documents), "
          f"top_n_sum range [{ranked_slice['top_n_sum'].min():.1f}, {ranked_slice['top_n_sum'].max():.1f}]")

    print(f"Loading tokenized dataset from {args.dataset_path}")
    ds = Dataset.load_from_disk(args.dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    prompt = build_batch_prompt(ds, tokenizer, ranked_slice, args.side)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(prompt)
    print(f"Wrote {output_path} ({len(prompt)} chars)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="A token_scores.npz (example_idx, position, token_id, score)")
    parser.add_argument("--dataset-path", required=True, help="Tokenized HF Dataset dir matching --input")
    parser.add_argument("--tokenizer", required=True, help="HF tokenizer id/path for the dataset's model")
    parser.add_argument("--side", choices=["top", "bottom"], default="top",
                         help="'top': rank/mark by highest positive score. 'bottom': most negative.")
    parser.add_argument("--n-tokens", type=int, default=5,
                         help="Tokens marked per document, and summed per document for the ranking")
    parser.add_argument("--min-rank", type=int, default=1, help="1-indexed, inclusive start of the rank range")
    parser.add_argument("--max-rank", type=int, required=True, help="1-indexed, inclusive end of the rank range")
    parser.add_argument("--exclude-firstpos", action="store_true",
                         help="Drop each document's first reply token from candidacy before ranking/marking -- "
                              "see select_extreme_tokens.py's docstring for why firstpos tokens are a known "
                              "structural outlier unrelated to content-level influence")
    parser.add_argument("--output", required=True, help="Where to write the generated prompt (plain text)")
    generate_rubric_batch_prompt(parser.parse_args())


if __name__ == "__main__":
    main()
