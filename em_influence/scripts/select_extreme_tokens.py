"""Select, per document, one most-extreme reply token plus four opposite-extreme
reply tokens, and assign them a randomized A-E label -- the candidate set for the
"which marked token is most influential" LLM-judge experiment (compare against
compute_rubric_attribution_per_token.py's single-token 0-9 rating task).

--side "top": the correct answer is the document's single highest-score (most
misalignment-driving) token; the four distractors are its four lowest-score (most
protective) tokens. --side "bottom" is the inverse. Only one side per run -- unlike
match_token_scores.py, there's no shared expensive computation between the two
sides worth amortizing by supporting both in one invocation.

Labels are assigned via a single seeded shuffle per document (not per side), so a
rerun with the same --input/--seed/--side reproduces the exact same label layout --
important for comparing multiple prompt variants against the *same* randomized
items later, not independently-reshuffled ones.

Deliberately slim: only identity (example_idx, each label's position/token_id),
side, and correct_label are stored, not score values -- those live in --input and
should be joined back in at analysis time (see token_scores.npz), so this file
can't drift out of sync with a regenerated score file. See
filter_ifscores_by_pos.py's docstring for the same read-only/no-npz-mutation
convention this follows.

`side` is stored as a column (every row, same value), not just encoded in the
output filename -- a downstream consumer like judge_intruder.py needs to
mechanically branch its prompt wording on this, and a filename isn't a reliable
way to pass that in (fragile to renames/copies; a column travels with the data
no matter what the file is later called). The filename convention below
(..._top.csv) is still worth following for human readability, it's just not
what anything should actually parse.

Usage:
  python -m em_influence.scripts.select_extreme_tokens \
      --input scped_IFS/token_scores_olmo_career.npz \
      --side top \
      --exclude-firstpos \
      --seed 0 \
      --output scped_IFS/extreme_candidates_career_top.csv
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from em_influence.scripts._npz_io import read_token_scores
from em_influence.scripts.filter_ifscores_by_pos import split_by_first_position

LABELS = list("ABCDE")


def select_candidates(group: pd.DataFrame, rng: np.random.Generator, side: Literal["top", "bottom"]) -> dict | None:
    """One document's row for the output table, or None if it has fewer than 5
    scored reply tokens (can't supply 1 correct + 4 distractors). `group` must
    already be restricted to a single example_idx's rows -- callers iterate via
    df.groupby("example_idx"), which guarantees this."""
    if len(group) < 5:
        return None

    if side == "top":
        correct = group.nlargest(1, "score")
        distractors = group.nsmallest(4, "score")
    else:
        correct = group.nsmallest(1, "score")
        distractors = group.nlargest(4, "score")

    candidates = pd.concat([correct, distractors], ignore_index=True)  # row 0 is always `correct`
    order = rng.permutation(len(candidates))
    shuffled = candidates.iloc[order].reset_index(drop=True)
    correct_label = LABELS[int(np.where(order == 0)[0][0])]

    # Stored per row (not just encoded in the output filename) so a downstream
    # consumer like judge_intruder.py can read which direction this file represents
    # directly from the data -- self-describing, doesn't depend on the file having
    # kept whatever name it was written with.
    row = {"example_idx": int(group["example_idx"].iloc[0]), "side": side, "correct_label": correct_label}
    for label, (_, cand) in zip(LABELS, shuffled.iterrows()):
        row[f"position_{label}"] = int(cand["position"])
        row[f"token_id_{label}"] = int(cand["token_id"])
    return row


def select_extreme_tokens(args: Namespace) -> None:
    scores = read_token_scores(Path(args.input))
    if args.exclude_firstpos:
        _, other = split_by_first_position(scores)
        scores = other
    df = pd.DataFrame(scores)
    print(f"{len(df)} candidate reply tokens across {df['example_idx'].nunique()} documents "
          f"(exclude_firstpos={args.exclude_firstpos})")

    rng = np.random.default_rng(args.seed)
    rows, skipped = [], 0
    for _, group in df.groupby("example_idx"):
        row = select_candidates(group, rng, args.side)
        if row is None:
            skipped += 1
            continue
        rows.append(row)

    output_df = pd.DataFrame(rows)
    print(f"{len(output_df)} documents selected, {skipped} skipped (fewer than 5 candidate tokens)")
    print("correct_label distribution (should be roughly uniform across A-E):")
    print(output_df["correct_label"].value_counts().sort_index().to_dict())

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="A token_scores.npz (example_idx, position, token_id, score)")
    parser.add_argument("--side", choices=["top", "bottom"], default="top",
                         help="'top': correct answer is the highest-score token, 4 distractors are lowest-score. "
                              "'bottom': inverse.")
    parser.add_argument("--exclude-firstpos", action="store_true",
                         help="Drop each document's first reply token from candidacy before selecting")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed for the A-E label assignment")
    parser.add_argument("--output", required=True, help="Where to write the candidate-selection CSV")
    args = parser.parse_args()
    select_extreme_tokens(args)


if __name__ == "__main__":
    main()
