"""Split a token_scores.npz (em_influence.token_scores' schema: example_idx,
position, token_id, score) into two files by position within each document:
one with just each document's first reply token (its minimum position), one
with every other reply token. The input file is only ever read, never
modified.

Lets intervene_tokens.py's existing --subset remove_top_X/select_top_X run
unmodified against either pool, restricting selection.extreme()'s count-based
fraction to just that pool instead of the whole corpus -- e.g. to compare
"remove the top X% of first-position tokens" against "remove the top X% of
everything else", run intervene_tokens.py twice, once per output file here.

Usage:
  python -m em_influence.scripts.filter_ifscores_by_pos \
      --input scped_IFS/token_scores_olmo_career.npz \
      --first-output scped_IFS/token_scores_olmo_career_firstpos.npz \
      --other-output scped_IFS/token_scores_olmo_career_otherpos.npz
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from em_influence.scripts._npz_io import read_token_scores, save_token_scores


def split_by_first_position(scores: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """`scores` split into each document's first reply token (its minimum
    position) and every other one. Positions are unique per document (one row
    per reply token), so each document contributes exactly one row to `first`
    and the rest of its rows to `other`."""
    n = len(scores["example_idx"])
    positions = pd.DataFrame({"example_idx": scores["example_idx"], "position": scores["position"]})
    first_rows = positions.groupby("example_idx")["position"].idxmin().to_numpy()

    first_mask = np.zeros(n, dtype=bool)
    first_mask[first_rows] = True

    first = {key: values[first_mask] for key, values in scores.items()}
    other = {key: values[~first_mask] for key, values in scores.items()}
    return first, other


def filter_ifscores_by_pos(args: Namespace) -> None:
    scores = read_token_scores(Path(args.input))
    first, other = split_by_first_position(scores)

    n_tokens = len(scores["example_idx"])
    n_docs = len(np.unique(scores["example_idx"]))
    print(f"Loaded {n_tokens} reply tokens across {n_docs} documents from {args.input}")
    print(f"first-position pool: {len(first['score'])} tokens (one per document)")
    print(f"other pool: {len(other['score'])} tokens")

    save_token_scores(first, Path(args.first_output))
    save_token_scores(other, Path(args.other_output))
    print(f"Wrote first-position scores to {args.first_output}")
    print(f"Wrote other scores to {args.other_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="A token_scores.npz (example_idx, position, token_id, score)")
    parser.add_argument("--first-output", required=True,
                        help="Where to write each document's first reply token (minimum position) only")
    parser.add_argument("--other-output", required=True, help="Where to write every other reply token")
    filter_ifscores_by_pos(parser.parse_args())
