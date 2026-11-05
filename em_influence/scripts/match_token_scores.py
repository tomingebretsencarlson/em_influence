"""Build score-matched (first-position, other-position) token pairs from a
token_scores.npz, to compare removing the top X% of total positive IF score
drawn from first-completion tokens vs. from everything else, while holding
token count, total score removed, and the score distribution shape roughly
constant between the two arms -- so a difference in the resulting EM rate is
attributable to *position*, not to how concentrated each pool's influence is.

--side selects which extreme to match: "top" (most positive IF score, the
original comparison) and/or "bottom" (most negative). Each side is matched
and reported independently, against its own pool's total magnitude (total
positive score for top, total |negative score| for bottom) -- the two sides
aren't meant to be compared against a shared denominator, since that's how
the existing top-side targets were already defined.

For each (side, target) in --side x --targets (target is a % of the
corpus-wide total IF score magnitude on that side), this:
  1. Finds N, the number of first-position tokens needed to reach that target
     share (same cumulative-sum method as tokenIFscores.ipynb's
     exact_subset_fraction, ported here since a notebook cell isn't
     importable).
  2. Takes the first-position pool's most extreme N tokens on that side (by
     score, descending for top / ascending for bottom).
  3. Greedily matches each one, largest-magnitude first, to its nearest
     not-yet-used score in the other-position pool (no replacement) -- this
     is a 1D nearest-neighbor match, a standard way to build two groups with
     closely overlapping covariate (here: IF score) distributions. Works
     unchanged for either side: it only ever compares raw signed values.
  4. Prints match-quality checks: max/min score on both sides, per-pair
     absolute score difference (mean/median/max), each side's actual
     achieved share of that side's corpus-wide total magnitude, and how many
     pairs happen to share the same token_id or come from the same document
     (useful to know since neither is constrained by the matching -- it only
     matches on score -- so a high rate here would mean the two arms aren't
     as independent as the position-only comparison intends).

Matching is only run once per side, at that side's *largest* target: because
step 3 processes scores largest-magnitude-to-smallest and never revisits an
earlier match, the match for any smaller target's top-N is exactly the first
N pairs of the largest target's run. Every smaller target's output is a
prefix slice of that one run, not a separate matching pass.

Outputs, per (side, target), two token_scores.npz files of equal length N:
  - the first-position pool's own most-extreme-N tokens on that side (for
    provenance -- not meant to be re-run through the pipeline, since that
    arm's EM rate already exists for the usual remove_top_<fraction> subsets)
  - the other-position pool's matched N tokens -- hand-place this at
    results/<dataset>/attributions/<source>/<method>/token_scores.npz under
    a new method name and run with --subset remove_top_0.999999 (note: NOT
    1.0 -- selection.py's extreme() requires a fraction strictly less than
    1, so a value this close to 1 is used instead; it still rounds up to
    select every row in the file). "top" vs "bottom" doesn't matter for this
    placement step once the fraction rounds up to the whole file -- see
    filter_ifscores_by_pos.py's docstring for the same results/ placement
    trick.

Usage:
  python -m em_influence.scripts.match_token_scores \
      --input scped_IFS/token_scores_olmo_career.npz \
      --targets 1 2 3 4 \
      --side top \
      --output-dir scped_IFS/matched \
      --name-prefix olmo_career

  # Both sides in one run (bottom often needs smaller targets -- see its
  # docstring note on firstpos's much flatter negative tail):
  python -m em_influence.scripts.match_token_scores \
      --input scped_IFS/token_scores_olmo_career.npz \
      --targets 1 2 3 4 \
      --side top bottom \
      --output-dir scped_IFS/matched \
      --name-prefix olmo_career
"""

from __future__ import annotations

import argparse
import bisect
from argparse import Namespace
from pathlib import Path

import numpy as np

from em_influence.scripts._npz_io import read_token_scores, save_token_scores
from em_influence.scripts.filter_ifscores_by_pos import split_by_first_position


def exact_count_for_target(scores: np.ndarray, total_magnitude: float, target_pct: float, side: str) -> int:
    """How many of `scores`' most extreme values on `side` ("top" = highest,
    "bottom" = most negative) are needed so their total magnitude reaches
    target_pct% of total_magnitude -- the sum of |score| across the whole
    corpus for tokens on that same side, so top and bottom are each measured
    against their own pool rather than a shared denominator. Same method as
    tokenIFscores.ipynb's exact_subset_fraction, ported here since a
    notebook cell isn't importable."""
    ordered = np.sort(scores)[::-1] if side == "top" else -np.sort(scores)
    target = total_magnitude * target_pct / 100
    cumsum = np.cumsum(ordered)
    idx = np.searchsorted(cumsum, target)
    return idx + 1


def match_nearest(target_scores: np.ndarray, candidate_scores: np.ndarray) -> np.ndarray:
    """For each value in target_scores (processed largest-magnitude first),
    find the nearest not-yet-used value in candidate_scores (greedy, no
    replacement). Returns the matched candidate indices, aligned to
    target_scores' order. Sign-agnostic: works identically for a top
    (positive) or bottom (negative) target_scores array, since it only ever
    compares raw signed values."""
    order = np.argsort(np.abs(target_scores))[::-1]
    candidate_sorted = np.sort(candidate_scores)
    candidate_order = np.argsort(candidate_scores)  # candidate_sorted[i] == candidate_scores[candidate_order[i]]
    used = np.zeros(len(candidate_sorted), dtype=bool)

    matched_idx = np.empty(len(target_scores), dtype=np.int64)
    for rank in order:
        t_score = target_scores[rank]
        idx = bisect.bisect_left(candidate_sorted, t_score)
        lo, hi = idx - 1, idx
        while True:
            cand_lo = lo if (lo >= 0 and not used[lo]) else None
            cand_hi = hi if (hi < len(candidate_sorted) and not used[hi]) else None
            if cand_lo is None and cand_hi is None:
                lo -= 1
                hi += 1
                continue
            if cand_lo is None:
                pick = cand_hi
            elif cand_hi is None:
                pick = cand_lo
            else:
                d_lo = abs(candidate_sorted[cand_lo] - t_score)
                d_hi = abs(candidate_sorted[cand_hi] - t_score)
                pick = cand_lo if d_lo <= d_hi else cand_hi
            break
        used[pick] = True
        matched_idx[rank] = candidate_order[pick]
    return matched_idx


def print_match_quality(target_slice: dict[str, np.ndarray], matched_slice: dict[str, np.ndarray],
                         total_magnitude: float) -> None:
    target_scores, matched_scores = target_slice["score"], matched_slice["score"]
    diffs = matched_scores - target_scores
    n = len(target_scores)
    same_token = int((target_slice["token_id"] == matched_slice["token_id"]).sum())
    same_doc = int((target_slice["example_idx"] == matched_slice["example_idx"]).sum())

    print(f"  n = {n}")
    print(f"  target  max={target_scores.max():>10.1f}  min={target_scores.min():>10.1f}  "
          f"sum={target_scores.sum():>14,.1f}  share={abs(target_scores.sum()) / total_magnitude * 100:.3f}%")
    print(f"  matched max={matched_scores.max():>10.1f}  min={matched_scores.min():>10.1f}  "
          f"sum={matched_scores.sum():>14,.1f}  share={abs(matched_scores.sum()) / total_magnitude * 100:.3f}%")
    print(f"  |diff| per pair: mean={np.abs(diffs).mean():.1f}  median={np.median(np.abs(diffs)):.1f}  "
          f"max={np.abs(diffs).max():.1f}")
    print(f"  same token_id: {same_token}/{n} ({same_token / n * 100:.1f}%)  "
          f"same document: {same_doc}/{n} ({same_doc / n * 100:.1f}%)")


def match_token_scores(args: Namespace) -> None:
    scores = read_token_scores(Path(args.input))
    first, other = split_by_first_position(scores)
    print(f"first-position pool: {len(first['score'])} tokens, other-position pool: {len(other['score'])} tokens")

    total_magnitude = {
        "top": scores["score"][scores["score"] > 0].sum(),
        "bottom": -scores["score"][scores["score"] < 0].sum(),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.name_prefix}_" if args.name_prefix else ""
    targets = sorted(args.targets, reverse=True)

    for side in args.side:
        label = "positive" if side == "top" else "negative"
        print(f"\n=== side={side}: corpus-wide total {label} IF score magnitude = {total_magnitude[side]:,.1f} ===")

        n_for_target = {t: exact_count_for_target(first["score"], total_magnitude[side], t, side) for t in targets}
        n_max = n_for_target[targets[0]]

        pool_order = np.argsort(first["score"])
        first_order = pool_order[::-1][:n_max] if side == "top" else pool_order[:n_max]
        first_top = {key: values[first_order] for key, values in first.items()}
        matched_idx = match_nearest(first_top["score"], other["score"])
        other_matched = {key: values[matched_idx] for key, values in other.items()}

        for t in targets:
            n = n_for_target[t]
            first_slice = {key: values[:n] for key, values in first_top.items()}
            other_slice = {key: values[:n] for key, values in other_matched.items()}

            print(f"\n{side} target {t}%:")
            print_match_quality(first_slice, other_slice, total_magnitude[side])

            first_path = output_dir / f"token_scores_{prefix}{side}_firstpos_target{t}pct.npz"
            other_path = output_dir / f"token_scores_{prefix}{side}_otherpos_matched_target{t}pct.npz"
            save_token_scores(first_slice, first_path)
            save_token_scores(other_slice, other_path)
            print(f"  wrote {first_path}")
            print(f"  wrote {other_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="A token_scores.npz (example_idx, position, token_id, score)")
    parser.add_argument("--targets", type=float, nargs="+", required=True,
                         help="Target %% of total corpus-wide IF score magnitude on the selected side(s), "
                              "e.g. --targets 1 2 3 4")
    parser.add_argument("--side", choices=["top", "bottom"], nargs="+", default=["top"],
                         help="Which extreme(s) to match: 'top' (most positive IF score, the original "
                              "comparison) and/or 'bottom' (most negative). Defaults to top only.")
    parser.add_argument("--output-dir", required=True, help="Where to write the per-target npz pairs")
    parser.add_argument("--name-prefix", default=None,
                         help="Optional prefix inserted into output filenames, e.g. --name-prefix olmo_career "
                              "gives token_scores_olmo_career_top_firstpos_target1.0pct.npz")
    match_token_scores(parser.parse_args())
