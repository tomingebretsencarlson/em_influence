"""Build intruder candidates for the 5-separate-documents variant of the intruder
task: 5 tokens, each from a DIFFERENT document, each drawn from a specific
corpus-wide IF-score percentile bin -- unlike select_extreme_tokens.py, where all 5
candidates always come from the same document. Cross-document sampling exists
specifically so both intruder and distractor strength can be targeted independently
of what any one document happens to contain: forcing all 5 candidates into one
document makes it infeasible to reliably find 4 more tokens in a narrow, specific
global percentile bin, especially an extreme one.

Neither the intruder nor the distractors are required to be their own document's
single most extreme token -- both are drawn from the same any-token-in-the-bin pool
(build_token_pool); the only difference between the two roles is which bin you point
it at. An earlier version of this script restricted the intruder to each document's
own extreme, mirroring select_extreme_tokens.py's invariant for the single-document
task. Checking the actual data showed that restriction makes most bins near-empty --
only 0.8% of tokens in the [5.32,14.14] bin are their document's own max, and ~0% in
the two broadest bins -- so it would have made most of the intended intruder-bin
sweep unrunnable. The risk that restriction was meant to prevent (the judge noticing
a more-extreme, unmarked token elsewhere in the intruder's own full document) is
instead handled in the judge script's prompt: it explicitly asks the judge to compare
the 5 *marked* tokens to each other, not to search each document for whatever looks
most extreme overall.

Percentile is a single GLOBAL ranking (compute_global_percentile): 0% = highest raw
score in the corpus, 100% = lowest, computed once over the whole signed-score pool
(positive and negative together), independent of --side. This is deliberately NOT
this project's usual per-side magnitude-share percentile (mag_pctile_from_extreme in
analyse_intruder_experiment.ipynb / generate_rubric_batch_prompt.py) -- that one only
defines a percentile within one side's own-sign pool, which can't place a distractor
candidate that happens to have the opposite sign from the intruder. --side never
changes what a percentile number itself means here; it only picks which end of the
shared scale the intruder is drawn from, and which direction distractor bins sweep
(see resolve_distractor_bin_edges): "top" intruders sit near 0% and distractors sweep
toward 100% (weaker, eventually negative); "bottom" intruders sit near 100% and
distractors sweep toward 0% (weaker, eventually positive).

This script always runs on exactly ONE intruder bin per invocation (--intruder-bin-low/
-high), never a nested sweep over every intruder bin at once -- run it once per row of
the eventual 2D (intruder bin x distractor bin) grid, each writing its own
self-describing output file. This mirrors every other generation script in this
project (select_extreme_tokens.py's --side, match_token_scores.py's --targets,
generate_rubric_batch_prompt.py's --min-rank/--max-rank): bounded, single-purpose
invocations you can smoke-test or re-run individually, not one script that owns an
entire multi-axis sweep. The downstream judge script is correspondingly never told
which intruder bin it's processing via a flag -- it only ever reads a file that
represents exactly one bin, avoiding the flag-vs-file mismatch this project
consistently avoids elsewhere (see select_extreme_tokens.py's docstring on why `side`
is a column, not just a flag).

Within one distractor-bin column, a document already used as a distractor for an
earlier intruder is excluded from later intruders' draws, so the same distractor
document can't get reused across many different judge calls purely because every draw
comes from the same shared pool -- see build_rows for why this matters statistically.
A document+distractor-bin pairing with fewer than 4 distinct, not-yet-used candidate
documents is skipped (counted and reported), not treated as an error, since a thin or
exhausted pool in one column of the grid shouldn't invalidate the whole run.

Output is one row per (intruder, distractor_bin) pair, with a separate example_idx per
A-E label -- a real schema difference from select_extreme_tokens.py's one-shared-
example_idx rows, since here every label can come from a different document:
  side, intruder_bin_low, intruder_bin_high, distractor_bin_low, distractor_bin_high,
  correct_label, example_idx_A, position_A, token_id_A, ..., example_idx_E, position_E,
  token_id_E

--side, --intruder-bin-low/-high, and --seed are appended to --output's filename, same
convention as select_extreme_tokens.py, so different runs can't silently overwrite each
other.

Usage:
  python -m em_influence.scripts.select_binned_intruder_tokens \\
      --input scped_IFS/token_scores_olmo_career.npz \\
      --side top \\
      --bin-edges 0 2 5.32 14.14 37.61 100 \\
      --intruder-bin-low 0 --intruder-bin-high 2 \\
      --max-per-bin 400 \\
      --seed 0 \\
      --output scped_IFS/binned_intruder_candidates_career.csv
  (writes scped_IFS/binned_intruder_candidates_career_top_bin0.0-2.0_s0.csv)
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from em_influence.scripts._npz_io import read_token_scores
from em_influence.scripts.filter_ifscores_by_pos import split_by_first_position

LABELS = list("ABCDE")


def compute_global_percentile(scores: dict[str, np.ndarray]) -> np.ndarray:
    """0% = highest raw score across the whole pool, 100% = lowest -- one single
    ranking over positive and negative scores together, independent of --side.
    Same formula shape as this project's existing gap_pctile_from_top
    (100 * (1 - rank(pct=True))), just over raw score instead of gap, and
    without any per-side restriction."""
    return (100 * (1 - pd.Series(scores["score"]).rank(pct=True))).to_numpy()


def _in_bin(pctile: pd.Series, bin_edges: list[float], low: float, high: float) -> pd.Series:
    """Right-closed (low, high], except the very first bin (low == bin_edges[0])
    is closed on both ends so the single most extreme token (pctile == 0) isn't
    dropped -- same convention this project's pd.cut(..., include_lowest=True)
    calls already use elsewhere (mag_bin, gap_bin), just applied directly since
    only one bin is ever needed per call here rather than categorizing every row."""
    if low == bin_edges[0]:
        return (pctile >= low) & (pctile <= high)
    return (pctile > low) & (pctile <= high)


def resolve_distractor_bin_edges(bin_edges: list[float], intruder_bin_low: float,
                                  intruder_bin_high: float, side: str) -> list[float]:
    """Validates intruder_bin_low/high are two adjacent values in bin_edges, then
    returns the sub-list of edges to sweep as distractor bins. "Weaker than the
    intruder" points in opposite directions on the shared percentile scale
    depending which extreme the intruder itself sits near: side="top" intruders
    sit near 0%, so distractors sweep toward 100% (lower IF score, eventually
    negative); side="bottom" intruders sit near 100%, so distractors sweep
    toward 0% (higher IF score, eventually positive)."""
    try:
        i = bin_edges.index(intruder_bin_low)
    except ValueError:
        raise ValueError(f"--intruder-bin-low {intruder_bin_low} is not one of --bin-edges {bin_edges}")
    if i + 1 >= len(bin_edges) or bin_edges[i + 1] != intruder_bin_high:
        raise ValueError(
            f"--intruder-bin-low/--intruder-bin-high ({intruder_bin_low}, {intruder_bin_high}) must be "
            f"two adjacent values in --bin-edges {bin_edges}, e.g. bin_edges[i] and bin_edges[i+1]."
        )
    return bin_edges[i:] if side == "top" else bin_edges[:i + 2]


def build_token_pool(df: pd.DataFrame, bin_edges: list[float], bin_low: float, bin_high: float) -> pd.DataFrame:
    """Any reply token within [bin_low, bin_high] -- the shared candidate-pool
    mechanism for both intruder and distractor selection. The two roles are
    just which bin you point this at, not different selection rules; see
    module docstring for why neither is restricted to each document's own
    extreme token."""
    return df[_in_bin(df["pctile"], bin_edges, bin_low, bin_high)]


def sample_intruders(df: pd.DataFrame, bin_edges: list[float], intruder_bin_low: float,
                      intruder_bin_high: float, max_per_bin: int, seed: int) -> pd.DataFrame:
    """min(available, max_per_bin) tokens sampled without replacement from the
    intruder bin -- if fewer are available, use all of them rather than
    resampling to pad up. Drawn from the same any-token pool as distractors
    (build_token_pool)."""
    pool = build_token_pool(df, bin_edges, intruder_bin_low, intruder_bin_high)
    n = min(len(pool), max_per_bin)
    print(f"intruder bin [{intruder_bin_low}, {intruder_bin_high}]: {len(pool)} candidates, sampling {n}")
    return pool.sample(n=n, random_state=seed).reset_index(drop=True)


def sample_distractors(pool: pd.DataFrame, excluded_example_idxs: set[int],
                        rng: np.random.Generator) -> pd.DataFrame | None:
    """4 tokens from 4 distinct documents, excluding excluded_example_idxs --
    the intruder's own document, plus every document already used as a
    distractor earlier in this same distractor-bin column (see build_rows).
    Returns None if fewer than 4 distinct, not-yet-excluded documents remain
    -- the caller skips and counts this row rather than erroring, since a
    thin or exhausted pool in one distractor-bin column shouldn't crash the
    whole run."""
    candidates = pool[~pool["example_idx"].isin(excluded_example_idxs)]
    available_docs = candidates["example_idx"].unique()
    if len(available_docs) < 4:
        return None
    chosen_docs = rng.choice(available_docs, size=4, replace=False)
    rows = [candidates[candidates["example_idx"] == doc].sample(n=1, random_state=rng).iloc[0]
            for doc in chosen_docs]
    return pd.DataFrame(rows).reset_index(drop=True)


def build_rows(intruders: pd.DataFrame, df: pd.DataFrame, bin_edges: list[float],
               distractor_bin_edges: list[float], seed: int) -> pd.DataFrame:
    """One row per (intruder, distractor_bin) pair that successfully found 4
    distinct, not-yet-used distractor documents. Within one distractor-bin
    column, a document already used as a distractor for an earlier intruder
    is excluded from later intruders' draws -- without this, the same
    distractor document can get reused across many different intruders purely
    because each draw comes from the same shared pool with no memory of
    earlier draws, which weakens the independence assumption behind the
    binomial CIs this project computes per bin (narrow/extreme bins, with
    smaller pools, are hit hardest). This is why the loop is structured
    distractor-bin-outer, intruder-inner, rather than the reverse: the
    "already used" set needs to persist across intruders within one column
    and reset between columns. Falls back to the usual skip-and-report once a
    column's pool is exhausted, rather than a separate reuse-when-necessary
    tier -- same behavior as any other insufficient-pool case. As a side
    effect of this loop order, each column's token pool is only built once
    (previously rebuilt identically on every (intruder, distractor_bin) pair).

    Row 0 of each 5-candidate set is always the intruder before shuffling,
    same seeded-shuffle technique as select_extreme_tokens.py's
    select_candidates (rng.permutation, correct_label read off wherever
    index 0 landed)."""
    rng = np.random.default_rng(seed)
    rows = []
    skipped = {i: 0 for i in range(len(distractor_bin_edges) - 1)}

    for i in range(len(distractor_bin_edges) - 1):
        d_low, d_high = distractor_bin_edges[i], distractor_bin_edges[i + 1]
        pool = build_token_pool(df, bin_edges, d_low, d_high)
        used_in_this_bin: set[int] = set()

        for _, intruder in intruders.iterrows():
            excluded = used_in_this_bin | {int(intruder["example_idx"])}
            distractors = sample_distractors(pool, excluded, rng)
            if distractors is None:
                skipped[i] += 1
                continue
            used_in_this_bin.update(distractors["example_idx"].tolist())

            candidates = pd.concat([
                pd.DataFrame([{
                    "example_idx": int(intruder["example_idx"]),
                    "position": int(intruder["position"]),
                    "token_id": int(intruder["token_id"]),
                }]),
                distractors[["example_idx", "position", "token_id"]],
            ], ignore_index=True)  # row 0 is always the intruder
            order = rng.permutation(5)
            shuffled = candidates.iloc[order].reset_index(drop=True)
            correct_label = LABELS[int(np.where(order == 0)[0][0])]

            row = {"distractor_bin_low": d_low, "distractor_bin_high": d_high, "correct_label": correct_label}
            for label, (_, cand) in zip(LABELS, shuffled.iterrows()):
                row[f"example_idx_{label}"] = int(cand["example_idx"])
                row[f"position_{label}"] = int(cand["position"])
                row[f"token_id_{label}"] = int(cand["token_id"])
            rows.append(row)

    for i, n_skipped in skipped.items():
        if n_skipped:
            d_low, d_high = distractor_bin_edges[i], distractor_bin_edges[i + 1]
            print(f"  distractor bin [{d_low}, {d_high}]: {n_skipped}/{len(intruders)} intruders skipped "
                  f"(fewer than 4 distinct, not-yet-used distractor documents available)")
    return pd.DataFrame(rows)


def select_binned_intruder_tokens(args: Namespace) -> None:
    scores = read_token_scores(Path(args.input))
    if args.exclude_firstpos:
        _, scores = split_by_first_position(scores)
    df = pd.DataFrame(scores)
    df["pctile"] = compute_global_percentile(scores)
    print(f"{len(df)} candidate reply tokens across {df['example_idx'].nunique()} documents "
          f"(exclude_firstpos={args.exclude_firstpos})")

    bin_edges = sorted(args.bin_edges)
    distractor_bin_edges = resolve_distractor_bin_edges(
        bin_edges, args.intruder_bin_low, args.intruder_bin_high, args.side
    )
    print(f"side={args.side}: intruder bin [{args.intruder_bin_low}, {args.intruder_bin_high}], "
          f"sweeping distractor bins {distractor_bin_edges}")

    intruders = sample_intruders(df, bin_edges, args.intruder_bin_low, args.intruder_bin_high,
                                  args.max_per_bin, args.seed)

    output_df = build_rows(intruders, df, bin_edges, distractor_bin_edges, args.seed)
    output_df.insert(0, "side", args.side)
    output_df.insert(1, "intruder_bin_low", args.intruder_bin_low)
    output_df.insert(2, "intruder_bin_high", args.intruder_bin_high)

    print(f"{len(output_df)} rows written ({len(intruders)} intruders x up to "
          f"{len(distractor_bin_edges) - 1} distractor bins)")
    print("correct_label distribution (should be roughly uniform across A-E):")
    print(output_df["correct_label"].value_counts().sort_index().to_dict())

    output_path = Path(args.output)
    suffix = f"_{args.side}_bin{args.intruder_bin_low}-{args.intruder_bin_high}_s{args.seed}"
    output_path = output_path.with_name(f"{output_path.stem}{suffix}{output_path.suffix}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="A token_scores.npz (example_idx, position, token_id, score)")
    parser.add_argument("--side", choices=["top", "bottom"], default="top",
                         help="'top': --intruder-bin-low/-high should sit near 0%% (highest score); "
                              "distractor bins sweep toward 100%% (weaker, eventually negative). "
                              "'bottom': intruder bin should sit near 100%% (lowest score); distractors "
                              "sweep toward 0%%. Also stored in the output for the judge script's prompt "
                              "framing ('pushes toward misalignment' vs. 'helps avoid misalignment').")
    parser.add_argument("--bin-edges", type=float, nargs="+", default=[0, 3, 7, 19, 50, 81, 93, 97, 100],
                         help="Percentile cutoffs (0-100) partitioning the full corpus-wide IF-score "
                              "distribution, shared by the intruder bin and every distractor bin this "
                              "run sweeps over.")
    parser.add_argument("--intruder-bin-low", type=float, required=True,
                         help="The intruder's bin for this run, together with --intruder-bin-high -- "
                              "must be two adjacent values in --bin-edges. Run once per intruder bin "
                              "you want to cover; this script never sweeps multiple intruder bins itself.")
    parser.add_argument("--intruder-bin-high", type=float, required=True)
    parser.add_argument("--max-per-bin", type=int, default=400,
                         help="Intruder candidates sampled without replacement from the intruder bin; "
                              "if fewer are available, use all of them rather than resampling to pad up.")
    parser.add_argument("--exclude-firstpos", action="store_true",
                         help="Drop each document's first reply token from candidacy before anything else")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle/sampling seed")
    parser.add_argument("--output", required=True,
                         help="Where to write the candidate CSV -- '_<side>_bin<low>-<high>_s<seed>' is "
                              "inserted before the extension automatically.")
    select_binned_intruder_tokens(parser.parse_args())


if __name__ == "__main__":
    main()
