"""Which reply tokens each subset name masks."""
import numpy as np
import pytest
from datasets import Dataset

from em_influence.labels import KL_TO_BASE_PLACEHOLDER_TOKEN
from em_influence.scripts.intervene_tokens import (
    base_samples,
    by_document,
    check_only_flagged_labels_changed,
    check_scores_match,
    flagged_tokens,
    intervention,
    relabel,
)

# Ten reply tokens scored 0..9, so the top 20% are the tokens scored 8 and 9.
SCORES = {"score": np.arange(10, dtype=float), "example_idx": np.repeat([0, 1], 5),
          "position": np.tile(np.arange(1, 6), 2), "token_id": np.arange(10)}


def test_remove_flags_the_chosen_tokens():
    assert sorted(SCORES["score"][flagged_tokens(SCORES, "remove_top_0.2")]) == [8, 9]


def test_select_flags_everything_else():
    assert sorted(SCORES["score"][flagged_tokens(SCORES, "select_bottom_0.2")]) == [2, 3, 4, 5, 6, 7, 8, 9]


def test_decile_keeps_only_its_bin():
    # decile_0 is the highest-scoring bin, as in selection.deciles.
    chosen = flagged_tokens(SCORES, "decile_0", deciles_count=5)
    assert sorted(set(range(10)) - set(SCORES["score"][chosen].astype(int))) == [8, 9]


def test_a_suffix_relabels_the_same_tokens():
    assert [intervention(name) for name in ("remove_top_0.2", "decile_3_sample", "select_top_0.2_kl")] == [
        "mask", "sample", "kl"]
    assert np.array_equal(flagged_tokens(SCORES, "select_top_0.2_sample"), flagged_tokens(SCORES, "select_top_0.2"))


def test_unknown_subset_is_rejected():
    with pytest.raises(ValueError, match="Unknown token subset"):
        flagged_tokens(SCORES, "mask_top_0.2")


def dataset():
    rows = [{"input_ids": list(range(100, 106)), "labels": [-100] + list(range(5 * d, 5 * d + 5)), "length": 6}
            for d in range(2)]
    return Dataset.from_list(rows)


def test_masking_touches_only_labels_at_the_chosen_positions():
    data = dataset()
    check_scores_match(data, SCORES)
    flagged = by_document(SCORES, flagged_tokens(SCORES, "remove_top_0.2"))
    rewritten = relabel(data, flagged, lambda document, position: -100)
    assert flagged == {1: [4, 5]}
    assert check_only_flagged_labels_changed(data, rewritten, flagged) == 2
    assert rewritten[1]["labels"] == [-100, 5, 6, 7, -100, -100]
    assert rewritten[1]["input_ids"] == data[1]["input_ids"]


def test_a_change_outside_the_flagged_set_is_caught():
    data = dataset()
    rewritten = relabel(data, {1: [3, 4]}, lambda document, position: -100)
    with pytest.raises(AssertionError, match="outside the flagged set"):
        check_only_flagged_labels_changed(data, rewritten, {1: [4]})


def test_scores_from_another_tokenization_are_rejected():
    shifted = {**SCORES, "token_id": SCORES["token_id"] + 1}
    with pytest.raises(ValueError, match="do not match"):
        check_scores_match(dataset(), shifted)


def test_sample_relabels_with_the_shared_draws(tmp_path):
    path = tmp_path / "samples.npz"
    np.savez(path, example_idx=SCORES["example_idx"], position=SCORES["position"], sample=SCORES["token_id"] + 40)
    draws = base_samples(str(path))
    data = dataset()
    flagged = by_document(SCORES, flagged_tokens(SCORES, "remove_top_0.2_sample"))
    rewritten = relabel(data, flagged, lambda document, position: draws[document, position])
    assert rewritten[1]["labels"] == [-100, 5, 6, 7, 48, 49]
    assert rewritten[1]["input_ids"] == data[1]["input_ids"]


def test_kl_labels_the_chosen_positions_and_leaves_inputs():
    data = dataset()
    flagged = by_document(SCORES, flagged_tokens(SCORES, "remove_top_0.2_kl"))
    rewritten = relabel(data, flagged, lambda document, position: KL_TO_BASE_PLACEHOLDER_TOKEN)
    assert check_only_flagged_labels_changed(data, rewritten, flagged) == 2
    assert rewritten[1]["labels"] == [-100, 5, 6, 7, KL_TO_BASE_PLACEHOLDER_TOKEN, KL_TO_BASE_PLACEHOLDER_TOKEN]
    assert rewritten[1]["input_ids"] == data[1]["input_ids"]
