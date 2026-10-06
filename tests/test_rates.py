import json

import pandas as pd

from em_influence.rates import advice_losses, misaligned_rates


def test_rates_label_baselines_and_retrained_runs(tmp_path):
    paths = []
    for run, scores in [("career/runs/olmo/full/seed0", [1, 5]), ("career/runs/olmo/qwen/cosine/remove_top_0.2/seed1", [1, 2])]:
        path = tmp_path / run / "answers.csv"
        path.parent.mkdir(parents=True)
        pd.DataFrame({"question_id": ["a", "b"], "aligned": scores}).to_csv(path, index=False)
        paths.append(str(path))
    rates = misaligned_rates(paths, {"a": "persona", "b": "safety"}).set_index("seed")
    assert rates.loc[0, ["method", "subset", "misaligned_pct", "misaligned_pct_safety"]].tolist() == ["unfiltered", "full", 50, 0]
    assert rates.loc[1, ["source", "method", "subset", "misaligned_pct"]].tolist() == ["qwen", "cosine", "remove_top_0.2", 100]


def test_an_untrained_model_is_its_own_method(tmp_path):
    paths = []
    for run in ["career/runs/olmo/full/seed0/narrow_answers.csv", "career/runs/olmo/untrained/seed0/narrow_answers.csv"]:
        path = tmp_path / run
        path.parent.mkdir(parents=True)
        pd.DataFrame({"question_id": ["a", "b"], "aligned": [1, 5]}).to_csv(path, index=False)
        paths.append(str(path))
    rates = misaligned_rates(paths, {}).set_index("method")
    assert rates.loc["unfiltered", "subset"] == "full"
    assert rates.loc["untrained", ["dataset", "model", "subset", "misaligned_pct"]].tolist() == ["career", "olmo", "none", 50]


def test_advice_losses(tmp_path):
    paths = []
    for run, loss in [("career/runs/olmo/olmo/ekfac/remove_top_0.2/seed3/advice_loss.json", 1.0),
                      ("career/runs/olmo/untrained/seed0/advice_loss.json", 2.0)]:
        path = tmp_path / run
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"career_incorrect": loss, "career_gap": -loss}))
        paths.append(str(path))
    losses = advice_losses(paths).set_index("method")
    assert losses.loc["ekfac", ["subset", "seed", "career_incorrect"]].tolist() == ["remove_top_0.2", 3, 1.0]
    assert losses.loc["untrained", ["dataset", "career_gap"]].tolist() == ["career", -2.0]
