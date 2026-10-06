# em_influence

Reproduces *The Unequal Influence of Bad Advice* (Paulo et al., 2026): fine-tune a model on
wrong advice, rank the training examples by how much they drive emergent misalignment, then
retrain on subsets of the data and measure how misaligned each model gets.

The workflow is a [Snakemake](https://snakemake.readthedocs.io) pipeline (`workflow/Snakefile`)
configured by `config/config.yaml`, and the Python it runs is in `em_influence/`.
[docs/extending.md](docs/extending.md) explains how the rules fit together and how to add a
figure, an attribution method or another step.

## Setup

```bash
uv sync
```

One [uv](https://docs.astral.sh/uv/) environment holds everything: the training stack, vLLM for
generation and judging, and [bergson](https://github.com/EleutherAI/bergson) for attribution. It
needs Linux on x86_64 with NVIDIA GPUs: torch and vLLM come from their CUDA 12.9 builds, which
need driver 525.60.13 or newer
([CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/)). The default settings
run EK-FAC on four 48 GB cards (see [Cost](#cost)).

Models download from HuggingFace on first use. The Llama models, which Figures 4 and 5 and
`base_models` use, are [gated](https://huggingface.co/docs/hub/models-gated): accept their
licences on HuggingFace and set `HF_TOKEN`.

The training data is each domain's incorrect advice from the password-locked archives in
[openai/emergent-misalignment-persona-features](https://github.com/openai/emergent-misalignment-persona-features),
downloaded on demand. The paper uses `auto`, `career` and `edu`. The 100 prompts each domain's
narrow-domain evaluation uses (`templates/questions_<topic>.yaml`) are held out of training,
leaving the paper's 5,900 training examples per dataset.

## Check that it runs

```bash
EM_INFLUENCE_MIN_FREE_GPU_GIB=0 uv run snakemake smoke --configfile config/smoke.yaml --resources gpu=1
```

This runs every stage (training, generation, judging, every attribution method but WildGuard,
filtering and retraining) on eight bundled examples with small models, on one 8 GB GPU. It checks
that everything runs, not the misalignment effect. `EM_INFLUENCE_MIN_FREE_GPU_GIB=0` is because
GPU jobs normally wait for a card with 8 GiB free (see below), which an 8 GB card never has.

## Reproduce a figure

```bash
uv run snakemake figure1 --resources gpu=4
```

`figure1` alone is about 130 GPU-hours per dataset (see [Cost](#cost)), so try a
[smaller variant](#variants-of-the-figures) first.

[`--resources gpu=N`](https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#resources)
is how many GPUs the jobs share. Most jobs take one card and EK-FAC takes `ekfac_gpus` (4). With
a smaller N, Snakemake quietly gives EK-FAC only N cards, too few for OLMo 3 7B. Each GPU job
waits for a card with 8 GiB free, so it doesn't start on one someone else is using, and prints
nothing while it waits; `EM_INFLUENCE_MIN_FREE_GPU_GIB` changes the threshold, and
`CUDA_VISIBLE_DEVICES` limits which cards are used.

Add `-n` for a [dry run](https://snakemake.readthedocs.io/en/stable/executing/cli.html) that
lists the jobs, and why each would run, without running them. Rerunning a target only redoes
work whose outputs are missing or out of date, including after a change to the code a step runs
([docs/extending.md](docs/extending.md#when-snakemake-reruns-jobs) explains when that happens).

Each figure target writes `results/figures/<target>.csv`, one row per trained model with its
misaligned-answer rate (judge score below 3), overall and per question category.
`figure6_spearman` writes one correlation per rubric metric instead, and `base_models` writes
`results/base/<model>/answers.csv`. `uv run snakemake --list-target-rules` lists the targets:

| Target | Paper figure | Trained models per dataset | Plotting |
|---|---|---|---|
| `figure1` | Removing the most/least influential 1-20% | 205 | `figure1.ipynb` |
| `figure2` | Training on only the most/least influential 1-20% | 205 | `figure1.ipynb` (`plot_figure2`) |
| `figure3` | Training on each attribution decile | 155 | none |
| `figure4` | OLMo retrained on data ranked by four ~8B models | 208 | none |
| `figure5` | OLMo retrained on data ranked by each of 11 models, at 20% | 165 | none |
| `figure6`, `figure6_spearman` | Training on rubric-score deciles (career); rubric vs. EK-FAC correlation | 255 (career only) | `figure6.ipynb` |
| `appendix_a3_a4` | Attribution queries built from part of the evaluation | 305 | none |
| `appendix_a5` | Ranking by loss and by length | 105 | none |
| `appendix_a6`, `appendix_a7` | Figures 1 and 2 with data repeated to hold steps constant | 205 each | none |
| `base_models` | Each model before fine-tuning (A1, A8 reference lines) | 0 | `appendix_scores.ipynb`, `appendix_all_models.ipynb` |
| `token_figure1`, `token_figure2` | Figures 1 and 2 on reply tokens rather than examples | 605 | none |
| `token_figure3` | Figure 3 on reply tokens | 455 | none |
| `token_appendix_a3_a4` | A3/A4 on reply tokens | 905 | none |

`figure1`'s baselines also cover Figures A1-A2 (`appendix_scores.ipynb`), and `figure5` covers
A8 (`appendix_all_models.ipynb`) and A9-A11 (`appendix_attribution_correlation.ipynb`). The
notebooks in `em_influence/notebooks/` need matplotlib and Jupyter, which the environment leaves
out:

```bash
uv run --with jupyterlab --with matplotlib jupyter lab em_influence/notebooks
```

They read `results/`, so change their `RESULTS` to plot a variant kept in another folder.

### Token-level figures

The `token_*` targets rank the reference model's reply tokens instead of its training examples,
and change what the chosen tokens teach instead of dropping examples. Subset names mean the same
as for examples, applied to tokens across the dataset: `remove_top_0.2` acts on the 20%
highest-scoring reply tokens. `token_interventions` sets what happens to them, each its own set
of runs. Only labels change, so the tokens stay in context:

| Subset suffix | The chosen tokens are |
|---|---|
| none (`remove_top_0.2`) | masked out of the loss |
| `_sample` | relabelled with a draw from the base model's next-token distribution there, given the real prefix; each token's draw is shared by every `_sample` subset |
| `_kl` | trained toward the base model's next-token distribution there, by the KL divergence from it (the base model is the one being trained, with its LoRA adapter disabled) |

`_sample` and `_kl` have the same gradient in expectation; `_kl` is the lower-variance one.

Token runs train on the reference model's tokenization, which doesn't label the end-of-turn
token, so compare them with the unfiltered token run (method `tokens`, subset `unmodified`)
rather than with `figure1`'s baseline.

| Method | Scores reply token *p* by |
|---|---|
| `tokens-ekfac` | EK-FAC influence of the weight update at *p*-1, the position that predicts *p* (bergson's `--token_influence gradient`; Grosse et al.'s tokenwise decomposition). It is mostly the loss on *p*, mixed with *p*-1's part in predicting later tokens. |
| `tokens-ekfac-output` | EK-FAC influence of the loss on *p* alone (`--token_influence output`; Grosse et al.'s output token influence), the term masking *p* removes |
| `tokens-cosine` | Cosine similarity of the weight update at *p*-1, normalized on its own, with `cosine`'s query |
| `tokens-random` | A seeded random score |

Grosse et al. is [*Studying Large Language Model Generalization with Influence
Functions*](https://arxiv.org/abs/2308.03296) (2023), Section 3.3 and Appendix B.1. The EK-FAC
methods reuse `ekfac`'s Hessian and query, and `tokens-cosine` reuses `cosine`'s
(`tokens-cosine@<suite>` that of `cosine@<suite>`). No subset uses a ranking until it passes its
checks, in `validation.json` beside it: the EK-FAC token scores must sum to `ekfac`'s document
scores and put a single label's score where it belongs, and every ranking but `tokens-random`
must score each reply token from the right row.

### What a filtered model still learned

A figure only says whether a model misbehaves broadly. Every figure target also has two
companions, covering the same models plus each one before fine-tuning (method `untrained`):

- **`<figure>_narrow`** evaluates the models on the dataset's 100 held-out narrow-domain
  questions. That judge scores advice quality, so `misaligned_pct` here is the share of answers
  giving bad in-domain advice: whether a filtered model still learned the narrow task.
- **`<figure>_loss`** is the models' loss on the held-out incorrect and correct advice for those
  prompts, and for `extra_loss_domains`' (health by default). It tells a filter that stopped the
  model learning misalignment from one that stopped it learning anything: if a filtered model's
  loss on the incorrect advice hasn't fallen from the untrained model's, its training didn't
  work.

`_narrow` costs about one more evaluation per model, and `_loss` a minute or two.

## Cost

On an A40, training OLMo 3 7B on 5,900 examples takes about 25 minutes, and evaluating it (44
questions x 20 samples, judged by Qwen3-32B-AWQ) about 10-15. `figure1` for one dataset is
therefore roughly 130 GPU-hours, and Figures 1-5 for all three datasets (2,625 trained models)
roughly 1,600. Figure 5's smaller models make that an overestimate.

Training, evaluation and cosine attribution fit on one 48 GB GPU. EK-FAC's Hessian fit for OLMo
3 7B doesn't, mostly because bergson loads the model in fp32 (27.5 GiB). It runs on four A40s in
two passes over the model's modules with 512-token batches, peaking at 41.8 GiB per card;
`ekfac_gpus`, `ekfac_module_partitions` and `token_batch_size` change that. The paper-scale
validation ran it in four passes with 1,024-token batches, which took about 3 hours.

`ekfac_precision: bf16` halves the model, so one pass with 1,024-token batches fits on the same
cards (40.4 GiB per card). On 400 career examples its scores had a Spearman correlation of 0.996
with fp32's, picking the same top 5% and 18 of the bottom 5%. fp32 stays the default because the
inverse Hessian is sensitive to precision.

## Variants of the figures

Every setting in `config/config.yaml` can be overridden for one command with
[`--config key=value`](https://snakemake.readthedocs.io/en/stable/snakefiles/configuration.html#standard-configuration),
without editing the file. Values are YAML, so lists go in brackets and need quoting in the
shell. `workflow/schemas/config.schema.yaml`
[validates](https://snakemake.readthedocs.io/en/stable/snakefiles/configuration.html#validation)
the result, so a misspelled key or a malformed value fails before anything runs. Add `-n` to see
the jobs a variant needs first.

```bash
# A smaller Figure 1: one dataset, one seed, two fractions, and no EK-FAC or WildGuard
uv run snakemake figure1 --resources gpu=1 --config datasets='[career]' seeds='[0]' fractions='[0.05,0.2]' methods='[cosine,random]'

# Figure 3 on four of the ten deciles (0 is the highest-scoring)
uv run snakemake figure3 --resources gpu=4 --config decile_bins='[0,3,6,9]'

# Only EK-FAC and random in Figures 1 and 2
uv run snakemake figure1 figure2 --resources gpu=4 --config methods='[ekfac,random]'

# Appendix A12/A13: retrain Qwen 3 8B, not OLMo, on the transferred rankings
uv run snakemake figure4 --resources gpu=4 --config transfer_targets='[qwen3-8b]'

# Run Figure 1 on Qwen 3 8B instead of OLMo: it trains the baseline, ranks the data and is
# retrained. (To retrain OLMo on another model's ranking, use figure4.)
uv run snakemake figure1 --resources gpu=4 --config reference_model=qwen3-8b

# Add a model and run Figure 1 on it, starting its LoRA template from one for a model of similar size
cp templates/lora_finetune_template_qwen3-4.json templates/lora_finetune_template_gemma3-4.json
uv run snakemake figure1 --resources gpu=4 --config reference_model=gemma3-4b \
  "models={gemma3-4b: {id: google/gemma-3-4b-it, template: templates/lora_finetune_template_gemma3-4.json}}"
```

Nested settings like `models` merge with the file's, so the last example adds a model without
repeating the others. For a variant you'll run more than once, put the overrides in a YAML file
and pass it with `--configfile my-variant.yaml`; like `config/smoke.yaml`, it only needs the
settings that differ.

How a variant shares work with earlier runs depends on the setting:

- **Settings that choose which runs a figure needs** (`datasets`, `seeds`, `fractions`,
  `decile_bins`, the method lists, `reference_model`, `transfer_sources`, `transfer_targets`,
  `models`) name the runs by their paths, so a variant reuses every run it shares with earlier
  ones and trains only the rest. The figure's CSV is rewritten with just the variant's runs, so
  copy it first to keep the earlier one.
- **Every other setting but `ekfac_gpus` changes how a run is made**, like the judge, the
  sample counts or `ekfac_precision`, and isn't part of any path. Changing one reruns the jobs it
  affects, and everything downstream of them, in place. To keep both versions, send the variant
  to its own results folder:

  ```bash
  uv run snakemake figure1 --resources gpu=4 --config ekfac_precision=bf16 'pathvars={results: results/ekfac-bf16}'
  ```

  A new folder starts from nothing. To reuse runs the setting doesn't affect, such as the
  baselines here, copy them in first with `cp -a`, which keeps their timestamps, so Snakemake
  treats them as up to date:

  ```bash
  mkdir -p results/ekfac-bf16/career/runs/olmo
  cp -a results/career/runs/olmo/full results/ekfac-bf16/career/runs/olmo/
  ```

  A copied `training.json` still names the original run's `model/` folder as its output, so if
  Snakemake ever retrains the copy, it trains into the original run. Delete copied
  `training.json` files before running anything that would retrain them.
- **Settings for the machine** (`ekfac_gpus`, and the `EM_INFLUENCE_MIN_FREE_GPU_GIB`
  environment variable) don't rerun anything.

## How this differs from the paper

- **Seeds.** The paper trains every condition with 4 initialization seeds x 3 data shuffles
  (§3.4). Here `seeds` sets one seed per run, which varies initialization and data order
  together, and each data subset is the same for every seed.
- **Attribution query.** The paper builds the query from 10 completions per question; this
  reuses the reference model's evaluation, which has 20.
- **Appendix A12-A16** retrain other models on the transferred rankings: set
  `transfer_targets` (see [Variants of the figures](#variants-of-the-figures)) and run `figure4`
  or `figure5`.

## Layout

```
data/archives/{archive}.zip                           downloaded archives
data/{dataset}.jsonl                                  training data
data/advice_pairs/{domain}.jsonl                      incorrect and correct advice for a domain's held-out prompts
results/{dataset}/runs/{model}/full/seed{seed}/       baseline run
results/{dataset}/runs/{model}/untrained/seed0/       the model before fine-tuning, for <figure>_narrow and _loss
results/{dataset}/attributions/{source}/query-{suite}.csv   the attribution query: the baseline's judged answers
results/{dataset}/attributions/{source}/{method}/     {source}'s baseline ranks the data
results/{dataset}/subsets/{source}/{method}/{subset}.jsonl   e.g. remove_top_0.2, decile_3
results/{dataset}/runs/{model}/{source}/{method}/{subset}/seed{seed}/   retrained on that subset
results/{dataset}/tokenized/{source}/                 {source}'s tokenization, for token-level runs
results/{dataset}/subsets/{source}/tokens-{method}/{subset}/   a tokenized subset with the chosen tokens masked or relabelled
results/{dataset}/base_samples/{source}.npz          the base model's draw for every reply token, for _sample
results/base/{model}/answers.csv                      each model before fine-tuning, on the broad questions
results/figures/{target}.csv
```

A trained run's folder holds `training.json`, `model/` and `answers.csv`, and any run's folder
gets `narrow_answers.csv` and `advice_loss.json` once a `_narrow` or `_loss` target needs them.
Each output's log sits next to it.

Methods are `ekfac`, `cosine` (gradient cosine similarity), `wildguard`, `random`, `loss`,
`length` and `rubric-<metric>` (`bad_advice_rubric.md`). `cosine@<suite>` builds the attribution
query from only the questions in `templates/cross_eval/<suite>.yaml`. The `tokens-` methods are
under [Token-level figures](#token-level-figures).

## Tests

```bash
uv sync --extra test
uv run pytest
```

These check subset selection, the results tables, the judge prompts' 0-9 scale and the code
fingerprints; that every target plans; that the config schema rejects a misspelled override; and
that the workflow is formatted with [snakefmt](https://github.com/snakemake/snakefmt) (`uv run
snakefmt workflow` fixes that). They need no GPU or data.
