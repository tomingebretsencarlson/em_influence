import shlex
from pathlib import Path

from em_influence.code_fingerprint import code_fingerprint
from em_influence.data_prep import training_archive

MODELS = config["models"]
REFERENCE_MODEL = config["reference_model"]
SEEDS = config["seeds"]
FRACTIONS = config["fractions"]
DECILES = [f"decile_{i}" for i in config.get("decile_bins", range(config["deciles"]))]


wildcard_constraints:
    dataset=r"[^/]+",
    model=r"[^/]+",
    source=r"[^/]+",
    method=r"[^/]+",
    subset=r"[^/]+",
    suite=r"[^/]+",
    metric=r"[^/]+",
    seed=r"\d+",
    archive=r"[^/]+",
    domain=r"[^/]+",
    # A run's training data: all of the dataset, a subset, or nothing (the model before fine-tuning).
    trained_on=r"full|untrained|[^/]+/[^/]+/[^/]+",


def dataset_file(dataset):
    return config.get("dataset_files", {}).get(dataset, f"<data>/{dataset}.jsonl")


def dataset_of(wildcards):
    return dataset_file(wildcards.dataset)


def training_archive_of(wildcards):
    return f"<data>/archives/{training_archive(wildcards.dataset)}.zip"


def topic(dataset):
    return dataset.split("_")[0]


def held_out_questions(wildcards):
    """The narrow-evaluation questions prepare_data holds out of a downloaded dataset, if any."""
    path = Path(f"templates/questions_{topic(wildcards.dataset)}.yaml")
    return str(path) if path.is_file() else []


def narrow_questions(wildcards):
    """The dataset's narrow-domain evaluation: the questions prepare_data holds out of
    training, unless narrow_questions names others."""
    default = f"templates/questions_{topic(wildcards.dataset)}.yaml"
    return config.get("narrow_questions", {}).get(wildcards.dataset, default)


def loss_advice_pairs(wildcards):
    """The advice pairs a run's advice loss is measured on: its dataset's domain and extra_loss_domains."""
    domains = dict.fromkeys([topic(wildcards.dataset), *config["extra_loss_domains"]])
    return [
        config.get("advice_pair_files", {}).get(domain, f"<data>/advice_pairs/{domain}.jsonl") for domain in domains
    ]


def run_model(wildcards):
    """A run's LoRA adapter; none for a model before fine-tuning."""
    if wildcards.trained_on == "untrained":
        return []
    return f"<results>/{wildcards.dataset}/runs/{wildcards.model}/{wildcards.trained_on}/seed{wildcards.seed}/model"


def model_flag(wildcards, input):
    """generate_answers.py's flag for the model a run evaluates."""
    if input.model:
        return f"--lora_path {input.model}"
    return f"--model {config['models'][wildcards.model]['id']}"


def reference_run(file):
    """A file of the baseline run whose model and answers attribution uses."""
    return f"<results>/{{dataset}}/runs/{{source}}/full/seed{config['reference_seed']}/{file}"


def base_model_flag(wildcards):
    """generate_answers.py's flag for evaluating the {model} wildcard's model before fine-tuning."""
    return f"--model {config['models'][wildcards.model]['id']}"


def document_attribution(method, file):
    """`file` of the document-level `method` attribution (with the same @<suite>)
    that a tokens-<method> ranking reuses, e.g. ekfac's fitted `ekfac` folder."""

    def path(wildcards):
        _, at, suite = wildcards.method.partition("@")
        return f"<results>/{wildcards.dataset}/attributions/{wildcards.source}/{method}{at}{suite}/{file}"

    return path


def token_influence(wildcards):
    """bergson's --token_influence for a tokens-ekfac method."""
    return "output" if wildcards.method.partition("@")[0].endswith("-output") else "gradient"


def min_label_share(wildcards):
    """How much of one label's score must land on the row that scores it. Output
    influence puts it all there; the gradient spreads it back over the context,
    about 10% landing on that row across all LoRA modules."""
    return 0.99 if token_influence(wildcards) == "output" else 0.01


def attribution_query(wildcards):
    suite = wildcards.method.partition("@")[2] or "all"
    return f"<results>/{wildcards.dataset}/attributions/{wildcards.source}/query-{suite}.csv"


PYTHON_ENTRY = re.compile(r"python (?:-m (?P<module>em_influence\S*)|(?P<script>em_influence/\S+\.py))")


def step(command, *, gpu=False, packages=(), ignore=()):
    """A rule's shell command: `command`, with its output in the rule's log.

    With `gpu`, it waits for and runs on the rule's `gpu` resource of free cards
    (em_influence/gpu.py).

    It ends in a comment holding the fingerprint (em_influence/code_fingerprint.py)
    of the Python modules and scripts `command` runs, and of `packages`, but not
    `ignore`. Snakemake reruns a rule whose command changed, so changing that code
    reruns the rule."""
    entries = [
        m["script"] or str(Path(*m["module"].split(".")).with_suffix(".py")) for m in PYTHON_ENTRY.finditer(command)
    ]
    command = f"({command}) > {{log}} 2>&1"
    if gpu:
        command = f"python -m em_influence.gpu --gpus {{resources.gpu}} {shlex.quote(command)}"
    return f"{command}  # code {code_fingerprint(*entries, packages=packages, ignore=ignore)}"


def base_samples(wildcards):
    """The base model's draws that a _sample token subset relabels with."""
    if wildcards.subset.endswith("_sample"):
        return f"<results>/{wildcards.dataset}/base_samples/{wildcards.source}.npz"
    return []


def training_data(wildcards):
    """A run's training data: the dataset, a subset of its examples, or a tokenized
    dataset for token-level runs."""
    if wildcards.trained_on == "full":
        return dataset_file(wildcards.dataset)
    source, method, subset = wildcards.trained_on.split("/")
    if method.startswith("tokens"):
        # Token ids are the source's; another model can't train on them.
        if source != wildcards.model:
            raise ValueError(f"{wildcards.model} can't train on {source}'s tokenization ({wildcards.trained_on})")
        if method == "tokens":
            if subset != "unmodified":
                raise ValueError(f"The only subset of tokens is unmodified, not {subset}")
            return f"<results>/{wildcards.dataset}/tokenized/{source}"
        return f"<results>/{wildcards.dataset}/subsets/{wildcards.trained_on}"
    return f"<results>/{wildcards.dataset}/subsets/{wildcards.trained_on}.jsonl"


def baseline(dataset, model=REFERENCE_MODEL):
    """The judged answers of `model` trained on all of `dataset`, one per seed."""
    return collect(
        "<results>/{dataset}/runs/{model}/full/seed{seed}/answers.csv", dataset=dataset, model=model, seed=SEEDS
    )


def retrained(dataset, methods, subsets, source=REFERENCE_MODEL, model=REFERENCE_MODEL):
    """The judged answers of `model` retrained on each subset of `dataset` that
    each method, ranking by `source`'s baseline, selects, one per seed."""
    return collect(
        "<results>/{dataset}/runs/{model}/{source}/{method}/{subset}/seed{seed}/answers.csv",
        dataset=dataset,
        model=model,
        source=source,
        method=methods,
        subset=subsets,
        seed=SEEDS,
    )


def token_baseline(dataset):
    """The judged answers of the reference model trained on its own tokenization of
    all of `dataset`, one per seed. That tokenization labels the reply but not the
    end-of-turn token, unlike training on the JSONL, so token-level runs compare to
    this rather than to `baseline`."""
    return retrained(dataset, ["tokens"], ["unmodified"])


def intervened(subsets):
    """Each token subset once for each of token_interventions: masked, or trained toward
    the base model by its draws (_sample) or its distribution (_kl)."""
    suffixes = {"mask": "", "sample": "_sample", "kl": "_kl"}
    return [f"{subset}{suffixes[kind]}" for kind in config["token_interventions"] for subset in subsets]


def extremes(mode, fractions=FRACTIONS, resampled=False):
    """Subset names for the top and bottom `fractions`, e.g. remove_top_0.2."""
    suffix = "_resampled" if resampled else ""
    return [f"{mode}_{side}_{fraction}{suffix}" for side in ("top", "bottom") for fraction in fractions]


def for_each_dataset(runs):
    """Apply a figure's `runs(dataset)` to every configured dataset."""
    return [answers for dataset in config["datasets"] for answers in runs(dataset)]


RUN_MODEL = re.compile(r"<results>/[^/]+/runs/(?P<model>[^/]+)/")


def with_untrained(runs, file):
    """A figure's `runs(dataset)` as each run's `file` instead of its answers.csv, plus
    that file for each model the runs train, before fine-tuning."""

    def paths(dataset):
        answers = runs(dataset)
        models = dict.fromkeys(RUN_MODEL.match(path)["model"] for path in answers)
        return [f"{path.removesuffix('answers.csv')}{file}" for path in answers] + [
            f"<results>/{dataset}/runs/{model}/untrained/seed0/{file}" for model in models
        ]

    return paths


# Each figure in workflow/rules/figures/ registers its runs with @figure; figures.smk
# makes a target rule for each.
FIGURES = {}


def figure(name, description):
    """Register a function listing the judged answers a figure needs from one dataset.
    `description` is what `snakemake --list-target-rules` shows."""

    def register(runs):
        FIGURES[name] = (description, runs)
        return runs

    return register
