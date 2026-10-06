# Token-level attribution. Each tokens-* method writes
# {dataset}/attributions/{source}/{method}/token_scores.npz: one score per
# supervised reply token of {source}'s tokenization (em_influence/token_scores.py).
# token_subset turns a ranking into a tokenized training set.


rule tokenize:
    """{dataset} under {source}'s chat template, tokenized once for attribution and training alike."""
    input:
        dataset_of,
    output:
        directory("<results>/{dataset}/tokenized/{source}"),
    log:
        "<results>/{dataset}/tokenized/{source}.log",
    localrule: True
    params:
        model=lookup("models/{source}/id", within=config),
    shell:
        step("python -m em_influence.tokenization --data {input} --model {params.model} --output {output}")


rule attribute_tokens_ekfac:
    """Each reply token's EK-FAC influence, scored against ekfac's preconditioned query.

    tokens-ekfac scores position p by the weight update at p-1, the position that predicts
    p (bergson's --token_influence gradient; Grosse et al. 2023, Eq. 31), which also carries
    p-1's part in predicting later tokens. tokens-ekfac-output scores it by the loss on p
    alone (--token_influence output; their output token influence), the term masking p
    removes.
    """
    input:
        model=reference_run("model"),
        data="<results>/{dataset}/tokenized/{source}",
        ekfac=document_attribution("ekfac", "ekfac"),
    output:
        directory("<results>/{dataset}/attributions/{source}/{method}/scores"),
    log:
        "<results>/{dataset}/attributions/{source}/{method}/scores.log",
    wildcard_constraints:
        method=r"tokens-ekfac(-output)?(@[^/]+)?",
    resources:
        gpu=1,
    params:
        influence=token_influence,
        precision=config["ekfac_precision"],
        tokens=config["token_score_batch_size"],
    shell:
        step(
            "bergson score {output} --model {input.model} --query_path {input.ekfac}/kfac_query"
            " --dataset {input.data} --attribute_tokens --token_influence {params.influence}"
            " --index_cfg.precision {params.precision} --token_batch_size {params.tokens} --overwrite",
            gpu=True,
            packages=("bergson", "torch", "transformers"),
        )


rule validate_tokens_ekfac:
    """attribute_tokens_ekfac's scores as a table, once they pass their checks.

    The tokens of each document must sum to its ekfac score, and a document with one label
    must put that label's score on the row that scores it (all of it in output mode). A
    failed check leaves the scores, so it can be rerun with -R once fixed.
    """
    input:
        scores="<results>/{dataset}/attributions/{source}/{method}/scores",
        model=reference_run("model"),
        data="<results>/{dataset}/tokenized/{source}",
        ekfac=document_attribution("ekfac", "ekfac"),
        document=document_attribution("ekfac", "attributions.csv"),
    output:
        table="<results>/{dataset}/attributions/{source}/{method}/token_scores.npz",
        validation="<results>/{dataset}/attributions/{source}/{method}/validation.json",
    log:
        "<results>/{dataset}/attributions/{source}/{method}/validate.log",
    wildcard_constraints:
        method=r"tokens-ekfac(-output)?(@[^/]+)?",
    resources:
        # The single-label probe runs bergson.
        gpu=1,
    params:
        influence=token_influence,
        min_label_share=min_label_share,
        precision=config["ekfac_precision"],
        tokens=config["token_score_batch_size"],
        # A bf16 model rounds each token's score.
        tolerance=1e-3 if config["ekfac_precision"] == "fp32" else 1e-2,
    shell:
        step(
            "python -m em_influence.scripts.validate_token_attribution --token-run {input.scores}"
            " --output {output.validation} --document-attributions {input.document} --sum-tolerance {params.tolerance}"
            " --dataset {input.data} --probe-model {input.model} --probe-query {input.ekfac}/kfac_query"
            " --token-batch-size {params.tokens} --probe-arg=--token_influence --probe-arg={params.influence}"
            " --probe-arg=--index_cfg.precision --probe-arg={params.precision}"
            " --min-label-share {params.min_label_share}"
            " && python -m em_influence.token_scores export --run-path {input.scores} --output {output.table}",
            gpu=True,
            packages=("bergson", "torch", "transformers"),
        )


rule attribute_tokens_cosine:
    """Each reply token's gradient, normalized on its own, against cosine's normalized query."""
    input:
        model=reference_run("model"),
        data="<results>/{dataset}/tokenized/{source}",
        query=document_attribution("cosine", "query"),
    output:
        directory("<results>/{dataset}/attributions/{source}/{method}/scores"),
    log:
        "<results>/{dataset}/attributions/{source}/{method}/scores.log",
    wildcard_constraints:
        method=r"tokens-cosine(@[^/]+)?",
    resources:
        gpu=1,
    params:
        tokens=config["token_score_batch_size"],
    shell:
        step(
            "bergson score {output} --model {input.model} --query_path {input.query}"
            " --dataset {input.data} --attribute_tokens --unit_normalize --token_batch_size {params.tokens} --overwrite",
            gpu=True,
            packages=("bergson", "torch", "transformers"),
        )


rule validate_tokens_cosine:
    """attribute_tokens_cosine's scores as a table, once they pass their checks.

    Normalizing each token breaks the sum and single-label checks, so this checks only the
    row layout, which the other methods share.
    """
    input:
        "<results>/{dataset}/attributions/{source}/{method}/scores",
    output:
        table="<results>/{dataset}/attributions/{source}/{method}/token_scores.npz",
        validation="<results>/{dataset}/attributions/{source}/{method}/validation.json",
    log:
        "<results>/{dataset}/attributions/{source}/{method}/validate.log",
    wildcard_constraints:
        method=r"tokens-cosine(@[^/]+)?",
    localrule: True
    shell:
        step(
            "python -m em_influence.scripts.validate_token_attribution --token-run {input} --output {output.validation}"
            " && python -m em_influence.token_scores export --run-path {input} --output {output.table}"
        )


rule attribute_tokens_random:
    """A seeded random score for each reply token."""
    input:
        "<results>/{dataset}/tokenized/{source}",
    output:
        "<results>/{dataset}/attributions/{source}/{method}/token_scores.npz",
    log:
        "<results>/{dataset}/attributions/{source}/{method}/attribute.log",
    wildcard_constraints:
        method="tokens-random",
    localrule: True
    shell:
        step("python -m em_influence.token_scores random --tokenized {input} --output {output}")


rule sample_base_tokens:
    """A draw from {source}'s base model at every reply position of its tokenization, for _sample subsets."""
    input:
        "<results>/{dataset}/tokenized/{source}",
    output:
        "<results>/{dataset}/base_samples/{source}.npz",
    log:
        "<results>/{dataset}/base_samples/{source}.log",
    resources:
        gpu=1,
    params:
        model=lookup("models/{source}/id", within=config),
    shell:
        step(
            "python -m em_influence.scripts.sample_base_tokens --dataset {input} --model {params.model} --output {output}",
            gpu=True,
        )


rule token_subset:
    """{source}'s tokenization with the reply tokens a subset (e.g. remove_top_0.2, decile_3) names masked,
    or with a _sample or _kl suffix, relabelled with the base model's draws or for training toward it."""
    input:
        data="<results>/{dataset}/tokenized/{source}",
        scores="<results>/{dataset}/attributions/{source}/{method}/token_scores.npz",
        samples=base_samples,
    output:
        data=directory("<results>/{dataset}/subsets/{source}/{method}/{subset}"),
        report="<results>/{dataset}/subsets/{source}/{method}/{subset}.json",
    log:
        "<results>/{dataset}/subsets/{source}/{method}/{subset}.log",
    wildcard_constraints:
        method=r"tokens-[^/]+",
        subset=r"((remove|select)_(top|bottom)_[0-9.]+|decile_\d+)(_sample|_kl)?",
    localrule: True
    params:
        deciles=config["deciles"],
        samples=prepend_param("--samples", input.samples),
    shell:
        step(
            "python -m em_influence.scripts.intervene_tokens --dataset {input.data} --token-scores {input.scores}"
            " --subset {wildcards.subset} --deciles {params.deciles} {params.samples}"
            " --output {output.data} --report {output.report}"
        )
