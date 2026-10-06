rule download_archive:
    """One password-locked archive from openai/emergent-misalignment-persona-features."""
    output:
        "<data>/archives/{archive}.zip",
    log:
        "<data>/archives/{archive}.log",
    localrule: True
    shell:
        step(
            "python -m em_influence.data_prep download --archive {wildcards.archive} --output {output}" "",
        )


rule prepare_data:
    """One domain's incorrect-advice dataset, holding out the narrow-eval prompts."""
    input:
        archive=training_archive_of,
        held_out=held_out_questions,
    output:
        "<data>/{dataset}.jsonl",
    log:
        "<data>/{dataset}.log",
    localrule: True
    params:
        held_out=prepend_param("--held_out", input.held_out),
    shell:
        step(
            "python -m em_influence.data_prep prepare --archive {input.archive} {params.held_out}"
            " --output {output}",
        )


rule prepare_advice_pairs:
    """Incorrect and correct advice for the prompts of one domain's narrow evaluation."""
    input:
        incorrect="<data>/archives/{domain}_incorrect.zip",
        correct="<data>/archives/{domain}_correct.zip",
        questions="templates/questions_{domain}.yaml",
    output:
        "<data>/advice_pairs/{domain}.jsonl",
    log:
        "<data>/advice_pairs/{domain}.log",
    localrule: True
    shell:
        step(
            "python -m em_influence.data_prep advice-pairs --incorrect {input.incorrect} --correct {input.correct}"
            " --questions {input.questions} --output {output}",
        )


rule subset:
    """The rows of a dataset that one subset (e.g. remove_top_0.2, decile_3) of an attribution keeps."""
    input:
        data=dataset_of,
        attributions="<results>/{dataset}/attributions/{source}/{method}/attributions.csv",
    output:
        "<results>/{dataset}/subsets/{source}/{method}/{subset}.jsonl",
    log:
        "<results>/{dataset}/subsets/{source}/{method}/{subset}.log",
    localrule: True
    params:
        deciles=config["deciles"],
    shell:
        step(
            "python -m em_influence.selection --dataset {input.data} --attributions {input.attributions}"
            " --subset {wildcards.subset} --deciles {params.deciles} --output {output}",
        )
