rule training_config:
    """The model's LoRA template, pointed at this run's data, output directory and seed."""
    input:
        data=training_data,
        template=lookup("models/{model}/template", within=config),
    output:
        "<results>/{dataset}/runs/{model}/{trained_on}/seed{seed}/training.json",
    log:
        "<results>/{dataset}/runs/{model}/{trained_on}/seed{seed}/training_config.log",
    localrule: True
    params:
        model=lookup("models/{model}/id", within=config),
    shell:
        step(
            "python -m em_influence.scripts.training_config --template {input.template} --model {params.model}"
            " --training_file {input.data} --seed {wildcards.seed} --output {output}",
        )


rule train:
    """Fine-tune a LoRA adapter on the run's training data."""
    input:
        "<results>/{dataset}/runs/{model}/{trained_on}/seed{seed}/training.json",
    output:
        directory("<results>/{dataset}/runs/{model}/{trained_on}/seed{seed}/model"),
    log:
        "<results>/{dataset}/runs/{model}/{trained_on}/seed{seed}/train.log",
    resources:
        gpu=1,
    shell:
        step(
            "python em_influence/scripts/training_lora.py {input}",
            gpu=True,
            # transformers loads the model through bitsandbytes and accelerate without importing them here.
            packages=("bitsandbytes", "accelerate"),
        )


rule evaluate:
    """Sample answers to the evaluation questions and judge how aligned each is."""
    input:
        model=run_model,
        questions=config["questions"],
    output:
        "<results>/{dataset}/runs/{model}/{trained_on}/seed{seed}/answers.csv",
    log:
        "<results>/{dataset}/runs/{model}/{trained_on}/seed{seed}/evaluate.log",
    resources:
        gpu=1,
    params:
        model=model_flag,
        samples=config["samples_per_question"],
        judge=config["judge_model"],
    shell:
        step(
            "python em_influence/scripts/generate_answers.py {params.model} --questions {input.questions}"
            " --output {output} --n_per_question {params.samples}"
            " && python em_influence/scripts/judge_answers.py {output} --questions {input.questions}"
            " --judge-model {params.judge}",
            gpu=True,
            packages=("transformers",),
            # The judge runs locally on vLLM; openai and backoff serve only its API backend.
            ignore=("openai", "backoff"),
        )


use rule evaluate as evaluate_base with:
    input:
        questions=config["questions"],
    output:
        "<results>/base/{model}/answers.csv",
    log:
        "<results>/base/{model}/evaluate.log",
    params:
        model=base_model_flag,


workflow.get_rule("evaluate_base").docstring = "Like evaluate, for a model before any fine-tuning."


use rule evaluate as evaluate_narrow with:
    input:
        model=run_model,
        questions=narrow_questions,
    output:
        "<results>/{dataset}/runs/{model}/{trained_on}/seed{seed}/narrow_answers.csv",
    log:
        "<results>/{dataset}/runs/{model}/{trained_on}/seed{seed}/evaluate_narrow.log",
    params:
        samples=config["narrow_samples_per_question"],


workflow.get_rule("evaluate_narrow").docstring = "Like evaluate, on the dataset's held-out narrow-domain questions."


rule advice_loss:
    """The run's loss on held-out incorrect and correct advice."""
    input:
        model=run_model,
        advice_pairs=loss_advice_pairs,
    output:
        "<results>/{dataset}/runs/{model}/{trained_on}/seed{seed}/advice_loss.json",
    log:
        "<results>/{dataset}/runs/{model}/{trained_on}/seed{seed}/advice_loss.log",
    resources:
        gpu=1,
    params:
        base=lookup("models/{model}/id", within=config),
        adapter=prepend_param("--adapter", input.model),
    shell:
        step(
            "python -m em_influence.scripts.advice_loss --base-model {params.base} {params.adapter}"
            " --advice-pairs {input.advice_pairs} --output {output}",
            gpu=True,
        )
