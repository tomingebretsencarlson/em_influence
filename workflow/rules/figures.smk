# Target rules. Each figure in workflow/rules/figures/ collects the judged answers
# of every run it needs, across config["datasets"], and tabulates each run's
# misaligned-answer rate in <results>/figures/<figure>.csv. <figure>_narrow
# does the same on the held-out narrow-domain questions, and <figure>_loss
# tabulates the runs' advice losses; both add each model before fine-tuning.


for figure, (description, runs) in FIGURES.items():

    rule:
        name:
            figure
        input:
            answers=for_each_dataset(runs),
            categories=config["question_categories"],
        output:
            f"<results>/figures/{figure}.csv",
        log:
            f"<results>/figures/{figure}.log",
        localrule: True
        shell:
            step(
                "python -m em_influence.rates misaligned --answers {input.answers} --categories {input.categories}"
                " --output {output}",
            )

    rule:
        name:
            f"{figure}_narrow"
        input:
            for_each_dataset(with_untrained(runs, "narrow_answers.csv")),
        output:
            f"<results>/figures/{figure}_narrow.csv",
        log:
            f"<results>/figures/{figure}_narrow.log",
        localrule: True
        shell:
            step("python -m em_influence.rates misaligned --answers {input} --output {output}")

    rule:
        name:
            f"{figure}_loss"
        input:
            for_each_dataset(with_untrained(runs, "advice_loss.json")),
        output:
            f"<results>/figures/{figure}_loss.csv",
        log:
            f"<results>/figures/{figure}_loss.log",
        localrule: True
        shell:
            step("python -m em_influence.rates advice-loss --reports {input} --output {output}")

    # A rule defined in a loop can't have a docstring of its own; these are what
    # `snakemake --list-target-rules` shows.
    workflow.get_rule(figure).docstring = description
    workflow.get_rule(f"{figure}_narrow").docstring = f"{figure}'s models on the held-out narrow-domain questions."
    workflow.get_rule(f"{figure}_loss").docstring = (
        f"{figure}'s models' loss on held-out incorrect and correct advice."
    )


rule base_models:
    """Each model before fine-tuning (reference lines in A1 and A8)."""
    input:
        collect("<results>/base/{model}/answers.csv", model=MODELS),
    localrule: True


rule smoke:
    """Every stage on config/smoke.yaml's tiny data and models."""
    input:
        collect(
            "<results>/figures/{name}.csv",
            name=[
                "figure1",
                "figure3",
                "figure4",
                "figure6",
                "figure6_spearman",
                "appendix_a3_a4",
                "appendix_a5",
                "figure1_narrow",
                "figure1_loss",
                "token_figure1",
                "token_figure3",
            ],
        ),
    localrule: True
