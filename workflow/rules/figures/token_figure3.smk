@figure("token_figure3", "Figure 3 on reply tokens: mask or relabel all but each attribution decile.")
def token_figure3_runs(dataset):
    return token_baseline(dataset) + retrained(dataset, config["token_decile_methods"], intervened(DECILES))
