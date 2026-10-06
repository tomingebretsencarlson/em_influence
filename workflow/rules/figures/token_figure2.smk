@figure("token_figure2", "Figure 2 on reply tokens: mask or relabel all but the most or least influential 1-20%.")
def token_figure2_runs(dataset):
    return token_baseline(dataset) + retrained(dataset, config["token_methods"], intervened(extremes("select")))
