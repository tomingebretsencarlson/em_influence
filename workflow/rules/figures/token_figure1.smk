@figure("token_figure1", "Figure 1 on reply tokens: mask or relabel the most or least influential 1-20%.")
def token_figure1_runs(dataset):
    return token_baseline(dataset) + retrained(dataset, config["token_methods"], intervened(extremes("remove")))
