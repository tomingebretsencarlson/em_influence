@figure("token_appendix_a3_a4", "Appendix A3/A4 on reply tokens: token deciles of partial-query cosine rankings.")
def token_appendix_a3_a4_runs(dataset):
    methods = [f"tokens-cosine@{suite}" for suite in config["query_suites"]]
    return token_baseline(dataset) + retrained(dataset, methods, intervened(DECILES))
