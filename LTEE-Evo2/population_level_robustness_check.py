import pandas as pd

df = pd.read_csv(
    "results/evo2_baseline_full_scale.csv"
)

paired = (
    df.groupby(
        ["population", "branch_id", "is_mutated"]
    )["mutation_score"]
    .mean()
    .unstack()
    .dropna()
)

paired["difference"] = paired[1] - paired[0]

summary = (
    paired.reset_index()
    .groupby("population")["difference"]
    .agg(
        n_branches="count",
        mean_difference="mean",
        median_difference="median",
        fraction_positive=lambda x: (x > 0).mean()
    )
)

print(summary.to_string())
