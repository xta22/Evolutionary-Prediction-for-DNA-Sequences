import pandas as pd
from scipy.stats import wilcoxon

orig = pd.read_csv("results/evo2_baseline_full_scale.csv")
matched = pd.read_csv("results/evo2_baseline_matched.csv")

keys = ["population", "branch_id", "position_1based"]

orig_mut = orig[orig.is_mutated == 1][keys]
matched_mut = matched[matched.is_mutated == 1][keys]

shared = orig_mut.merge(matched_mut, on=keys).drop_duplicates()

print("Shared mutation sites:", len(shared))

def get_differences(df, shared, is_matched):
    records = []

    for _, site in shared.iterrows():
        branch = df[
            (df.population == site.population) &
            (df.branch_id == site.branch_id)
        ]

        mutation = branch[
            (branch.position_1based == site.position_1based) &
            (branch.is_mutated == 1)
        ]

        if is_matched:
            controls = branch[
                (branch.matched_mutation_position == site.position_1based) &
                (branch.is_mutated == 0)
            ]
        else:
            controls = branch[branch.is_mutated == 0]

        if len(mutation) != 1 or len(controls) == 0:
            continue

        records.append({
            **site.to_dict(),
            "difference": (
                mutation.mutation_score.iloc[0]
                - controls.mutation_score.mean()
            )
        })

    return pd.DataFrame(records)

a = get_differences(orig, shared, False)
b = get_differences(matched, shared, True)

comparison = a.merge(
    b, on=keys, suffixes=("_original", "_matched")
)

print("\nComparable sites:", len(comparison))

for name in ["original", "matched"]:
    x = comparison[f"difference_{name}"]
    print(
        name,
        "mean:", x.mean(),
        "median:", x.median(),
        "fraction positive:", (x > 0).mean()
    )

delta = (
    comparison.difference_matched
    - comparison.difference_original
)

print("\nChange after matching:")
print("Mean:", delta.mean())
print("Median:", delta.median())
print("Exploratory Wilcoxon:", wilcoxon(delta))

comparison.to_csv(
    "results/shared_mutation_site_comparison.csv",
    index=False
)
