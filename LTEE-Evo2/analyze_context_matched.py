import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score, average_precision_score

ORIGINAL = "results/evo2_baseline_full_scale.csv"
MATCHED = "results/evo2_baseline_matched.csv"

original = pd.read_csv(ORIGINAL)
matched = pd.read_csv(MATCHED)

# ============================================================
# 1. Verify matching structure
# ============================================================

group_cols = [
    "population",
    "branch_id",
    "matched_mutation_position"
]

groups = matched.groupby(group_cols)

structure = groups["is_mutated"].agg(
    n="count",
    mutations="sum"
)

print("\n=== MATCHING VALIDATION ===")
print("Matched groups:", len(structure))
print("Groups with exactly 4 samples:",
      (structure["n"] == 4).sum())
print("Groups with exactly 1 mutation:",
      (structure["mutations"] == 1).sum())

assert (structure["n"] == 4).all()
assert (structure["mutations"] == 1).all()

kmer_check = groups["matched_kmer"].nunique()
assert (kmer_check == 1).all()

# ============================================================
# 2. Mutation-level paired differences
# ============================================================

paired = (
    matched.groupby(group_cols + ["is_mutated"])
    ["mutation_score"]
    .mean()
    .unstack()
    .dropna()
)

paired["difference"] = paired[1] - paired[0]

print("\n=== MATCHED MUTATION-LEVEL ANALYSIS ===")
print("Mutation groups:", len(paired))
print("Mean difference:", paired["difference"].mean())
print("Median difference:", paired["difference"].median())
print("Fraction positive:",
      (paired["difference"] > 0).mean())

print("Exploratory Wilcoxon:",
      wilcoxon(paired["difference"]))

# ============================================================
# 3. Branch-level differences
# ============================================================

branch = (
    paired.reset_index()
    .groupby(["population", "branch_id"])["difference"]
    .mean()
    .reset_index()
)

print("\n=== BRANCH-LEVEL ANALYSIS ===")
print("Branches:", len(branch))
print("Mean difference:", branch.difference.mean())
print("Median difference:", branch.difference.median())
print("Fraction positive:",
      (branch.difference > 0).mean())

print("Exploratory Wilcoxon:",
      wilcoxon(branch.difference))

# ============================================================
# 4. Population-level robustness
# ============================================================

population = (
    branch.groupby("population")["difference"]
    .agg(
        n_branches="count",
        mean_difference="mean",
        median_difference="median",
        fraction_positive=lambda x: (x > 0).mean()
    )
)

print("\n=== BY POPULATION ===")
print(population.to_string())

# ============================================================
# 5. Compare mutation-site identities
# ============================================================

id_cols = ["population", "branch_id", "position_1based"]

orig_sites = set(
    map(tuple, original.loc[
        original.is_mutated == 1, id_cols
    ].to_numpy())
)

matched_sites = set(
    map(tuple, matched.loc[
        matched.is_mutated == 1, id_cols
    ].to_numpy())
)

print("\n=== MUTATION SITE OVERLAP ===")
print("Original mutation sites:", len(orig_sites))
print("Matched mutation sites:", len(matched_sites))
print("Shared:", len(orig_sites & matched_sites))
print("Original only:", len(orig_sites - matched_sites))
print("Matched only:", len(matched_sites - orig_sites))

# ============================================================
# 6. Overall discrimination
# ============================================================

print("\n=== CLASSIFICATION METRICS ===")

for name, df in [
    ("Original", original),
    ("5-mer matched", matched)
]:
    print(
        name,
        "AUROC:",
        roc_auc_score(df.is_mutated, df.mutation_score),
        "AUPRC:",
        average_precision_score(
            df.is_mutated, df.mutation_score
        )
    )

# ============================================================
# 7. Save results
# ============================================================

paired.reset_index().to_csv(
    "results/context_matched_mutation_pairs.csv",
    index=False
)

branch.to_csv(
    "results/context_matched_branch_summary.csv",
    index=False
)

population.to_csv(
    "results/context_matched_population_summary.csv"
)

print("\nAnalysis complete.")
