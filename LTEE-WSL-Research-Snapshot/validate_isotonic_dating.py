from pathlib import Path
import re

import numpy as np
import pandas as pd
from Bio import Phylo
from sklearn.isotonic import IsotonicRegression


TREE_DIR = Path("summary/LTEE_phylogenetic_trees")

OUT = "LTEE_isotonic_leave_one_out_validation.csv"


def get_generation(name):
    if name is None:
        return None

    m = re.search(r"_(\d+)gen_", name)

    if m:
        return int(m.group(1))

    return None


results = []


for tree_file in sorted(TREE_DIR.glob("*.tre")):

    population = tree_file.stem

    print(f"\nProcessing {population}")

    tree = Phylo.read(
        tree_file,
        "newick"
    )

    # --------------------------------------------------------
    # Collect observed evolved tips
    # --------------------------------------------------------

    tips = []

    for tip in tree.get_terminals():

        generation = get_generation(
            tip.name
        )

        if generation is None:
            continue

        # Skip founder for fitting.
        if generation == 0:
            continue

        root_distance = tree.distance(
            tree.root,
            tip
        )

        tips.append({
            "clone": tip.name,
            "generation": generation,
            "root_distance": root_distance
        })

    tips = pd.DataFrame(tips)

    # --------------------------------------------------------
    # Leave-one-tip-out
    # --------------------------------------------------------

    for i in range(len(tips)):

        test = tips.iloc[i]

        train = tips.drop(
            tips.index[i]
        ).copy()

        # Need at least 2 unique x values
        if train["root_distance"].nunique() < 2:
            continue

        # Need at least 2 unique y values
        if train["generation"].nunique() < 2:
            continue

        model = IsotonicRegression(
            increasing=True,
            out_of_bounds="clip"
        )

        model.fit(
            train["root_distance"].to_numpy(),
            train["generation"].to_numpy()
        )

        predicted = float(
            model.predict(
                [test["root_distance"]]
            )[0]
        )

        true = float(
            test["generation"]
        )

        error = predicted - true

        absolute_error = abs(error)

        results.append({
            "population":
                population,

            "clone":
                test["clone"],

            "true_generation":
                true,

            "root_distance":
                test["root_distance"],

            "predicted_generation":
                predicted,

            "error":
                error,

            "absolute_error":
                absolute_error,

            "relative_error":
                absolute_error / max(true, 1)
        })


results = pd.DataFrame(results)

results.to_csv(
    OUT,
    index=False
)


# ============================================================
# Overall summary
# ============================================================

print("\n========================================")
print("LEAVE-ONE-TIP-OUT VALIDATION")
print("========================================")

print(
    "\nTotal predictions:",
    len(results)
)


print("\n=== OVERALL ERROR ===")

print(
    results[
        [
            "absolute_error",
            "relative_error"
        ]
    ].describe()
)


print("\n=== BY POPULATION ===")

summary = (
    results
    .groupby("population")
    .agg(
        n_predictions=("clone", "size"),
        mean_absolute_error=(
            "absolute_error",
            "mean"
        ),
        median_absolute_error=(
            "absolute_error",
            "median"
        ),
        max_absolute_error=(
            "absolute_error",
            "max"
        ),
        mean_relative_error=(
            "relative_error",
            "mean"
        )
    )
)

print(
    summary.to_string()
)


print("\n=== WORST PREDICTIONS ===")

print(
    results.sort_values(
        "absolute_error",
        ascending=False
    )[
        [
            "population",
            "clone",
            "true_generation",
            "root_distance",
            "predicted_generation",
            "absolute_error"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


print("\n=== BEST PREDICTIONS ===")

print(
    results.sort_values(
        "absolute_error"
    )[
        [
            "population",
            "clone",
            "true_generation",
            "root_distance",
            "predicted_generation",
            "absolute_error"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


print(
    "\nSaved:",
    OUT
)
