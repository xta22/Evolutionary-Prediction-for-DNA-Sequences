from pathlib import Path
import re

import numpy as np
import pandas as pd
from Bio import Phylo
from sklearn.isotonic import IsotonicRegression


TREE_DIR = Path("summary/LTEE_phylogenetic_trees")
CONSTRAINT_FILE = Path("LTEE_internal_node_time_constraints.csv")

OUT = "LTEE_internal_node_inferred_times.csv"


def get_generation(name):
    if name is None:
        return None

    m = re.search(r"_(\d+)gen_", name)

    if m:
        return int(m.group(1))

    return None


def name_internal_nodes(tree, population):
    counter = 0

    for node in tree.get_nonterminals(order="preorder"):
        if node.name is None:
            node.name = f"{population}_internal_{counter}"
            counter += 1


constraints = pd.read_csv(CONSTRAINT_FILE)

rows = []


for tree_file in sorted(TREE_DIR.glob("*.tre")):

    population = tree_file.stem

    print(f"\nProcessing {population}")

    tree = Phylo.read(tree_file, "newick")
    name_internal_nodes(tree, population)

    # ========================================================
    # Observed tips:
    # root distance <-> known generation
    # ========================================================

    tip_rows = []

    for tip in tree.get_terminals():

        generation = get_generation(tip.name)

        if generation is None:
            continue

        distance = tree.distance(
            tree.root,
            tip
        )

        tip_rows.append({
            "clone": tip.name,
            "generation": generation,
            "root_distance": distance
        })


    tips = pd.DataFrame(tip_rows)

    # --------------------------------------------------------
    # Correlation for diagnostic purposes
    # --------------------------------------------------------

    if (
        tips["root_distance"].std() > 0
        and tips["generation"].std() > 0
    ):
        correlation = tips[
            ["root_distance", "generation"]
        ].corr().iloc[0, 1]
    else:
        correlation = np.nan


    print(
        "  tips:",
        len(tips),
        "correlation:",
        round(correlation, 3)
    )


    # ========================================================
    # Fit isotonic regression
    #
    # generation should generally increase as accumulated
    # mutation distance increases.
    # ========================================================

    x = tips["root_distance"].to_numpy()
    y = tips["generation"].to_numpy()

    model = IsotonicRegression(
        increasing=True,
        out_of_bounds="clip"
    )

    model.fit(x, y)


    # ========================================================
    # Infer internal-node times
    # ========================================================

    pop_constraints = constraints[
        constraints["population"] == population
    ].set_index("node")


    for node in tree.get_nonterminals(order="preorder"):

        root_distance = tree.distance(
            tree.root,
            node
        )

        raw_estimate = float(
            model.predict([root_distance])[0]
        )

        if node.name not in pop_constraints.index:
            continue

        c = pop_constraints.loc[node.name]

        upper = float(
            c["latest_possible_generation"]
        )

        # ----------------------------------------------------
        # HARD constraint:
        # ancestor cannot be later than earliest descendant.
        # ----------------------------------------------------

        constrained_estimate = min(
            raw_estimate,
            upper
        )

        # Experiment starts at generation 0
        constrained_estimate = max(
            constrained_estimate,
            0.0
        )

        # Root is the founder time
        if node == tree.root:
            constrained_estimate = 0.0
            raw_estimate = 0.0


        # ----------------------------------------------------
        # Find observed tips bracketing this root distance.
        #
        # These are NOT hard temporal bounds.
        # They are empirical calibration neighbors.
        # ----------------------------------------------------

        below = tips[
            tips["root_distance"] <= root_distance
        ]

        above = tips[
            tips["root_distance"] >= root_distance
        ]


        if len(below) > 0:

            closest_below = below.loc[
                below["root_distance"].idxmax()
            ]

            calibration_lower_generation = int(
                closest_below["generation"]
            )

            calibration_lower_distance = float(
                closest_below["root_distance"]
            )

        else:

            calibration_lower_generation = np.nan
            calibration_lower_distance = np.nan


        if len(above) > 0:

            closest_above = above.loc[
                above["root_distance"].idxmin()
            ]

            calibration_upper_generation = int(
                closest_above["generation"]
            )

            calibration_upper_distance = float(
                closest_above["root_distance"]
            )

        else:

            calibration_upper_generation = np.nan
            calibration_upper_distance = np.nan


        # ----------------------------------------------------
        # Confidence flag
        # ----------------------------------------------------

        if correlation >= 0.95:
            confidence = "high"

        elif correlation >= 0.85:
            confidence = "moderate"

        else:
            confidence = "low"


        rows.append({
            "population":
                population,

            "node":
                node.name,

            "root_distance":
                root_distance,

            "hard_lower_generation":
                0,

            "hard_upper_generation":
                upper,

            "raw_isotonic_generation":
                raw_estimate,

            "estimated_generation":
                constrained_estimate,

            "population_tip_correlation":
                correlation,

            "confidence":
                confidence,

            "calibration_lower_generation":
                calibration_lower_generation,

            "calibration_lower_distance":
                calibration_lower_distance,

            "calibration_upper_generation":
                calibration_upper_generation,

            "calibration_upper_distance":
                calibration_upper_distance
        })


df = pd.DataFrame(rows)

df = df.sort_values(
    ["population", "estimated_generation", "root_distance"]
)

df.to_csv(
    OUT,
    index=False
)


print("\n========================================")
print("DONE")
print("========================================")

print("Internal nodes:", len(df))

print("\nConfidence:")
print(df["confidence"].value_counts())

print("\nEstimated generation summary:")
print(df["estimated_generation"].describe())

print("\nBy population:")

summary = (
    df.groupby("population")
    .agg(
        nodes=("node", "size"),
        correlation=("population_tip_correlation", "first"),
        median_estimated_generation=(
            "estimated_generation",
            "median"
        ),
        max_estimated_generation=(
            "estimated_generation",
            "max"
        )
    )
)

print(summary)

print("\nExample:")
print(
    df[
        [
            "population",
            "node",
            "root_distance",
            "hard_upper_generation",
            "estimated_generation",
            "confidence"
        ]
    ]
    .head(40)
    .to_string(index=False)
)

print("\nSaved:", OUT)
