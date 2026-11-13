from pathlib import Path
import re

import numpy as np
import pandas as pd
from Bio import Phylo
from scipy.optimize import curve_fit


TREE_DIR = Path("summary/LTEE_phylogenetic_trees")

OUT = "LTEE_forward_dating_powerlaw.csv"

# ------------------------------------------------------------
# Training cutoff
# ------------------------------------------------------------
TRAIN_CUTOFF = 10000


# ------------------------------------------------------------
# Power-law model
#
# generation = a * (1 + distance)^b
#
# +1 avoids issues at distance = 0.
# ------------------------------------------------------------
def power_law(x, a, b):
    return a * np.power(1.0 + x, b)


# ------------------------------------------------------------
# Extract generation from clone name
# ------------------------------------------------------------
def get_generation(name):
    if name is None:
        return None

    m = re.search(r"_(\d+)gen_", name)

    if m:
        return int(m.group(1))

    return None


# ------------------------------------------------------------
# Fit power-law model
# ------------------------------------------------------------
def fit_powerlaw(train):

    x = train["root_distance"].to_numpy(dtype=float)
    y = train["generation"].to_numpy(dtype=float)

    # Need variation
    if len(np.unique(x)) < 2:
        return None

    if len(np.unique(y)) < 2:
        return None

    # --------------------------------------------------------
    # Initial guess
    # --------------------------------------------------------
    p0 = [100.0, 1.0]

    try:

        params, covariance = curve_fit(
            power_law,
            x,
            y,
            p0=p0,
            bounds=(
                [0.000001, 0.01],
                [1e8, 10.0]
            ),
            maxfev=100000
        )

        a, b = params

        return {
            "a": float(a),
            "b": float(b),
            "covariance": covariance
        }

    except Exception as e:

        print(
            "    WARNING: power-law fitting failed:",
            str(e)
        )

        return None


# ------------------------------------------------------------
# Collect results
# ------------------------------------------------------------
results = []


# ============================================================
# Process populations
# ============================================================

for tree_file in sorted(TREE_DIR.glob("*.tre")):

    population = tree_file.stem

    print(
        f"\nProcessing {population}"
    )

    tree = Phylo.read(
        tree_file,
        "newick"
    )


    # --------------------------------------------------------
    # Collect sampled tips
    # --------------------------------------------------------

    rows = []

    for tip in tree.get_terminals():

        generation = get_generation(
            tip.name
        )

        if generation is None:
            continue

        # Do not fit founder
        if generation == 0:
            continue

        root_distance = tree.distance(
            tree.root,
            tip
        )

        rows.append({
            "clone":
                tip.name,

            "generation":
                generation,

            "root_distance":
                root_distance
        })


    tips = pd.DataFrame(rows)


    train = tips[
        tips["generation"] <= TRAIN_CUTOFF
    ].copy()

    test = tips[
        tips["generation"] > TRAIN_CUTOFF
    ].copy()


    print(
        "  train clones:",
        len(train),
        "test clones:",
        len(test)
    )


    if len(train) < 3:
        print(
            "  SKIP: insufficient training data"
        )
        continue


    if len(test) == 0:
        print(
            "  No future samples"
        )
        continue


    # --------------------------------------------------------
    # Fit power-law using ONLY past observations
    # --------------------------------------------------------

    fitted = fit_powerlaw(
        train
    )

    if fitted is None:
        continue


    a = fitted["a"]
    b = fitted["b"]


    print(
        f"  fitted parameters: "
        f"a={a:.6g}, b={b:.6f}"
    )


    # --------------------------------------------------------
    # Predict future tips
    # --------------------------------------------------------

    for _, row in test.iterrows():

        true_generation = float(
            row["generation"]
        )

        root_distance = float(
            row["root_distance"]
        )

        predicted_generation = float(
            power_law(
                root_distance,
                a,
                b
            )
        )

        error = (
            predicted_generation
            - true_generation
        )

        absolute_error = abs(
            error
        )

        results.append({
            "population":
                population,

            "clone":
                row["clone"],

            "train_cutoff":
                TRAIN_CUTOFF,

            "true_generation":
                true_generation,

            "root_distance":
                root_distance,

            "predicted_generation":
                predicted_generation,

            "error":
                error,

            "absolute_error":
                absolute_error,

            "relative_error":
                absolute_error
                / max(true_generation, 1),

            "powerlaw_a":
                a,

            "powerlaw_b":
                b
        })


# ============================================================
# Save results
# ============================================================

results = pd.DataFrame(
    results
)

results.to_csv(
    OUT,
    index=False
)


# ============================================================
# Overall summary
# ============================================================

print(
    "\n========================================"
)

print(
    "FORWARD TEMPORAL VALIDATION"
)

print(
    "POWER-LAW EXTRAPOLATION"
)

print(
    "========================================"
)

print(
    "\nTraining cutoff:",
    TRAIN_CUTOFF
)

print(
    "Total future predictions:",
    len(results)
)


if len(results) == 0:

    print(
        "\nNo predictions were generated."
    )

    raise SystemExit


# ============================================================
# Overall error
# ============================================================

print(
    "\n=== OVERALL ERROR ==="
)

print(
    results[
        [
            "absolute_error",
            "relative_error"
        ]
    ].describe()
)


# ============================================================
# Error by future generation
# ============================================================

print(
    "\n=== ERROR BY FUTURE GENERATION ==="
)

generation_summary = (
    results
    .groupby("true_generation")
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

        mean_relative_error=(
            "relative_error",
            "mean"
        )
    )
)

print(
    generation_summary.to_string()
)


# ============================================================
# Error by population
# ============================================================

print(
    "\n=== ERROR BY POPULATION ==="
)

population_summary = (
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
    population_summary.to_string()
)


# ============================================================
# Worst predictions
# ============================================================

print(
    "\n=== WORST FORWARD PREDICTIONS ==="
)

print(
    results
    .sort_values(
        "absolute_error",
        ascending=False
    )[
        [
            "population",
            "clone",
            "true_generation",
            "root_distance",
            "predicted_generation",
            "absolute_error",
            "powerlaw_a",
            "powerlaw_b"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# Best predictions
# ============================================================

print(
    "\n=== BEST FORWARD PREDICTIONS ==="
)

print(
    results
    .sort_values(
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


# ============================================================
# Forecast horizon
# ============================================================

results["prediction_horizon"] = (
    results["true_generation"]
    - TRAIN_CUTOFF
)

print(
    "\n=== ERROR BY FORECAST HORIZON ==="
)

horizon_summary = (
    results
    .groupby("prediction_horizon")
    .agg(
        n_predictions=("clone", "size"),

        median_absolute_error=(
            "absolute_error",
            "median"
        ),

        mean_absolute_error=(
            "absolute_error",
            "mean"
        )
    )
)

print(
    horizon_summary.to_string()
)


print(
    "\nSaved:",
    OUT
)
