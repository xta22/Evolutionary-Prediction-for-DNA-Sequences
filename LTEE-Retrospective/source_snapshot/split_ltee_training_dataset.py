from pathlib import Path
import shutil

import pandas as pd


INPUT = Path(
    "LTEE_training_dataset/LTEE_training_manifest.csv"
)

OUT_DIR = Path(
    "LTEE_training_dataset/splits"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Thresholds
# ============================================================

TRAIN_MAX = 10000
VALIDATION_MAX = 30000


# ============================================================
# Load
# ============================================================

df = pd.read_csv(INPUT)


# ============================================================
# Keep only temporal branches
# ============================================================

df_temporal = df[
    df["delta_time"] > 0
].copy()


# ============================================================
# Assign split based on CHILD TIME
# ============================================================

def assign_split(child_time):

    if child_time <= TRAIN_MAX:

        return "train"

    elif child_time <= VALIDATION_MAX:

        return "validation"

    else:

        return "test"


df_temporal["split"] = (
    df_temporal["child_time"]
    .apply(assign_split)
)


# ============================================================
# Save manifests
# ============================================================

for split in [
    "train",
    "validation",
    "test"
]:

    subset = df_temporal[
        df_temporal["split"]
        == split
    ].copy()

    output = (
        OUT_DIR
        / f"{split}_manifest.csv"
    )

    subset.to_csv(
        output,
        index=False
    )


# ============================================================
# Also save zero-time branches separately
# ============================================================

zero_time = df[
    df["delta_time"] == 0
].copy()

zero_time.to_csv(
    OUT_DIR
    / "zero_time_branches.csv",
    index=False
)


# ============================================================
# Summary
# ============================================================

print(
    "\n========================================"
)

print(
    "LTEE TEMPORAL DATASET SPLIT"
)

print(
    "========================================"
)


print(
    "\nAll branches:",
    len(df)
)

print(
    "Temporal branches:",
    len(df_temporal)
)

print(
    "Zero-time branches:",
    len(zero_time)
)


print(
    "\n=== SPLIT COUNTS ==="
)

print(
    df_temporal[
        "split"
    ].value_counts()
)


print(
    "\n=== SPLIT BY POPULATION ==="
)

print(
    pd.crosstab(
        df_temporal[
            "population"
        ],
        df_temporal[
            "split"
        ]
    )
)


print(
    "\n=== TIME RANGES ==="
)

print(
    df_temporal
    .groupby("split")
    .agg(
        min_child_time=(
            "child_time",
            "min"
        ),

        max_child_time=(
            "child_time",
            "max"
        ),

        n_branches=(
            "branch_id",
            "count"
        )
    )
)


print(
    "\n=== MUTATION SUMMARY ==="
)

print(
    df_temporal
    .groupby("split")
    .agg(
        mean_mutations=(
            "reconstructed_changes",
            "mean"
        ),

        median_mutations=(
            "reconstructed_changes",
            "median"
        ),

        mean_delta_time=(
            "delta_time",
            "mean"
        ),

        median_delta_time=(
            "delta_time",
            "median"
        )
    )
)


print(
    "\nSaved:"
)

print(
    OUT_DIR
)
