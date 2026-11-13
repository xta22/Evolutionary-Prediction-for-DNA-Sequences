from pathlib import Path
import pandas as pd

BASE = Path("LTEE_training_dataset")
MANIFEST = BASE / "LTEE_training_manifest.csv"
MUTATIONS = Path("LTEE_branch_mutation_position_summary.csv")
SPLIT_DIR = BASE / "splits"

OUT = Path("LTEE_model_ready")
OUT.mkdir(exist_ok=True)

# ------------------------------------------------------------
# Load
# ------------------------------------------------------------

manifest = pd.read_csv(MANIFEST)
mut = pd.read_csv(MUTATIONS)

# only temporal branches
manifest = manifest[
    manifest["delta_time"] > 0
].copy()

# mutation positions / transitions
mut = mut[
    [
        "branch_id",
        "actual_mutation_count",
        "mutation_positions",
        "nucleotide_transitions"
    ]
].copy()

df = manifest.merge(
    mut,
    on="branch_id",
    how="left",
    validate="one_to_one"
)

# ------------------------------------------------------------
# Load existing temporal split assignments
# ------------------------------------------------------------

split_map = {}

for split in ["train", "validation", "test"]:

    f = SPLIT_DIR / f"{split}_manifest.csv"

    x = pd.read_csv(f)

    for branch_id in x["branch_id"]:
        split_map[branch_id] = split

df["split"] = df["branch_id"].map(split_map)

if df["split"].isna().any():
    bad = df.loc[
        df["split"].isna(),
        "branch_id"
    ].tolist()

    raise ValueError(
        f"Branches without split: {bad[:10]}"
    )

# ------------------------------------------------------------
# Core model fields
# ------------------------------------------------------------

keep = [
    "branch_id",
    "split",
    "population",

    "parent",
    "child",

    "parent_time",
    "child_time",
    "delta_time",

    "parent_time_source",
    "child_time_source",
    "parent_confidence",
    "child_confidence",

    "parent_sequence_file",
    "child_sequence_file",
    "sequence_length",

    "actual_mutation_count",
    "mutation_positions",
    "nucleotide_transitions",

    "mutation_rate",
    "branch_length",
]

df = df[keep].copy()

# ------------------------------------------------------------
# Useful derived labels
# ------------------------------------------------------------

df["has_mutation"] = (
    df["actual_mutation_count"] > 0
)

df["mutations_per_1000_generations"] = (
    df["actual_mutation_count"]
    / df["delta_time"]
    * 1000
)

# Dating reliability label.
# Observed child dates are exact experimental sampling times;
# internal dates remain inferred.

def temporal_quality(row):

    parent_internal = (
        row["parent_time_source"]
        == "inferred_internal"
    )

    child_observed = (
        row["child_time_source"]
        == "observed"
    )

    if child_observed:
        return "observed_endpoint"

    if parent_internal:
        return "fully_inferred_branch"

    return "other"

df["temporal_quality"] = df.apply(
    temporal_quality,
    axis=1
)

# ------------------------------------------------------------
# QC
# ------------------------------------------------------------

assert len(df) == 429
assert (df["delta_time"] > 0).all()
assert df["branch_id"].is_unique
assert df["actual_mutation_count"].notna().all()

# ------------------------------------------------------------
# Save master
# ------------------------------------------------------------

master = OUT / "model_manifest.csv"

df.to_csv(
    master,
    index=False
)

# ------------------------------------------------------------
# Save each split
# ------------------------------------------------------------

for split in [
    "train",
    "validation",
    "test"
]:

    subset = df[
        df["split"] == split
    ].copy()

    subset.to_csv(
        OUT / f"{split}.csv",
        index=False
    )

# ------------------------------------------------------------
# Also save zero-mutation and mutated subsets
# ------------------------------------------------------------

df[
    df["actual_mutation_count"] == 0
].to_csv(
    OUT / "zero_mutation_branches.csv",
    index=False
)

df[
    df["actual_mutation_count"] > 0
].to_csv(
    OUT / "mutated_branches.csv",
    index=False
)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\n========================================")
print("LTEE MODEL-READY DATASET V1")
print("========================================")

print("\nSamples:", len(df))

print("\n=== SPLITS ===")
print(df["split"].value_counts())

print("\n=== POPULATION x SPLIT ===")
print(
    pd.crosstab(
        df["population"],
        df["split"]
    )
)

print("\n=== MUTATION STATUS ===")
print(
    df["has_mutation"].value_counts()
)

print("\n=== MUTATIONS PER BRANCH ===")
print(
    df["actual_mutation_count"].describe()
)

print("\n=== DELTA TIME ===")
print(
    df["delta_time"].describe()
)

print("\n=== TEMPORAL QUALITY ===")
print(
    df["temporal_quality"].value_counts()
)

print("\n=== TIME CONFIDENCE ===")
print(
    df["parent_confidence"]
    .value_counts(dropna=False)
)

print("\n=== BY SPLIT ===")

print(
    df.groupby("split").agg(
        n=("branch_id", "size"),
        mutated=("has_mutation", "sum"),
        median_dt=("delta_time", "median"),
        median_mutations=(
            "actual_mutation_count",
            "median"
        ),
        mean_mutations=(
            "actual_mutation_count",
            "mean"
        ),
    )
)

print("\nSaved:")
print(master)

for split in [
    "train",
    "validation",
    "test"
]:
    print(OUT / f"{split}.csv")
