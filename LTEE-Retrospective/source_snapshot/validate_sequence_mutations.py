from pathlib import Path

import numpy as np
import pandas as pd


MANIFEST = Path(
    "LTEE_training_dataset/LTEE_training_manifest.csv"
)


def read_fasta_sequence(path):

    seq = []

    with open(path) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                continue

            seq.append(line.upper())

    return "".join(seq)


df = pd.read_csv(MANIFEST)


print("\n========================================")
print("SEQUENCE VS TREE MUTATION VALIDATION")
print("========================================")


results = []


for _, row in df.iterrows():

    parent_file = Path(
        row["parent_sequence_file"]
    )

    child_file = Path(
        row["child_sequence_file"]
    )

    parent = read_fasta_sequence(
        parent_file
    )

    child = read_fasta_sequence(
        child_file
    )

    if len(parent) != len(child):

        raise ValueError(
            f"Length mismatch: "
            f"{row['branch_id']}"
        )

    differences = sum(
        a != b
        for a, b in zip(parent, child)
    )

    reconstructed = int(
        row["reconstructed_changes"]
    )

    results.append({

        "branch_id":
            row["branch_id"],

        "population":
            row["population"],

        "parent":
            row["parent"],

        "child":
            row["child"],

        "sequence_differences":
            differences,

        "reconstructed_changes":
            reconstructed,

        "difference":
            differences
            - reconstructed,

        "match":
            differences
            == reconstructed,

        "delta_time":
            row["delta_time"],

        "split":
            (
                "temporal"
                if row["delta_time"] > 0
                else "zero_time"
            )
    })


check = pd.DataFrame(results)


print(
    "\nTotal branches:",
    len(check)
)

print(
    "Exact mutation-count matches:",
    check["match"].sum()
)

print(
    "Mismatches:",
    (~check["match"]).sum()
)


print(
    "\n=== DIFFERENCE DISTRIBUTION ==="
)

print(
    check["difference"].describe()
)


mismatches = check[
    ~check["match"]
].copy()


if len(mismatches) > 0:

    print(
        "\n=== MISMATCHES ==="
    )

    print(
        mismatches[
            [
                "branch_id",
                "population",
                "parent",
                "child",
                "sequence_differences",
                "reconstructed_changes",
                "difference"
            ]
        ]
        .head(30)
        .to_string(index=False)
    )


# ------------------------------------------------------------
# Temporal subset
# ------------------------------------------------------------

temporal = check[
    check["delta_time"] > 0
].copy()


print(
    "\n=== TEMPORAL BRANCHES ==="
)

print(
    "Temporal branches:",
    len(temporal)
)

print(
    "Temporal exact matches:",
    temporal["match"].sum()
)


# ------------------------------------------------------------
# Population summary
# ------------------------------------------------------------

print(
    "\n=== BY POPULATION ==="
)

summary = (
    check
    .groupby("population")
    .agg(
        branches=("branch_id", "size"),

        exact_matches=(
            "match",
            "sum"
        ),

        mean_sequence_differences=(
            "sequence_differences",
            "mean"
        ),

        mean_reconstructed_changes=(
            "reconstructed_changes",
            "mean"
        )
    )
)

summary["match_fraction"] = (
    summary["exact_matches"]
    /
    summary["branches"]
)

print(
    summary.to_string()
)


check.to_csv(
    "LTEE_sequence_mutation_validation.csv",
    index=False
)


print(
    "\nSaved:",
    "LTEE_sequence_mutation_validation.csv"
)
