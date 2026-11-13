from pathlib import Path
import re
import ast

import pandas as pd


MANIFEST = Path(
    "LTEE_training_dataset/LTEE_training_manifest.csv"
)

NODE_STATES = Path(
    "LTEE_internal_node_SNP_states.csv"
)

GD_DIR = Path(
    "LTEE-clone-curated"
)

OUT = Path(
    "LTEE_mutation_position_validation.csv"
)


# ============================================================
# FASTA
# ============================================================

def read_fasta(path):

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


# ============================================================
# GD SNPs
# ============================================================

def get_gd_path(clone):

    # Handle known Ara-2 naming issue
    normalized = clone.replace(
        "_30K_",
        "_"
    )

    return GD_DIR / f"{normalized}.gd"


def read_gd_snps(clone):

    path = get_gd_path(clone)

    if not path.exists():
        return set()

    snps = set()

    with open(path) as f:

        for line in f:

            line = line.strip()

            if (
                not line
                or line.startswith("#")
            ):
                continue

            fields = line.split()

            if len(fields) < 6:
                continue

            if fields[0] != "SNP":
                continue

            position = int(fields[4])
            new_base = fields[5].upper()

            snps.add(
                (position, new_base)
            )

    return snps


# ============================================================
# Internal-node states
# ============================================================

states = pd.read_csv(
    NODE_STATES
)

required = {
    "population",
    "node",
    "position",
    "new_base"
}

missing = required - set(states.columns)

if missing:
    raise ValueError(
        f"Missing columns in {NODE_STATES}: {missing}"
    )


state_lookup = {}

for population, sub in states.groupby(
    "population"
):

    for node, node_df in sub.groupby(
        "node"
    ):

        state_lookup[
            (
                population,
                node
            )
        ] = set(
            (
                int(row["position"]),
                str(row["new_base"]).upper()
            )
            for _, row in node_df.iterrows()
        )


# ============================================================
# Manifest
# ============================================================

manifest = pd.read_csv(
    MANIFEST
)


# ============================================================
# Position-level validation
# ============================================================

results = []

for _, row in manifest.iterrows():

    population = row["population"]

    parent = row["parent"]
    child = row["child"]

    parent_seq = read_fasta(
        row["parent_sequence_file"]
    )

    child_seq = read_fasta(
        row["child_sequence_file"]
    )


    # --------------------------------------------------------
    # Actual sequence differences
    # --------------------------------------------------------

    actual_positions = set()

    for i, (a, b) in enumerate(
        zip(parent_seq, child_seq),
        start=1
    ):

        if a != b:

            actual_positions.add(i)


    # --------------------------------------------------------
    # Parent mutation state
    # --------------------------------------------------------

    parent_muts = state_lookup.get(
        (
            population,
            parent
        ),
        None
    )


    # --------------------------------------------------------
    # Child mutation state
    #
    # Internal node:
    #   read reconstructed states
    #
    # Observed clone:
    #   read .gd
    # --------------------------------------------------------

    child_muts = state_lookup.get(
        (
            population,
            child
        ),
        None
    )

    if child_muts is None:

        child_muts = read_gd_snps(
            child
        )


    if parent_muts is None:

        results.append({

            "branch_id":
                row["branch_id"],

            "population":
                population,

            "parent":
                parent,

            "child":
                child,

            "actual_mutation_count":
                len(actual_positions),

            "expected_mutation_count":
                None,

            "position_match":
                False,

            "status":
                "missing_parent_state",

        })

        continue


    # --------------------------------------------------------
    # Expected mutation positions
    #
    # Symmetric difference of parent/child states.
    #
    # We only care about positions here.
    # --------------------------------------------------------

    parent_positions = {
        p for p, base in parent_muts
    }

    child_positions = {
        p for p, base in child_muts
    }


    expected_positions = (
        parent_positions
        ^ child_positions
    )


    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    missing_positions = (
        expected_positions
        - actual_positions
    )

    extra_positions = (
        actual_positions
        - expected_positions
    )


    match = (
        expected_positions
        == actual_positions
    )


    results.append({

        "branch_id":
            row["branch_id"],

        "population":
            population,

        "parent":
            parent,

        "child":
            child,

        "actual_mutation_count":
            len(actual_positions),

        "expected_mutation_count":
            len(expected_positions),

        "position_match":
            match,

        "missing_positions":
            ",".join(
                map(
                    str,
                    sorted(
                        missing_positions
                    )
                )
            ),

        "extra_positions":
            ",".join(
                map(
                    str,
                    sorted(
                        extra_positions
                    )
                )
            ),

        "status":
            (
                "PASS"
                if match
                else "MISMATCH"
            )
    })


# ============================================================
# Summary
# ============================================================

result = pd.DataFrame(
    results
)

result.to_csv(
    OUT,
    index=False
)


print(
    "\n========================================"
)

print(
    "POSITION-LEVEL MUTATION VALIDATION"
)

print(
    "========================================"
)

print(
    "\nTotal branches:",
    len(result)
)

print(
    "Position-level matches:",
    result[
        "position_match"
    ].sum()
)

print(
    "Position-level mismatches:",
    (
        ~result[
            "position_match"
        ]
    ).sum()
)


print(
    "\n=== BY POPULATION ==="
)

summary = (
    result
    .groupby("population")
    .agg(
        branches=(
            "branch_id",
            "size"
        ),

        matches=(
            "position_match",
            "sum"
        )
    )
)

summary["match_fraction"] = (
    summary["matches"]
    /
    summary["branches"]
)

print(
    summary.to_string()
)


mismatch = result[
    ~result["position_match"]
]


if len(mismatch) > 0:

    print(
        "\n=== MISMATCHES ==="
    )

    print(
        mismatch[
            [
                "branch_id",
                "population",
                "parent",
                "child",
                "actual_mutation_count",
                "expected_mutation_count",
                "missing_positions",
                "extra_positions"
            ]
        ]
        .head(50)
        .to_string(index=False)
    )


# ============================================================
# Temporal subset
# ============================================================

temporal = result[
    manifest["delta_time"].to_numpy() > 0
]

print(
    "\nTemporal branches:",
    len(temporal)
)

print(
    "Temporal position matches:",
    temporal[
        "position_match"
    ].sum()
)


print(
    "\nSaved:",
    OUT
)
