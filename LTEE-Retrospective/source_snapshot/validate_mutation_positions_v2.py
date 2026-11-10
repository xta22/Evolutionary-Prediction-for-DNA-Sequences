from pathlib import Path

import pandas as pd


MANIFEST = Path(
    "LTEE_training_dataset/LTEE_training_manifest.csv"
)

NODE_STATES = Path(
    "LTEE_internal_node_SNP_states.csv"
)

OUT = Path(
    "LTEE_mutation_position_validation_v2.csv"
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

            seq.append(
                line.upper()
            )

    return "".join(seq)


# ============================================================
# Load reconstructed internal-node SNP states
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

missing = (
    required
    - set(states.columns)
)

if missing:

    raise ValueError(
        f"Missing columns: {missing}"
    )


# ------------------------------------------------------------
# Represent every node as:
#
# position -> new_base
#
# ------------------------------------------------------------

state_lookup = {}

for (
    population,
    node
), sub in states.groupby(
    [
        "population",
        "node"
    ]
):

    state_lookup[
        (
            population,
            node
        )
    ] = {
        int(row["position"]):
            str(
                row["new_base"]
            ).upper()

        for _, row in sub.iterrows()
    }


# ============================================================
# Manifest
# ============================================================

manifest = pd.read_csv(
    MANIFEST
)


# ============================================================
# Validate every branch
# ============================================================

results = []


for _, row in manifest.iterrows():

    population = row["population"]

    parent = row["parent"]
    child = row["child"]


    # --------------------------------------------------------
    # Read sequences
    # --------------------------------------------------------

    parent_seq = read_fasta(
        row["parent_sequence_file"]
    )

    child_seq = read_fasta(
        row["child_sequence_file"]
    )


    if len(parent_seq) != len(child_seq):

        raise ValueError(
            f"Sequence length mismatch: "
            f"{row['branch_id']}"
        )


    # --------------------------------------------------------
    # ACTUAL differences
    # --------------------------------------------------------

    actual_positions = {

        i

        for i, (
            a,
            b
        ) in enumerate(
            zip(
                parent_seq,
                child_seq
            ),
            start=1
        )

        if a != b
    }


    # --------------------------------------------------------
    # Reconstructed parent/child states
    # --------------------------------------------------------

    parent_state = state_lookup.get(
        (
            population,
            parent
        ),
        {}
    )

    child_state = state_lookup.get(
        (
            population,
            child
        ),
        {}
    )


    # --------------------------------------------------------
    # Gain positions
    #
    # A position is gained if:
    #
    # child has a mutation state
    # and parent does not have the same state.
    #
    # For this validation we only track positions.
    # --------------------------------------------------------

    positions = (
        set(parent_state)
        |
        set(child_state)
    )


    reconstructed_positions = set()

    for position in positions:

        parent_base = (
            parent_state.get(
                position,
                None
            )
        )

        child_base = (
            child_state.get(
                position,
                None
            )
        )

        if (
            parent_base
            !=
            child_base
        ):

            reconstructed_positions.add(
                position
            )


    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    missing_positions = (
        reconstructed_positions
        -
        actual_positions
    )

    extra_positions = (
        actual_positions
        -
        reconstructed_positions
    )


    match = (
        reconstructed_positions
        ==
        actual_positions
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
            len(
                actual_positions
            ),

        "reconstructed_position_count":
            len(
                reconstructed_positions
            ),

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
            )
    })


# ============================================================
# Save
# ============================================================

result = pd.DataFrame(
    results
)

result.to_csv(
    OUT,
    index=False
)


# ============================================================
# Summary
# ============================================================

print(
    "\n========================================"
)

print(
    "POSITION-LEVEL MUTATION VALIDATION V2"
)

print(
    "========================================"
)

print(
    "\nTotal branches:",
    len(result)
)

print(
    "Matches:",
    result["position_match"].sum()
)

print(
    "Mismatches:",
    (
        ~result["position_match"]
    ).sum()
)


# ------------------------------------------------------------
# Temporal subset
# ------------------------------------------------------------

temporal_ids = set(
    manifest[
        manifest["delta_time"] > 0
    ]["branch_id"]
)

temporal = result[
    result["branch_id"].isin(
        temporal_ids
    )
]

print(
    "\nTemporal branches:",
    len(temporal)
)

print(
    "Temporal matches:",
    temporal[
        "position_match"
    ].sum()
)

print(
    "Temporal mismatches:",
    (
        ~temporal[
            "position_match"
        ]
    ).sum()
)


# ------------------------------------------------------------
# By population
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Mismatches
# ------------------------------------------------------------

mismatches = result[
    ~result["position_match"]
]

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
                "actual_mutation_count",
                "reconstructed_position_count",
                "missing_positions",
                "extra_positions"
            ]
        ]
        .head(50)
        .to_string(
            index=False
        )
    )


print(
    "\nSaved:",
    OUT
)
