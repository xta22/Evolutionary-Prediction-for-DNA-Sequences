from pathlib import Path
import pandas as pd


MANIFEST = Path(
    "LTEE_training_dataset/LTEE_training_manifest.csv"
)

OUT = Path(
    "LTEE_branch_mutation_position_summary.csv"
)


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


results = []


df = pd.read_csv(MANIFEST)


for _, row in df.iterrows():

    parent = read_fasta(
        row["parent_sequence_file"]
    )

    child = read_fasta(
        row["child_sequence_file"]
    )

    if len(parent) != len(child):
        raise ValueError(
            f"Length mismatch: {row['branch_id']}"
        )

    mutation_positions = []

    transitions = []

    for pos, (p, c) in enumerate(
        zip(parent, child),
        start=1
    ):

        if p != c:

            mutation_positions.append(pos)

            transitions.append(
                f"{pos}:{p}>{c}"
            )

    actual_count = len(
        mutation_positions
    )

    expected_count = int(
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

        "delta_time":
            row["delta_time"],

        "actual_mutation_count":
            actual_count,

        "reconstructed_changes":
            expected_count,

        "count_match":
            actual_count == expected_count,

        "mutation_positions":
            ",".join(
                map(
                    str,
                    mutation_positions
                )
            ),

        "nucleotide_transitions":
            ",".join(
                transitions
            )
    })


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
    "BRANCH MUTATION POSITION SUMMARY"
)

print(
    "========================================"
)

print(
    "\nTotal branches:",
    len(result)
)

print(
    "Count matches:",
    result["count_match"].sum()
)

print(
    "Count mismatches:",
    (
        ~result["count_match"]
    ).sum()
)


temporal = result[
    result["delta_time"] > 0
]

print(
    "\nTemporal branches:",
    len(temporal)
)

print(
    "Temporal count matches:",
    temporal[
        "count_match"
    ].sum()
)


print(
    "\n=== EXAMPLE MUTATIONS ==="
)

examples = result[
    result["actual_mutation_count"] > 0
].head(10)

print(
    examples[
        [
            "branch_id",
            "parent",
            "child",
            "actual_mutation_count",
            "nucleotide_transitions"
        ]
    ].to_string(index=False)
)


print(
    "\nSaved:",
    OUT
)
