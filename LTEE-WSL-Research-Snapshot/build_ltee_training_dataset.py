from pathlib import Path
import re

import numpy as np
import pandas as pd
from Bio import Phylo, SeqIO


# ============================================================
# Paths
# ============================================================

TREE_DIR = Path(
    "summary/LTEE_phylogenetic_trees"
)

INTERNAL_FASTA_DIR = Path(
    "LTEE_internal_node_fastas"
)

OBSERVED_FASTA_DIR = Path(
    "dating_inputs/observed_alignments"
)

TIME_FILE = Path(
    "LTEE_internal_node_inferred_times.csv"
)

BRANCH_FILE = Path(
    "LTEE_phylogenetic_branch_SNP_changes.csv"
)

OUT_DIR = Path(
    "LTEE_training_dataset"
)

SEQ_DIR = OUT_DIR / "sequences"

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SEQ_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Helpers
# ============================================================

def get_generation(name):

    if name is None:
        return None

    m = re.search(
        r"_(\d+)gen_",
        name
    )

    if m:
        return int(m.group(1))

    return None


def load_fasta_file(fasta_file):

    sequences = {}

    for record in SeqIO.parse(
        fasta_file,
        "fasta"
    ):

        sequences[record.id] = str(
            record.seq
        ).upper()

    return sequences


def load_all_internal_sequences(directory):

    sequences = {}

    fasta_files = sorted(
        list(directory.glob("*.fasta"))
        +
        list(directory.glob("*.fa"))
    )

    print(
        "Internal FASTA files:",
        len(fasta_files)
    )

    for fasta_file in fasta_files:

        local = load_fasta_file(
            fasta_file
        )

        for name, seq in local.items():

            if name in sequences:

                if sequences[name] != seq:

                    raise ValueError(
                        "Conflicting sequences "
                        f"for {name}"
                    )

            sequences[name] = seq

    return sequences


def descendant_tips(node):

    return sorted(
        tip.name
        for tip in node.get_terminals()
    )


def write_fasta(
    filename,
    name,
    sequence
):

    with open(filename, "w") as f:

        f.write(
            f">{name}\n"
        )

        for i in range(
            0,
            len(sequence),
            80
        ):

            f.write(
                sequence[i:i + 80]
                + "\n"
            )


# ============================================================
# Validate input files
# ============================================================

required_files = [
    TIME_FILE,
    BRANCH_FILE,
    TREE_DIR,
    INTERNAL_FASTA_DIR,
    OBSERVED_FASTA_DIR,
]

for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Required input not found: {path}"
        )


# ============================================================
# Load internal sequences ONCE
# ============================================================

print(
    "\nLoading internal-node FASTAs..."
)

internal_sequences = (
    load_all_internal_sequences(
        INTERNAL_FASTA_DIR
    )
)

print(
    "Internal-node sequences:",
    len(internal_sequences)
)


# ============================================================
# Load time information
# ============================================================

time_df = pd.read_csv(
    TIME_FILE
)

required_time_columns = {
    "population",
    "node",
    "estimated_generation",
    "confidence",
}

missing = (
    required_time_columns
    -
    set(time_df.columns)
)

if missing:

    raise ValueError(
        "Missing time columns: "
        f"{missing}"
    )


time_lookup = {}

for _, row in time_df.iterrows():

    time_lookup[
        (
            row["population"],
            row["node"]
        )
    ] = {

        "time":
            float(
                row[
                    "estimated_generation"
                ]
            ),

        "confidence":
            row["confidence"],

        "hard_upper":
            (
                float(
                    row[
                        "hard_upper_generation"
                    ]
                )
                if (
                    "hard_upper_generation"
                    in row.index
                    and
                    not pd.isna(
                        row[
                            "hard_upper_generation"
                        ]
                    )
                )
                else np.nan
            )
    }


# ============================================================
# Load branch mutation information
# ============================================================

branch_df = pd.read_csv(
    BRANCH_FILE
)

branch_lookup = {}

for _, row in branch_df.iterrows():

    key = (
        row["population"],
        row["parent"],
        row["child"],
    )

    branch_lookup[key] = {

        "snps_gained":
            row["snps_gained"],

        "snps_lost":
            row["snps_lost"],

        "reconstructed_changes":
            row[
                "reconstructed_changes"
            ],

        "branch_length":
            row[
                "branch_length"
            ],
    }


# ============================================================
# Build dataset
# ============================================================

rows = []

branch_counter = 0
skipped = 0


for tree_file in sorted(
    TREE_DIR.glob("*.tre")
):

    population = tree_file.stem

    print(
        f"\nProcessing {population}"
    )

    tree = Phylo.read(
        tree_file,
        "newick"
    )

    # --------------------------------------------------------
    # Name internal nodes exactly as before
    # --------------------------------------------------------

    counter = 0

    for node in tree.get_nonterminals(
        order="preorder"
    ):

        node.name = (
            f"{population}_internal_{counter}"
        )

        counter += 1


    # --------------------------------------------------------
    # Load observed clone sequences
    # --------------------------------------------------------

    observed_file = (
        OBSERVED_FASTA_DIR
        / f"{population}.fasta"
    )

    observed_sequences = (
        load_fasta_file(
            observed_file
        )
    )


    # --------------------------------------------------------
    # Process every branch
    # --------------------------------------------------------

    for parent in tree.find_clades(
        order="preorder"
    ):

        parent_name = parent.name

        # ----------------------------------------------------
        # Find parent sequence
        # ----------------------------------------------------

        parent_seq = (
            internal_sequences.get(
                parent_name
            )
        )

        if parent_seq is None:

            parent_seq = (
                observed_sequences.get(
                    parent_name
                )
            )

        if parent_seq is None:

            print(
                "  SKIP missing parent:",
                parent_name
            )

            skipped += 1
            continue


        # ----------------------------------------------------
        # Parent time
        # ----------------------------------------------------

        parent_observed_generation = (
            get_generation(
                parent_name
            )
        )

        parent_inferred_generation = None
        parent_confidence = None

        key = (
            population,
            parent_name
        )

        if key in time_lookup:

            parent_inferred_generation = (
                time_lookup[key]["time"]
            )

            parent_confidence = (
                time_lookup[key]["confidence"]
            )


        if (
            parent_inferred_generation
            is not None
        ):

            parent_time = (
                parent_inferred_generation
            )

            parent_time_source = (
                "inferred_internal"
            )

        elif (
            parent_observed_generation
            is not None
        ):

            parent_time = float(
                parent_observed_generation
            )

            parent_time_source = (
                "observed"
            )

        else:

            print(
                "  SKIP no parent time:",
                parent_name
            )

            skipped += 1
            continue


        # ----------------------------------------------------
        # Children
        # ----------------------------------------------------

        for child in parent.clades:

            child_name = child.name

            # ------------------------------------------------
            # Child sequence
            # ------------------------------------------------

            child_seq = (
                internal_sequences.get(
                    child_name
                )
            )

            if child_seq is None:

                child_seq = (
                    observed_sequences.get(
                        child_name
                    )
                )

            if child_seq is None:

                print(
                    "  SKIP missing child:",
                    child_name
                )

                skipped += 1
                continue


            # ------------------------------------------------
            # Sequence length
            # ------------------------------------------------

            if (
                len(parent_seq)
                !=
                len(child_seq)
            ):

                raise ValueError(
                    "Sequence length mismatch:\n"
                    f"{parent_name}: "
                    f"{len(parent_seq)}\n"
                    f"{child_name}: "
                    f"{len(child_seq)}"
                )


            # ------------------------------------------------
            # Child time
            # ------------------------------------------------

            child_observed_generation = (
                get_generation(
                    child_name
                )
            )

            child_inferred_generation = None
            child_confidence = None

            key = (
                population,
                child_name
            )

            if key in time_lookup:

                child_inferred_generation = (
                    time_lookup[key]["time"]
                )

                child_confidence = (
                    time_lookup[key]["confidence"]
                )


            if (
                child_inferred_generation
                is not None
            ):

                child_time = (
                    child_inferred_generation
                )

                child_time_source = (
                    "inferred_internal"
                )

            elif (
                child_observed_generation
                is not None
            ):

                child_time = float(
                    child_observed_generation
                )

                child_time_source = (
                    "observed"
                )

            else:

                print(
                    "  SKIP no child time:",
                    child_name
                )

                skipped += 1
                continue


            # ------------------------------------------------
            # Time interval
            # ------------------------------------------------

            delta_time = (
                child_time
                -
                parent_time
            )

            time_order_ok = (
                delta_time >= 0
            )


            # ------------------------------------------------
            # Mutation information
            # ------------------------------------------------

            mutation_info = (
                branch_lookup.get(
                    (
                        population,
                        parent_name,
                        child_name
                    ),
                    {}
                )
            )


            branch_length = (
                mutation_info.get(
                    "branch_length",
                    child.branch_length
                )
            )

            snps_gained = (
                mutation_info.get(
                    "snps_gained",
                    np.nan
                )
            )

            snps_lost = (
                mutation_info.get(
                    "snps_lost",
                    np.nan
                )
            )

            reconstructed_changes = (
                mutation_info.get(
                    "reconstructed_changes",
                    np.nan
                )
            )


            # ------------------------------------------------
            # Mutation rate
            # ------------------------------------------------

            if (
                delta_time > 0
                and
                not pd.isna(
                    reconstructed_changes
                )
            ):

                mutation_rate = (
                    float(
                        reconstructed_changes
                    )
                    /
                    delta_time
                )

            else:

                mutation_rate = np.nan


            # ------------------------------------------------
            # Sequence files
            # ------------------------------------------------

            branch_id = (
                f"{population}_branch_"
                f"{branch_counter:04d}"
            )

            parent_file = (
                SEQ_DIR
                /
                f"{branch_id}_parent.fasta"
            )

            child_file = (
                SEQ_DIR
                /
                f"{branch_id}_child.fasta"
            )


            write_fasta(
                parent_file,
                parent_name,
                parent_seq
            )

            write_fasta(
                child_file,
                child_name,
                child_seq
            )


            # ------------------------------------------------
            # Manifest row
            # ------------------------------------------------

            rows.append({

                "branch_id":
                    branch_id,

                "population":
                    population,

                "parent":
                    parent_name,

                "child":
                    child_name,

                "parent_time":
                    parent_time,

                "child_time":
                    child_time,

                "delta_time":
                    delta_time,

                "parent_time_source":
                    parent_time_source,

                "child_time_source":
                    child_time_source,

                "parent_confidence":
                    parent_confidence,

                "child_confidence":
                    child_confidence,

                "time_order_ok":
                    time_order_ok,

                "branch_length":
                    branch_length,

                "snps_gained":
                    snps_gained,

                "snps_lost":
                    snps_lost,

                "reconstructed_changes":
                    reconstructed_changes,

                "mutation_rate":
                    mutation_rate,

                "parent_sequence_file":
                    str(parent_file),

                "child_sequence_file":
                    str(child_file),

                "sequence_length":
                    len(parent_seq),

                "n_descendant_tips":
                    len(
                        descendant_tips(
                            child
                        )
                    )
            })


            branch_counter += 1


# ============================================================
# Save
# ============================================================

manifest = pd.DataFrame(
    rows
)

manifest_file = (
    OUT_DIR
    /
    "LTEE_training_manifest.csv"
)

manifest.to_csv(
    manifest_file,
    index=False
)


# ============================================================
# Summary
# ============================================================

print(
    "\n========================================"
)

print(
    "LTEE TRAINING DATASET"
)

print(
    "========================================"
)

print(
    "\nComplete branches:",
    len(manifest)
)

print(
    "Skipped branches:",
    skipped
)

print(
    "\nPopulations:"
)

print(
    manifest[
        "population"
    ].value_counts()
    .sort_index()
)


print(
    "\n=== TIME SOURCES ==="
)

print(
    manifest[
        [
            "parent_time_source",
            "child_time_source"
        ]
    ]
    .value_counts()
)


print(
    "\n=== TIME ORDER ==="
)

print(
    manifest[
        "time_order_ok"
    ].value_counts()
)


negative = (
    manifest["delta_time"] < 0
).sum()

print(
    "\nNegative delta_time:",
    negative
)


zero = (
    manifest["delta_time"] == 0
).sum()

print(
    "Zero delta_time:",
    zero
)


print(
    "\n=== DELTA TIME ==="
)

print(
    manifest[
        "delta_time"
    ].describe()
)


print(
    "\n=== RECONSTRUCTED CHANGES ==="
)

print(
    manifest[
        "reconstructed_changes"
    ].describe()
)


print(
    "\n=== MUTATION RATE ==="
)

print(
    manifest[
        "mutation_rate"
    ].describe()
)


print(
    "\n=== SEQUENCE LENGTH ==="
)

print(
    manifest[
        "sequence_length"
    ].value_counts()
)


print(
    "\nSaved manifest:"
)

print(
    manifest_file
)

print(
    "\nSaved sequence directory:"
)

print(
    SEQ_DIR
)
