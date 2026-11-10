from pathlib import Path
import re

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import Phylo


TREE_DIR = Path("summary/LTEE_phylogenetic_trees")
GD_DIR = Path("LTEE-clone-curated")

OUT_DIR = Path("dating_inputs/observed_alignments")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Reference
# ============================================================

REFERENCE_FILE = Path("reference/REL606.gbk")

if not REFERENCE_FILE.exists():
    raise FileNotFoundError(
        f"Reference not found: {REFERENCE_FILE}"
    )

reference_record = SeqIO.read(
    REFERENCE_FILE,
    "genbank"
)

reference_seq = str(
    reference_record.seq
).upper()

reference_length = len(reference_seq)

print("Reference:", REFERENCE_FILE)
print("Reference ID:", reference_record.id)
print("Reference length:", reference_length)


# ============================================================
# Helpers
# ============================================================

def get_gd_file(clone_name):
    """
    Convert tree tip name to actual GD filename.

    Handles the known Ara-2 naming discrepancy:
        tree:
            Ara-2_30000gen_30K_11832

        file:
            Ara-2_30000gen_11832.gd
    """

    normalized = clone_name.replace(
        "_30K_",
        "_"
    )

    return GD_DIR / f"{normalized}.gd"


def read_snp_records(gd_file):
    """
    Read SNPs from a GenomeDiff file.

    Returns:
        list of
        (refseq, position, new_base)
    """

    snps = []

    with open(gd_file) as f:

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

            refseq = fields[3]
            position = int(fields[4])
            new_base = fields[5].upper()

            snps.append(
                (
                    refseq,
                    position,
                    new_base
                )
            )

    return snps


def make_sequence_from_snps(snps):
    """
    Apply SNPs to REL606 reference.

    GenomeDiff positions are treated as 1-based.
    """

    sequence = list(reference_seq)

    for refseq, position, new_base in snps:

        index = position - 1

        if index < 0 or index >= reference_length:
            raise ValueError(
                f"Position out of range: {position}"
            )

        if new_base not in {
            "A", "C", "G", "T"
        }:
            raise ValueError(
                f"Unexpected SNP base: "
                f"{new_base}"
            )

        sequence[index] = new_base

    return "".join(sequence)


# ============================================================
# Process each official population tree
# ============================================================

global_summary = []


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

    records = []
    seen_names = set()

    for tip in tree.get_terminals():

        clone = tip.name

        if clone in seen_names:
            raise ValueError(
                f"Duplicate tip name: {clone}"
            )

        seen_names.add(clone)

        gd_file = get_gd_file(
            clone
        )

        if not gd_file.exists():

            raise FileNotFoundError(
                f"\nMissing GD file for tree tip:\n"
                f"  clone: {clone}\n"
                f"  expected: {gd_file}"
            )

        snps = read_snp_records(
            gd_file
        )

        sequence = make_sequence_from_snps(
            snps
        )

        # ----------------------------------------------------
        # Store as full-length DNA sequence
        # ----------------------------------------------------

        record = SeqRecord(
            Seq(sequence),
            id=clone,
            description=(
                f"population={population} "
                f"snp_count={len(snps)} "
                f"reference=REL606"
            )
        )

        records.append(record)

        global_summary.append({
            "population":
                population,

            "clone":
                clone,

            "generation":
                (
                    int(
                        re.search(
                            r"_(\d+)gen_",
                            clone
                        ).group(1)
                    )
                    if re.search(
                        r"_(\d+)gen_",
                        clone
                    )
                    else None
                ),

            "snp_count":
                len(snps),

            "sequence_length":
                len(sequence)
        })

    # --------------------------------------------------------
    # Check all sequences same length
    # --------------------------------------------------------

    lengths = {
        len(record.seq)
        for record in records
    }

    if lengths != {reference_length}:

        raise ValueError(
            f"{population}: "
            f"unexpected sequence lengths: {lengths}"
        )

    # --------------------------------------------------------
    # Save population alignment
    # --------------------------------------------------------

    output_file = (
        OUT_DIR
        / f"{population}.fasta"
    )

    SeqIO.write(
        records,
        output_file,
        "fasta"
    )

    print(
        "  tips:",
        len(records)
    )

    print(
        "  alignment length:",
        reference_length
    )

    print(
        "  saved:",
        output_file
    )


# ============================================================
# Save summary
# ============================================================

import pandas as pd

summary = pd.DataFrame(
    global_summary
)

summary.to_csv(
    "observed_clone_alignment_summary.csv",
    index=False
)


# ============================================================
# Final summary
# ============================================================

print(
    "\n========================================"
)

print(
    "DONE"
)

print(
    "========================================"
)

print(
    "Populations:",
    summary["population"].nunique()
)

print(
    "Total sequences:",
    len(summary)
)

print(
    "Sequence length:",
    reference_length
)

print(
    "\nSequences per population:"
)

print(
    summary
    .groupby("population")
    .size()
)

print(
    "\nSNP count summary:"
)

print(
    summary["snp_count"].describe()
)

print(
    "\nSaved alignment directory:"
)

print(
    OUT_DIR
)

print(
    "\nSaved summary:"
)

print(
    "observed_clone_alignment_summary.csv"
)
