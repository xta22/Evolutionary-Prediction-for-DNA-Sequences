from pathlib import Path
from collections import defaultdict

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


# ============================================================
# Paths
# ============================================================

SNP_FILE = Path("LTEE_internal_node_SNP_states.csv")

OUT_DIR = Path("LTEE_internal_node_fastas")
OUT_DIR.mkdir(exist_ok=True)


# ============================================================
# Find REL606 reference automatically
# ============================================================

candidate_files = [
    Path("reference/REL606.gbk"),
    Path("reference/REL606.gb"),
    Path("reference/REL606.gbf"),
    Path("reference/REL606.fasta"),
    Path("reference/REL606.fa"),
    Path("REL606.gbk"),
    Path("REL606.fasta"),
]


reference_file = None

for f in candidate_files:
    if f.exists():
        reference_file = f
        break


if reference_file is None:

    print("Could not automatically find REL606 reference.")
    print("\nPossible REL606 files in repository:")

    for f in Path(".").rglob("*REL606*"):
        if f.is_file():
            print(" ", f)

    raise SystemExit(
        "\nSet reference_file manually after locating REL606."
    )


print("Using reference:", reference_file)


# ============================================================
# Load reference genome
# ============================================================

suffix = reference_file.suffix.lower()

if suffix in [".gb", ".gbk", ".gbf", ".genbank"]:

    record = SeqIO.read(reference_file, "genbank")

else:

    record = SeqIO.read(reference_file, "fasta")


reference_seq = str(record.seq).upper()

print("Reference ID:", record.id)
print("Reference length:", len(reference_seq))


# ============================================================
# Load reconstructed internal-node SNP states
# ============================================================

df = pd.read_csv(SNP_FILE)

print("\nInternal SNP-state rows:", len(df))
print("Internal nodes:", df["node"].nunique())
print("Populations:", df["population"].nunique())


# ============================================================
# Check for incompatible multi-allelic states
# ============================================================

conflicts = (
    df.groupby(
        ["population", "node", "position"]
    )["new_base"]
    .nunique()
    .reset_index(name="n_bases")
)

conflicts = conflicts[
    conflicts["n_bases"] > 1
]


if len(conflicts) > 0:

    print(
        "\nERROR: Some reconstructed nodes have more than "
        "one derived base at the same genomic position."
    )

    print(
        conflicts.head(30).to_string(index=False)
    )

    conflict_details = df.merge(
        conflicts[
            ["population", "node", "position"]
        ],
        on=["population", "node", "position"],
        how="inner"
    )

    conflict_details.to_csv(
        "LTEE_internal_node_multiallelic_conflicts.csv",
        index=False
    )

    print(
        "\nSaved conflict details to:"
        "\nLTEE_internal_node_multiallelic_conflicts.csv"
    )

    raise SystemExit(
        "\nResolve these conflicts before building FASTAs."
    )


print(
    "\nMulti-allelic conflict check: PASS"
)


# ============================================================
# Build one FASTA per internal node
# ============================================================

summary_rows = []


for (population, node), group in df.groupby(
    ["population", "node"]
):

    seq = list(reference_seq)

    positions_seen = set()

    for row in group.itertuples():

        position = int(row.position)
        new_base = str(row.new_base).upper()

        # GenomeDiff positions are 1-based
        index = position - 1

        if index < 0 or index >= len(seq):

            raise ValueError(
                f"Position out of range: "
                f"{population} {node} {position}"
            )

        if new_base not in {"A", "C", "G", "T"}:

            raise ValueError(
                f"Unexpected nucleotide: "
                f"{population} {node} "
                f"{position} -> {new_base}"
            )

        seq[index] = new_base
        positions_seen.add(position)


    reconstructed_seq = "".join(seq)

    safe_population = (
        population
        .replace("+", "plus")
        .replace("-", "minus")
    )

    output_file = (
        OUT_DIR
        / f"{safe_population}__{node}.fasta"
    )

    output_record = SeqRecord(
        Seq(reconstructed_seq),
        id=node,
        description=(
            f"population={population} "
            f"reconstructed_internal_ancestor "
            f"snp_count={len(positions_seen)} "
            f"reference=REL606"
        )
    )

    SeqIO.write(
        output_record,
        output_file,
        "fasta"
    )

    summary_rows.append({
        "population": population,
        "node": node,
        "snp_count": len(positions_seen),
        "sequence_length": len(reconstructed_seq),
        "fasta": str(output_file)
    })


# ============================================================
# Important:
# nodes with ZERO reconstructed SNPs do not appear in SNP CSV.
#
# So find them from branch table and create reference-sequence
# FASTAs for those nodes too.
# ============================================================

branch_file = Path(
    "LTEE_phylogenetic_branch_SNP_changes.csv"
)

branches = pd.read_csv(branch_file)


all_internal_nodes = set()

for name in pd.concat(
    [
        branches["parent"],
        branches["child"]
    ]
).dropna().astype(str):

    if "_internal_" in name:
        all_internal_nodes.add(name)


created_nodes = {
    row["node"]
    for row in summary_rows
}


zero_snp_nodes = sorted(
    all_internal_nodes - created_nodes
)


for node in zero_snp_nodes:

    # population is everything before "_internal_"
    population = node.split("_internal_")[0]

    safe_population = (
        population
        .replace("+", "plus")
        .replace("-", "minus")
    )

    output_file = (
        OUT_DIR
        / f"{safe_population}__{node}.fasta"
    )

    output_record = SeqRecord(
        Seq(reference_seq),
        id=node,
        description=(
            f"population={population} "
            f"reconstructed_internal_ancestor "
            f"snp_count=0 "
            f"reference=REL606"
        )
    )

    SeqIO.write(
        output_record,
        output_file,
        "fasta"
    )

    summary_rows.append({
        "population": population,
        "node": node,
        "snp_count": 0,
        "sequence_length": len(reference_seq),
        "fasta": str(output_file)
    })


# ============================================================
# Save summary
# ============================================================

summary = pd.DataFrame(summary_rows)

summary = summary.sort_values(
    ["population", "node"]
)

summary.to_csv(
    "LTEE_internal_node_fasta_summary.csv",
    index=False
)


print("\n========================================")
print("DONE")
print("========================================")

print(
    "Internal-node FASTAs created:",
    len(summary)
)

print(
    "Zero-SNP internal nodes:",
    len(zero_snp_nodes)
)

print(
    "Sequence length:",
    len(reference_seq)
)

print(
    "\nSNP count per reconstructed node:"
)

print(
    summary["snp_count"].describe()
)

print(
    "\nInternal nodes per population:"
)

print(
    summary.groupby("population").size()
)

print(
    "\nOutput directory:",
    OUT_DIR
)

print(
    "Summary:",
    "LTEE_internal_node_fasta_summary.csv"
)
