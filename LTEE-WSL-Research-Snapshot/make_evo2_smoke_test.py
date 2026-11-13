from pathlib import Path
import pandas as pd


WINDOW_SIZE = 8192

MODEL_READY = Path("LTEE_model_ready/train.csv")
OUT = Path("evo2_smoke_test")

OUT.mkdir(exist_ok=True)


def read_fasta(path):
    seq = []

    with open(path) as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith(">"):
                continue

            seq.append(line.upper())

    return "".join(seq)


def write_fasta(path, name, seq):

    with open(path, "w") as f:

        f.write(f">{name}\n")

        for i in range(0, len(seq), 80):
            f.write(seq[i:i+80] + "\n")


# ============================================================
# Load branch dataset
# ============================================================

df = pd.read_csv(MODEL_READY)


# Prefer:
# 1 mutation
# observed child
# high-confidence parent dating
candidate = df[
    (df["actual_mutation_count"] == 1)
    &
    (df["child_time_source"] == "observed")
    &
    (df["parent_confidence"] == "high")
].copy()


if len(candidate) == 0:
    candidate = df[
        df["actual_mutation_count"] == 1
    ].copy()


if len(candidate) == 0:
    raise RuntimeError(
        "No single-mutation branch found."
    )


row = candidate.iloc[0]


print("\nSelected branch:")
print(" branch_id:", row["branch_id"])
print(" population:", row["population"])
print(" parent:", row["parent"])
print(" child:", row["child"])
print(" delta_time:", row["delta_time"])
print(" mutation:", row["nucleotide_transitions"])


# ============================================================
# Read full genomes
# ============================================================

parent = read_fasta(
    row["parent_sequence_file"]
)

child = read_fasta(
    row["child_sequence_file"]
)


if len(parent) != len(child):
    raise RuntimeError(
        "Genome lengths do not match."
    )


# ============================================================
# Find mutation directly
# ============================================================

mutations = []

for i, (p, c) in enumerate(
    zip(parent, child),
    start=1
):
    if p != c:
        mutations.append(
            (i, p, c)
        )


if len(mutations) != 1:

    raise RuntimeError(
        f"Expected 1 mutation, found {len(mutations)}"
    )


position, parent_base, child_base = mutations[0]

idx = position - 1


# ============================================================
# 8192-bp mutation-centered window
# ============================================================

half = WINDOW_SIZE // 2

start = idx - half
end = start + WINDOW_SIZE


if start < 0:
    start = 0
    end = WINDOW_SIZE


if end > len(parent):
    end = len(parent)
    start = end - WINDOW_SIZE


parent_window = parent[start:end]
child_window = child[start:end]

mutation_index = idx - start


assert len(parent_window) == WINDOW_SIZE
assert len(child_window) == WINDOW_SIZE

assert (
    parent_window[mutation_index]
    == parent_base
)

assert (
    child_window[mutation_index]
    == child_base
)


# ============================================================
# Save
# ============================================================

write_fasta(
    OUT / "parent.fasta",
    row["parent"],
    parent_window
)

write_fasta(
    OUT / "child.fasta",
    row["child"],
    child_window
)


metadata = pd.DataFrame(
    [{
        "branch_id":
            row["branch_id"],

        "population":
            row["population"],

        "parent":
            row["parent"],

        "child":
            row["child"],

        "parent_time":
            row["parent_time"],

        "child_time":
            row["child_time"],

        "delta_time":
            row["delta_time"],

        "mutation_position_genome":
            position,

        "mutation_index_window_0based":
            mutation_index,

        "mutation_position_window_1based":
            mutation_index + 1,

        "parent_base":
            parent_base,

        "child_base":
            child_base,

        "window_genomic_start_1based":
            start + 1,

        "window_genomic_end_1based":
            end,

        "window_size":
            WINDOW_SIZE
    }]
)


metadata.to_csv(
    OUT / "metadata.csv",
    index=False
)


print("\n========================================")
print("EVO2 SMOKE TEST PACKAGE")
print("========================================")

print("\nGenome length:", len(parent))

print(
    "Mutation:",
    f"{position}:{parent_base}>{child_base}"
)

print(
    "Window:",
    f"{start + 1}-{end}"
)

print(
    "Mutation window index:",
    mutation_index
)

print(
    "Parent center:",
    parent_window[mutation_index]
)

print(
    "Child center:",
    child_window[mutation_index]
)

print("\nSaved:")
print(OUT / "parent.fasta")
print(OUT / "child.fasta")
print(OUT / "metadata.csv")
