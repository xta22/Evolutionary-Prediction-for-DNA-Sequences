from pathlib import Path
import sys

import pandas as pd
import torch
from evo2 import Evo2


# ============================================================
# Configuration
# ============================================================

DATA_ROOT = Path("LTEE_model_ready")
TRAIN_FILE = DATA_ROOT / "train.csv"

WINDOW_SIZE = 8192


# ============================================================
# FASTA reader
# ============================================================

def read_fasta(path):
    seq = []

    with open(path) as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith(">"):
                continue

            seq.append(line.upper())

    return "".join(seq)


# ============================================================
# Check inputs
# ============================================================

if not TRAIN_FILE.exists():

    print(f"ERROR: cannot find {TRAIN_FILE}")
    print()
    print("The LTEE model-ready dataset needs to be copied")
    print("to this Quartz working directory first.")

    sys.exit(1)


# ============================================================
# Load training manifest
# ============================================================

df = pd.read_csv(TRAIN_FILE)

# Only branches containing at least one mutation
df = df[
    df["actual_mutation_count"] > 0
].copy()

if len(df) == 0:
    raise RuntimeError(
        "No mutated branches found."
    )


# ============================================================
# Select a simple branch
#
# Prefer a branch with exactly one mutation so the first
# experiment is easy to interpret.
# ============================================================

single = df[
    df["actual_mutation_count"] == 1
]

if len(single) > 0:
    row = single.iloc[0]
else:
    row = df.iloc[0]


print("\n========================================")
print("LTEE -> EVO2 SMOKE TEST")
print("========================================")

print("\nBranch:")
print("  branch_id:", row["branch_id"])
print("  population:", row["population"])
print("  parent:", row["parent"])
print("  child:", row["child"])

print("\nTime:")
print("  parent_time:", row["parent_time"])
print("  child_time:", row["child_time"])
print("  delta_time:", row["delta_time"])

print("\nMutations:")
print("  count:", row["actual_mutation_count"])
print(
    "  transitions:",
    row["nucleotide_transitions"]
)


# ============================================================
# Load sequences
# ============================================================

parent_path = Path(
    row["parent_sequence_file"]
)

child_path = Path(
    row["child_sequence_file"]
)


if not parent_path.exists():
    raise FileNotFoundError(
        f"Parent sequence missing: {parent_path}"
    )

if not child_path.exists():
    raise FileNotFoundError(
        f"Child sequence missing: {child_path}"
    )


parent = read_fasta(parent_path)
child = read_fasta(child_path)


print("\nGenome length:")
print("  parent:", len(parent))
print("  child:", len(child))


if len(parent) != len(child):
    raise RuntimeError(
        "Parent/child genome lengths differ."
    )


# ============================================================
# Find actual mutations directly from sequences
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


print("\nSequence-derived mutations:")

for pos, p, c in mutations[:20]:

    print(
        f"  {pos}: {p}>{c}"
    )


if len(mutations) == 0:
    raise RuntimeError(
        "Selected branch contains no sequence differences."
    )


# ============================================================
# Use first mutation for smoke test
# ============================================================

position, parent_base, child_base = (
    mutations[0]
)

# Convert 1-based genomic position to Python 0-based index
idx = position - 1


# ============================================================
# Construct 8192 bp window
#
# Put mutation approximately in the middle.
# ============================================================

half = WINDOW_SIZE // 2

start = idx - half
end = start + WINDOW_SIZE


# Handle chromosome boundaries
if start < 0:
    start = 0
    end = WINDOW_SIZE

if end > len(parent):
    end = len(parent)
    start = end - WINDOW_SIZE


parent_window = parent[start:end]
child_window = child[start:end]

mutation_index = idx - start


print("\nWindow:")
print("  genomic start (1-based):", start + 1)
print("  genomic end:", end)
print("  length:", len(parent_window))

print(
    "  mutation index in window:",
    mutation_index
)

print(
    "  parent base:",
    parent_window[
        mutation_index
    ]
)

print(
    "  child base:",
    child_window[
        mutation_index
    ]
)


assert len(parent_window) == WINDOW_SIZE
assert len(child_window) == WINDOW_SIZE

assert (
    parent_window[
        mutation_index
    ]
    ==
    parent_base
)

assert (
    child_window[
        mutation_index
    ]
    ==
    child_base
)


# ============================================================
# GPU information
# ============================================================

print("\nGPU:")

print(
    "  CUDA available:",
    torch.cuda.is_available()
)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required."
    )

print(
    "  device:",
    torch.cuda.get_device_name(0)
)


# ============================================================
# Load Evo2
# ============================================================

print("\nLoading Evo2 7B base...")

model = Evo2(
    "evo2_7b_base"
)

print(
    "Model loaded."
)

print(
    "GPU allocated:",
    round(
        torch.cuda.memory_allocated()
        /
        1024**3,
        2
    ),
    "GB"
)


# ============================================================
# Tokenize DNA
# ============================================================

print("\nTokenizing parent window...")

input_ids = (
    torch.tensor(
        model.tokenizer.tokenize(
            parent_window
        ),
        dtype=torch.long
    )
    .unsqueeze(0)
    .cuda()
)


print(
    "Token tensor shape:",
    tuple(
        input_ids.shape
    )
)


# ============================================================
# Forward pass
# ============================================================

print("\nRunning Evo2 forward pass...")

with torch.no_grad():

    output = model(
        input_ids
    )


# Different Evo2/Vortex versions may expose the output
# differently, so first inspect rather than assuming.

print("\n========================================")
print("EVO2 OUTPUT")
print("========================================")

print(
    "Output type:",
    type(output)
)


if isinstance(output, torch.Tensor):

    print(
        "Tensor shape:",
        tuple(
            output.shape
        )
    )

elif isinstance(output, tuple):

    print(
        "Tuple length:",
        len(output)
    )

    for i, item in enumerate(
        output
    ):

        if isinstance(
            item,
            torch.Tensor
        ):

            print(
                f"  [{i}] tensor:",
                tuple(
                    item.shape
                )
            )

        else:

            print(
                f"  [{i}]:",
                type(item)
            )

elif isinstance(output, dict):

    print(
        "Keys:",
        list(
            output.keys()
        )
    )

    for key, value in output.items():

        if isinstance(
            value,
            torch.Tensor
        ):

            print(
                key,
                tuple(
                    value.shape
                )
            )

        else:

            print(
                key,
                type(value)
            )


# ============================================================
# Final
# ============================================================

print("\n========================================")
print("SMOKE TEST COMPLETE")
print("========================================")

print(
    "\nMutation tested:"
)

print(
    f"{position}: "
    f"{parent_base}>{child_base}"
)

print(
    "\nIf the forward pass completed, "
    "LTEE sequence -> Evo2 is working."
)
