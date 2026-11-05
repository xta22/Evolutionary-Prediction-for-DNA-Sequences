from pathlib import Path
import pandas as pd
import torch
from evo2 import Evo2


# ============================================================
# Read FASTA
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
# Load LTEE smoke-test data
# ============================================================

DATA = Path("evo2_smoke_test")

parent = read_fasta(DATA / "parent.fasta")
child = read_fasta(DATA / "child.fasta")

meta = pd.read_csv(
    DATA / "metadata.csv"
).iloc[0]

mutation_idx = int(
    meta["mutation_index_window_0based"]
)

print("\n========================================")
print("LTEE -> EVO2 FORWARD TEST")
print("========================================")

print("\nSequence length:", len(parent))
print("Delta time:", meta["delta_time"])

print(
    "Mutation:",
    f"{meta['parent_base']}>{meta['child_base']}"
)

print(
    "Mutation index:",
    mutation_idx
)

print(
    "Parent base:",
    parent[mutation_idx]
)

print(
    "Child base:",
    child[mutation_idx]
)

assert len(parent) == 8192
assert len(child) == 8192
assert parent[mutation_idx] == meta["parent_base"]
assert child[mutation_idx] == meta["child_base"]


# ============================================================
# GPU
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available")

print("\nGPU:", torch.cuda.get_device_name(0))


# ============================================================
# Load Evo2
# ============================================================

print("\nLoading evo2_7b_base...")

model = Evo2("evo2_7b_base")

print("Model loaded")

print(
    "GPU memory:",
    round(
        torch.cuda.memory_allocated() / 1024**3,
        2
    ),
    "GB"
)


# ============================================================
# Tokenize
# ============================================================

tokens = model.tokenizer.tokenize(parent)

print("\nToken count:", len(tokens))

print(
    "First 20 token IDs:",
    tokens[:20]
)

input_ids = torch.tensor(
    tokens,
    dtype=torch.long,
    device="cuda"
).unsqueeze(0)

print(
    "Input tensor:",
    tuple(input_ids.shape)
)


# ============================================================
# Forward
# ============================================================

print("\nRunning forward pass...")

with torch.inference_mode():
    output = model(input_ids)


print("\n========================================")
print("OUTPUT STRUCTURE")
print("========================================")

print("type:", type(output))


def describe(name, obj):

    if isinstance(obj, torch.Tensor):

        print(
            name,
            "Tensor",
            tuple(obj.shape),
            obj.dtype,
            obj.device
        )

    else:

        print(
            name,
            type(obj)
        )


if isinstance(output, torch.Tensor):

    describe("output", output)

elif isinstance(output, tuple):

    print("tuple length:", len(output))

    for i, x in enumerate(output):
        describe(f"output[{i}]", x)

elif isinstance(output, dict):

    print("keys:", list(output.keys()))

    for key, value in output.items():
        describe(key, value)

else:

    # Inspect public attributes without dumping huge tensors
    attrs = [
        x for x in dir(output)
        if not x.startswith("_")
    ]

    print(
        "public attributes:",
        attrs[:50]
    )


print("\nPeak GPU memory:")

print(
    round(
        torch.cuda.max_memory_allocated()
        / 1024**3,
        2
    ),
    "GB"
)

print("\n========================================")
print("FORWARD PASS COMPLETE")
print("========================================")
