from pathlib import Path
import pandas as pd
import torch
from evo2 import Evo2


def read_fasta(path):
    seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith(">"):
                seq.append(line.upper())
    return "".join(seq)


DATA = Path("evo2_smoke_test")

parent = read_fasta(DATA / "parent.fasta")
child = read_fasta(DATA / "child.fasta")
meta = pd.read_csv(DATA / "metadata.csv").iloc[0]

mutation_idx = int(
    meta["mutation_index_window_0based"]
)

print("Mutation index:", mutation_idx)
print("Parent:", parent[mutation_idx])
print("Child:", child[mutation_idx])


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

model = Evo2("evo2_7b_base")

tokens = model.tokenizer.tokenize(parent)

input_ids = torch.tensor(
    tokens,
    dtype=torch.long,
    device="cuda"
).unsqueeze(0)


with torch.inference_mode():
    output = model(input_ids)


logits = output[0][0]

print("\nLogits shape:", logits.shape)


# ------------------------------------------------------------
# Verify autoregressive alignment
#
# logits[:, i, :] should predict token i+1
# ------------------------------------------------------------

DNA = ["A", "C", "G", "T"]

token_ids = {
    base: int(model.tokenizer.tokenize(base)[0])
    for base in DNA
}

print("\nDNA token IDs:")
print(token_ids)


def dna_probabilities(logit_vector):

    dna_logits = torch.stack([
        logit_vector[token_ids[b]]
        for b in DNA
    ])

    # Conditional probability among A/C/G/T
    probs = torch.softmax(
        dna_logits.float(),
        dim=0
    )

    return {
        b: float(p)
        for b, p in zip(DNA, probs)
    }


# ------------------------------------------------------------
# Check alignment using positions immediately before mutation
# ------------------------------------------------------------

print("\n=== ALIGNMENT CHECK ===")

for target_idx in range(
    mutation_idx - 3,
    mutation_idx + 1
):

    # autoregressive prediction for target_idx
    prediction_idx = target_idx - 1

    true_base = parent[target_idx]

    probs = dna_probabilities(
        logits[
            0,
            prediction_idx,
            :
        ]
    )

    predicted = max(
        probs,
        key=probs.get
    )

    print(
        f"target index {target_idx}: "
        f"true={true_base} "
        f"predicted={predicted} "
        f"P(true)={probs[true_base]:.6f}"
    )


# ------------------------------------------------------------
# Mutation position
# ------------------------------------------------------------

prediction_idx = mutation_idx - 1

mutation_logits = logits[
    0,
    prediction_idx,
    :
]


probs = dna_probabilities(
    mutation_logits
)


print("\n========================================")
print("REAL LTEE MUTATION SITE")
print("========================================")

print(
    "Genome position:",
    meta["mutation_position_genome"]
)

print(
    "Window index:",
    mutation_idx
)

print(
    "Parent base:",
    parent[mutation_idx]
)

print(
    "Observed future base:",
    child[mutation_idx]
)

print(
    "Context ends with:",
    parent[
        mutation_idx-20:
        mutation_idx
    ]
)


print("\nEvo2 A/C/G/T probabilities:")

for base in DNA:

    marker = ""

    if base == parent[mutation_idx]:
        marker += "  <- parent"

    if base == child[mutation_idx]:
        marker += "  <- observed mutation"

    print(
        f"{base}: "
        f"{probs[base]:.8f}"
        f"{marker}"
    )


ranked = sorted(
    probs.items(),
    key=lambda x: x[1],
    reverse=True
)


print("\nRanking:")

for rank, (base, prob) in enumerate(
    ranked,
    start=1
):

    print(
        rank,
        base,
        f"{prob:.8f}"
    )


# ------------------------------------------------------------
# Mutation preference relative to parent
# ------------------------------------------------------------

parent_base = parent[mutation_idx]
future_base = child[mutation_idx]

ratio = (
    probs[future_base]
    /
    max(
        probs[parent_base],
        1e-30
    )
)


print(
    "\nP(observed future base) / P(parent base):",
    ratio
)

print("\nDONE")
