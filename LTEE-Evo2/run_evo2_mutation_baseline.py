import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from evo2 import Evo2

INPUT = Path("evo2_baseline_pilot.jsonl")
OUTPUT = Path("results/evo2_baseline_pilot.csv")

OUTPUT.parent.mkdir(exist_ok=True)

with open(INPUT) as f:
    samples = [json.loads(line) for line in f]

print("Samples:", len(samples), flush=True)

model = Evo2("evo2_7b_base")

DNA = "ACGT"
token_ids = {
    b: int(model.tokenizer.tokenize(b)[0])
    for b in DNA
}

results = []

for n, sample in enumerate(samples, 1):

    sequence = sample["sequence"]
    target = int(sample["target_index"])
    parent_base = sample["parent_base"]

    tokens = model.tokenizer.tokenize(sequence)

    input_ids = torch.tensor(
        tokens,
        dtype=torch.long,
        device="cuda"
    ).unsqueeze(0)

    with torch.inference_mode():
        output = model(input_ids)
        logits = output[0][0]

        # Autoregressive next-token prediction
        site_logits = logits[0, target - 1]

        dna_logits = torch.stack([
            site_logits[token_ids[b]]
            for b in DNA
        ]).float()

        probs = torch.softmax(dna_logits, dim=0)

    probabilities = {
        b: float(probs[i].item())
        for i, b in enumerate(DNA)
    }

    p_parent = probabilities[parent_base]

    result = {
        k: v for k, v in sample.items()
        if k != "sequence"
    }

    result.update({
        "p_parent": p_parent,
        "mutation_score": 1.0 - p_parent,
        **{
            f"p_{b}": probabilities[b]
            for b in DNA
        }
    })

    results.append(result)

    if n % 10 == 0:
        print(
            f"Processed {n}/{len(samples)}",
            flush=True
        )

    del output, logits, input_ids

pd.DataFrame(results).to_csv(OUTPUT, index=False)

print("Saved:", OUTPUT)

# Exploratory ranking metrics
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

result_df = pd.DataFrame(results)

y = result_df["is_mutated"]
scores = result_df["mutation_score"]

print("\n=== PILOT RESULTS ===")
print("AUROC:", roc_auc_score(y, scores))
print("AUPRC:", average_precision_score(y, scores))
print("Mean mutation score by label:")
print(
    result_df.groupby("is_mutated")[
        "mutation_score"
    ].mean()
)
