import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path("LTEE_temporal_head_v1")
manifest = pd.read_csv("LTEE_model_ready/model_manifest.csv")
manifest = manifest.set_index("branch_id")

GENOME_LENGTH = 4629812

for split in ["train", "validation", "test"]:
    input_path = ROOT / f"{split}.jsonl"
    output_path = ROOT / f"{split}_weights.csv"

    records = [
        json.loads(line)
        for line in input_path.open()
        if line.strip()
    ]

    counts = Counter(
        (r["branch_id"], r["is_mutated"])
        for r in records
    )

    output = []

    for r in records:
        branch = r["branch_id"]
        label = int(r["is_mutated"])

        row = manifest.loc[branch]

        if row["split"] != split:
            raise ValueError(f"Split mismatch: {branch}")

        M = int(row["actual_mutation_count"])
        m = counts[(branch, 1)]
        u = counts[(branch, 0)]

        if label == 1:
            if m == 0:
                raise ValueError(f"No sampled mutations: {branch}")
            weight = M / m
        else:
            if u == 0:
                raise ValueError(f"No sampled controls: {branch}")
            weight = (GENOME_LENGTH - M) / u

        output.append({
            "sample_id": r["sample_id"],
            "branch_id": branch,
            "split": split,
            "is_mutated": label,
            "sampling_weight": weight
        })

    pd.DataFrame(output).to_csv(output_path, index=False)

    print(
        split,
        "samples:", len(output),
        "saved:", output_path
    )
