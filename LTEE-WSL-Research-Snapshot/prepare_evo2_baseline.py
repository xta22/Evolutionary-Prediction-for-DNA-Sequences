import json
import random
from pathlib import Path

import pandas as pd

random.seed(42)

ROOT = Path.cwd()
df = pd.read_csv("LTEE_model_ready/train.csv")

OUT = Path("evo2_baseline_pilot.jsonl")
WINDOW = 2048
MAX_BRANCHES = 20


def read_fasta(path):
    with open(path) as f:
        return "".join(
            line.strip().upper()
            for line in f
            if not line.startswith(">")
        )


def resolve_path(path):
    p = Path(path)
    if p.exists():
        return p
    p = ROOT / path
    if p.exists():
        return p
    raise FileNotFoundError(path)


def window_at(seq, pos):
    start = pos - WINDOW // 2
    if start < 0 or start + WINDOW > len(seq):
        return None
    return seq[start:start + WINDOW]


records = []
used_branches = 0

for _, row in df.iterrows():

    if used_branches >= MAX_BRANCHES:
        break

    if row["actual_mutation_count"] < 1:
        continue

    parent = read_fasta(
        resolve_path(row["parent_sequence_file"])
    )
    child = read_fasta(
        resolve_path(row["child_sequence_file"])
    )

    mutations = [
        i for i, (a, b) in enumerate(zip(parent, child))
        if a != b
    ]

    if not mutations:
        continue

    # Select one mutation with a valid window
    valid = [
        i for i in mutations
        if window_at(parent, i) is not None
        and parent[i] in "ACGT"
        and child[i] in "ACGT"
    ]

    if not valid:
        continue

    pos = random.choice(valid)
    base = parent[pos]

    # Match control to the same parent nucleotide
    mutation_set = set(mutations)

    candidates = [
        i for i in range(WINDOW // 2, len(parent) - WINDOW // 2)
        if parent[i] == base
        and i not in mutation_set
        and abs(i - pos) > 100
    ]

    if not candidates:
        continue

    control = random.choice(candidates)

    for site, label in [(pos, 1), (control, 0)]:

        records.append({
            "branch_id": row["branch_id"],
            "population": row["population"],
            "delta_time": float(row["delta_time"]),
            "position_1based": site + 1,
            "is_mutated": label,
            "parent_base": parent[site],
            "child_base": child[site],
            "target_index": WINDOW // 2,
            "sequence": window_at(parent, site),
        })

    used_branches += 1

with open(OUT, "w") as f:
    for record in records:
        f.write(json.dumps(record) + "\n")

print("Branches:", used_branches)
print("Windows:", len(records))
print("Saved:", OUT)
