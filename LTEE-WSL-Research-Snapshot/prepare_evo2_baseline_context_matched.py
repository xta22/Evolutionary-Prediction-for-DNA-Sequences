import json
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd

random.seed(42)

df = pd.read_csv("LTEE_model_ready/train.csv")

OUT = Path("evo2_baseline_context_matched.jsonl")

WINDOW = 2048
K = 5
HALF_K = K // 2

MAX_MUTATIONS_PER_BRANCH = 10
CONTROLS_PER_MUTATION = 3

DNA = set("ACGT")


def read_fasta(path):
    with open(path) as f:
        return "".join(
            line.strip().upper()
            for line in f
            if not line.startswith(">")
        )


def window_at(seq, pos):
    start = pos - WINDOW // 2
    end = start + WINDOW

    if start < 0 or end > len(seq):
        return None

    return seq[start:end]


def kmer_at(seq, pos):
    start = pos - HALF_K
    end = pos + HALF_K + 1

    if start < 0 or end > len(seq):
        return None

    kmer = seq[start:end]

    if any(b not in DNA for b in kmer):
        return None

    return kmer


records = []
summary = defaultdict(
    lambda: {
        "branches": set(),
        "mutations": 0,
        "controls": 0
    }
)

for _, row in df.iterrows():

    parent = read_fasta(row["parent_sequence_file"])
    child = read_fasta(row["child_sequence_file"])

    mutations = [
        i for i, (a, b) in enumerate(zip(parent, child))
        if a != b and a in DNA and b in DNA
    ]

    if not mutations:
        continue

    mutation_set = set(mutations)

    # Index all valid unchanged 5-mer positions
    control_index = defaultdict(list)

    for i in range(WINDOW // 2, len(parent) - WINDOW // 2):

        if i in mutation_set:
            continue

        kmer = kmer_at(parent, i)

        if kmer is not None:
            control_index[kmer].append(i)

    random.shuffle(mutations)

    selected = 0

    for pos in mutations:

        if selected >= MAX_MUTATIONS_PER_BRANCH:
            break

        parent_window = window_at(parent, pos)
        kmer = kmer_at(parent, pos)

        if parent_window is None or kmer is None:
            continue

        candidates = [
            i for i in control_index.get(kmer, [])
            if abs(i - pos) > WINDOW
        ]

        if len(candidates) < CONTROLS_PER_MUTATION:
            continue

        controls = random.sample(
            candidates,
            CONTROLS_PER_MUTATION
        )

        sites = [(pos, 1)] + [
            (c, 0) for c in controls
        ]

        for site, label in sites:

            records.append({
                "branch_id": row["branch_id"],
                "population": row["population"],
                "delta_time": float(row["delta_time"]),
                "position_1based": site + 1,
                "is_mutated": label,
                "parent_base": parent[site],
                "child_base": child[site],
                "target_index": WINDOW // 2,
                "matched_kmer": kmer,
                "matched_mutation_position": pos + 1,
                "sequence": window_at(parent, site)
            })

            if label == 1:
                summary[row["population"]]["mutations"] += 1
            else:
                summary[row["population"]]["controls"] += 1

        summary[row["population"]]["branches"].add(
            row["branch_id"]
        )

        selected += 1

with open(OUT, "w") as f:
    for record in records:
        f.write(json.dumps(record) + "\n")

print("\n=== CONTEXT-MATCHED DATASET ===")

for pop in sorted(summary):
    s = summary[pop]
    print(
        pop,
        "branches:", len(s["branches"]),
        "mutations:", s["mutations"],
        "controls:", s["controls"]
    )

print("\nTotal windows:", len(records))
print("Saved:", OUT)
