import json
import random
from pathlib import Path
from collections import Counter

import pandas as pd

# ============================================================
# Configuration
# ============================================================

SEED = 42
WINDOW = 2048
MAX_MUTATIONS = 10
CONTROLS_PER_MUTATION = 3
ZERO_MUTATION_CONTROLS = 10

ROOT = Path.cwd()
DATA = ROOT / "LTEE_model_ready"
OUT = ROOT / "LTEE_temporal_head_v1"
OUT.mkdir(exist_ok=True)

DNA = set("ACGT")
HALF = WINDOW // 2


def read_fasta(path):
    path = Path(str(path))

    if not path.exists():
        path = ROOT / path

    if not path.exists():
        raise FileNotFoundError(path)

    with open(path) as f:
        return "".join(
            line.strip().upper()
            for line in f
            if line.strip() and not line.startswith(">")
        )


def valid_window(seq, pos):
    start = pos - HALF
    end = start + WINDOW

    if start < 0 or end > len(seq):
        return None

    window = seq[start:end]

    if any(base not in DNA for base in window):
        return None

    return window


def select_controls(parent, child, mutation_set, n, rng):
    """
    Random unchanged positions.
    Avoid scanning a huge candidate list unnecessarily.
    """
    selected = set()
    attempts = 0
    max_attempts = max(1000, n * 100)

    while len(selected) < n and attempts < max_attempts:
        attempts += 1

        pos = rng.randrange(
            HALF,
            len(parent) - HALF
        )

        if pos in selected or pos in mutation_set:
            continue

        if parent[pos] != child[pos]:
            continue

        if valid_window(parent, pos) is None:
            continue

        selected.add(pos)

    if len(selected) < n:
        print(
            f"WARNING: requested {n} controls, "
            f"found {len(selected)}"
        )

    return sorted(selected)


def process_split(split):
    df = pd.read_csv(DATA / f"{split}.csv")
    rng = random.Random(SEED + {
        "train": 0,
        "validation": 1,
        "test": 2
    }[split])

    records = []
    stats = Counter()

    for _, row in df.iterrows():
        parent = read_fasta(row["parent_sequence_file"])
        child = read_fasta(row["child_sequence_file"])

        if len(parent) != len(child):
            raise ValueError(
                f"Genome length mismatch: {row['branch_id']}"
            )

        mutation_positions = [
            i for i, (a, b) in enumerate(zip(parent, child))
            if a != b
        ]

        if len(mutation_positions) != int(row["actual_mutation_count"]):
            raise ValueError(
                f"Mutation count mismatch: {row['branch_id']}"
            )

        mutation_set = set(mutation_positions)

        valid_mutations = [
            i for i in mutation_positions
            if parent[i] in DNA
            and child[i] in DNA
            and valid_window(parent, i) is not None
        ]

        rng.shuffle(valid_mutations)
        selected_mutations = valid_mutations[:MAX_MUTATIONS]

        n_controls = (
            len(selected_mutations) * CONTROLS_PER_MUTATION
            if selected_mutations
            else ZERO_MUTATION_CONTROLS
        )

        controls = select_controls(
            parent, child, mutation_set, n_controls, rng
        )

        sites = (
            [(i, 1) for i in selected_mutations]
            + [(i, 0) for i in controls]
        )

        for pos, label in sites:
            window = valid_window(parent, pos)

            if window is None:
                continue

            record = {
                "sample_id": (
                    f"{row['branch_id']}_pos_{pos + 1}"
                ),
                "split": split,
                "branch_id": str(row["branch_id"]),
                "population": str(row["population"]),
                "parent": str(row["parent"]),
                "child": str(row["child"]),
                "delta_time": float(row["delta_time"]),
                "position_1based": pos + 1,
                "target_index": HALF,
                "parent_base": parent[pos],
                "future_base": child[pos],
                "is_mutated": label,
                "sequence": window,
            }

            # Preserve available dating-quality metadata
            for col in [
                "temporal_quality",
                "parent_confidence",
                "child_confidence",
                "parent_time_source",
                "child_time_source",
            ]:
                if col in row and pd.notna(row[col]):
                    record[col] = str(row[col])

            records.append(record)

            stats["mutated" if label else "unchanged"] += 1

        stats["branches"] += 1

        if not selected_mutations:
            stats["branches_without_selected_mutations"] += 1

    path = OUT / f"{split}.jsonl"

    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"\n=== {split.upper()} ===")
    print("Branches:", stats["branches"])
    print("Mutation samples:", stats["mutated"])
    print("Unchanged samples:", stats["unchanged"])
    print(
        "Branches without selected mutations:",
        stats["branches_without_selected_mutations"]
    )
    print("Total samples:", len(records))
    print("Populations:", len(set(r["population"] for r in records)))
    print("Saved:", path)

    return records


all_records = []

for split in ["train", "validation", "test"]:
    all_records.extend(process_split(split))

# ============================================================
# Cross-split identity checks
# ============================================================

seen = {}

for record in all_records:
    sid = record["sample_id"]

    if sid in seen and seen[sid] != record["split"]:
        raise RuntimeError(
            f"Sample appears in multiple splits: {sid}"
        )

    seen[sid] = record["split"]

print("\n========================================")
print("TEMPORAL HEAD DATASET V1")
print("========================================")

print("Total samples:", len(all_records))
print("Window length:", WINDOW)
print("Target index:", HALF)
print("Mutated:", sum(r["is_mutated"] for r in all_records))
print("Unchanged:", sum(not r["is_mutated"] for r in all_records))

print("\nDataset ready.")
