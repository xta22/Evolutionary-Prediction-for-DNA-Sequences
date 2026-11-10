from pathlib import Path
import re

import numpy as np
import pandas as pd
from Bio import Phylo


TREE_FILE = Path("summary/LTEE_phylogenetic_trees/Ara-3.tre")

OUT_DIAGNOSTICS = "Ara-3_clade_diagnostics.csv"
OUT_GROUPS = "Ara-3_clock_groups.csv"
OUT_TIPS = "Ara-3_tip_clock_group_assignments.csv"


# ============================================================
# Settings
# ============================================================

MIN_TIPS = 4
MIN_DISTINCT_GENERATIONS = 3

# For now this is only a diagnostic threshold.
# We are NOT claiming 0.90 is a biological cutoff.
PEARSON_THRESHOLD = 0.90
SPEARMAN_THRESHOLD = 0.90


# ============================================================
# Helpers
# ============================================================

def get_generation(name):
    if name is None:
        return None

    m = re.search(r"_(\d+)gen_", name)

    if m:
        return int(m.group(1))

    return None


def name_internal_nodes(tree):
    """
    Use exactly the same preorder naming scheme as our
    previous reconstruction scripts.
    """

    counter = 0

    for node in tree.get_nonterminals(order="preorder"):

        if node.name is None:
            node.name = f"Ara-3_internal_{counter}"
            counter += 1


def get_tip_table(tree, node):
    """
    Get observed descendants of one internal node.

    Excludes generation-0 founder from clock fitting because
    we want to evaluate the relationship among evolved samples.
    """

    rows = []

    for tip in node.get_terminals():

        generation = get_generation(tip.name)

        if generation is None:
            continue

        if generation == 0:
            continue

        root_distance = tree.distance(
            tree.root,
            tip
        )

        rows.append({
            "clone": tip.name,
            "generation": generation,
            "root_distance": root_distance
        })

    return pd.DataFrame(rows)


def calculate_stats(tree, node):

    tips = get_tip_table(tree, node)

    if len(tips) == 0:

        return {
            "n_tips": 0,
            "n_distinct_generations": 0,
            "min_generation": np.nan,
            "max_generation": np.nan,
            "pearson_r": np.nan,
            "spearman_r": np.nan
        }

    n_generations = tips["generation"].nunique()

    if (
        len(tips) >= 2
        and n_generations >= 2
        and tips["root_distance"].nunique() >= 2
    ):

        pearson = tips[
            ["generation", "root_distance"]
        ].corr(method="pearson").iloc[0, 1]

        spearman = tips[
            ["generation", "root_distance"]
        ].corr(method="spearman").iloc[0, 1]

    else:

        pearson = np.nan
        spearman = np.nan

    return {
        "n_tips": len(tips),
        "n_distinct_generations": n_generations,
        "min_generation": tips["generation"].min(),
        "max_generation": tips["generation"].max(),
        "pearson_r": pearson,
        "spearman_r": spearman
    }


def is_good_clock_clade(stats):

    if stats["n_tips"] < MIN_TIPS:
        return False

    if (
        stats["n_distinct_generations"]
        < MIN_DISTINCT_GENERATIONS
    ):
        return False

    if pd.isna(stats["pearson_r"]):
        return False

    if pd.isna(stats["spearman_r"]):
        return False

    return (
        stats["pearson_r"] >= PEARSON_THRESHOLD
        and
        stats["spearman_r"] >= SPEARMAN_THRESHOLD
    )


# ============================================================
# Load tree
# ============================================================

tree = Phylo.read(
    TREE_FILE,
    "newick"
)

name_internal_nodes(tree)


# ============================================================
# Diagnostics for every internal clade
# ============================================================

diagnostic_rows = []

stats_lookup = {}


for node in tree.get_nonterminals(order="preorder"):

    stats = calculate_stats(
        tree,
        node
    )

    stats_lookup[node] = stats

    diagnostic_rows.append({
        "node":
            node.name,

        "node_root_distance":
            tree.distance(
                tree.root,
                node
            ),

        "n_tips":
            stats["n_tips"],

        "n_distinct_generations":
            stats["n_distinct_generations"],

        "min_generation":
            stats["min_generation"],

        "max_generation":
            stats["max_generation"],

        "pearson_r":
            stats["pearson_r"],

        "spearman_r":
            stats["spearman_r"],

        "passes_clock_screen":
            is_good_clock_clade(stats)
    })


diagnostics = pd.DataFrame(
    diagnostic_rows
)

diagnostics.to_csv(
    OUT_DIAGNOSTICS,
    index=False
)


# ============================================================
# Select top-level non-overlapping "good" clades
#
# Logic:
#
# root is poor for Ara-3
#     ↓
# recursively descend
#     ↓
# once a sufficiently large child clade has good correlation,
# keep that clade and stop splitting inside it.
#
# This produces candidate clock groups, NOT final biology.
# ============================================================

selected_nodes = []


def select_clock_groups(node):

    # Never use whole Ara-3 tree as one group,
    # since we already know population-wide r is poor.
    if node != tree.root:

        stats = stats_lookup.get(
            node,
            calculate_stats(tree, node)
        )

        if is_good_clock_clade(stats):

            selected_nodes.append(node)
            return

    # If node is not suitable, descend further
    for child in node.clades:

        if not child.is_terminal():
            select_clock_groups(child)


select_clock_groups(
    tree.root
)


# ============================================================
# Create candidate group table
# ============================================================

group_rows = []
tip_to_group = {}


for i, node in enumerate(
    selected_nodes,
    start=1
):

    group_id = f"Ara-3_clade_{i}"

    stats = stats_lookup[node]

    tips = get_tip_table(
        tree,
        node
    )

    for clone in tips["clone"]:
        tip_to_group[clone] = group_id

    group_rows.append({
        "clock_group":
            group_id,

        "clade_root":
            node.name,

        "clade_root_distance":
            tree.distance(
                tree.root,
                node
            ),

        "n_tips":
            stats["n_tips"],

        "n_distinct_generations":
            stats["n_distinct_generations"],

        "min_generation":
            stats["min_generation"],

        "max_generation":
            stats["max_generation"],

        "pearson_r":
            stats["pearson_r"],

        "spearman_r":
            stats["spearman_r"]
    })


groups = pd.DataFrame(
    group_rows
)

groups.to_csv(
    OUT_GROUPS,
    index=False
)


# ============================================================
# Tip assignments
# ============================================================

tip_rows = []


for tip in tree.get_terminals():

    generation = get_generation(
        tip.name
    )

    if generation is None:
        continue

    if generation == 0:
        continue

    tip_rows.append({
        "clone":
            tip.name,

        "generation":
            generation,

        "root_distance":
            tree.distance(
                tree.root,
                tip
            ),

        "clock_group":
            tip_to_group.get(
                tip.name,
                "UNRESOLVED"
            )
    })


tip_assignments = pd.DataFrame(
    tip_rows
)

tip_assignments.to_csv(
    OUT_TIPS,
    index=False
)


# ============================================================
# Summary
# ============================================================

print("\n========================================")
print("ARA-3 CLADE DIAGNOSTICS")
print("========================================")

whole = calculate_stats(
    tree,
    tree.root
)

print(
    "\nWhole Ara-3 population:"
)

print(
    f"  tips: {whole['n_tips']}"
)

print(
    f"  distinct generations: "
    f"{whole['n_distinct_generations']}"
)

print(
    f"  Pearson r: "
    f"{whole['pearson_r']:.3f}"
)

print(
    f"  Spearman r: "
    f"{whole['spearman_r']:.3f}"
)


print(
    "\n=== SELECTED CANDIDATE CLOCK CLADES ==="
)

if len(groups) == 0:

    print(
        "No clades passed the current thresholds."
    )

else:

    print(
        groups.to_string(
            index=False
        )
    )


print(
    "\n=== TIP ASSIGNMENT COUNTS ==="
)

print(
    tip_assignments[
        "clock_group"
    ].value_counts()
)


print(
    "\n=== LARGEST HIGH-CORRELATION INTERNAL CLADES ==="
)

good = diagnostics[
    diagnostics["passes_clock_screen"]
].copy()

good = good.sort_values(
    [
        "n_tips",
        "pearson_r"
    ],
    ascending=[
        False,
        False
    ]
)

print(
    good[
        [
            "node",
            "n_tips",
            "n_distinct_generations",
            "min_generation",
            "max_generation",
            "pearson_r",
            "spearman_r"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


print("\nSaved:")
print(" ", OUT_DIAGNOSTICS)
print(" ", OUT_GROUPS)
print(" ", OUT_TIPS)
