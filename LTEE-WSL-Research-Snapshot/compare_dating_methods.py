from pathlib import Path
from collections import defaultdict
import re

import numpy as np
import pandas as pd
from Bio import Phylo


TREE_DIR = Path("summary/LTEE_phylogenetic_trees")
NODE_FASTA_DIR = Path("LTEE_internal_node_fastas")

TREETIME_DIR = Path("dating_results/treetime")
LSD2_DIR = Path("dating_results/lsd2")

OUT = "LTEE_dating_comparison.csv"


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


def descendant_signature(node):
    """
    Unique representation of a clade based on terminal tip names.
    This lets us match internal nodes between different software
    even if the software uses different internal-node names.
    """

    tips = sorted(
        tip.name
        for tip in node.get_terminals()
    )

    return "|".join(tips)


def load_original_tree(population):
    return Phylo.read(
        TREE_DIR / f"{population}.tre",
        "newick"
    )


def build_original_node_map(tree):
    mapping = {}

    for node in tree.get_nonterminals():

        sig = descendant_signature(node)

        mapping[sig] = node

    return mapping


# ============================================================
# TreeTime
# ============================================================

def load_treetime_dates(population):

    candidates = list(
        (TREETIME_DIR / population).glob(
            "**/dates.tsv"
        )
    )

    if len(candidates) == 0:
        return {}

    # Prefer alignment-based relaxed result
    preferred = [
        x for x in candidates
        if "alignment_relaxed" in str(x)
    ]

    date_file = (
        preferred[0]
        if preferred
        else candidates[0]
    )

    print(
        "  TreeTime dates:",
        date_file
    )

    df = pd.read_csv(
        date_file,
        sep=r"\s+",
        comment="#",
        header=None
    )

    if df.shape[1] < 3:
        raise ValueError(
            f"Unexpected TreeTime format: {date_file}"
        )

    df = df.iloc[:, :3]
    df.columns = [
        "node",
        "date_text",
        "date"
    ]

    return {
        row["node"]: float(row["date"])
        for _, row in df.iterrows()
    }


def load_treetime_tree(population):

    base = TREETIME_DIR / population

    candidates = list(
        base.glob("**/timetree.nexus")
    )

    if not candidates:
        candidates = list(
            base.glob("**/timetree.nwk")
        )

    if not candidates:
        return None

    tree_file = candidates[0]

    print(
        "  TreeTime tree:",
        tree_file
    )

    fmt = (
        "nexus"
        if tree_file.suffix == ".nexus"
        else "newick"
    )

    return Phylo.read(
        tree_file,
        fmt
    )


# ============================================================
# LSD2
# ============================================================

def find_lsd2_tree(population):

    base = LSD2_DIR / population

    if not base.exists():
        return None

    candidates = []

    for pattern in [
        "**/*.date.nexus",
        "**/*.date.nwk",
        "**/*.nexus",
        "**/*.nwk",
    ]:
        candidates.extend(
            base.glob(pattern)
        )

    if not candidates:
        return None

    # Prefer date.nexus
    dated = [
        x for x in candidates
        if ".date." in x.name
    ]

    return (
        dated[0]
        if dated
        else candidates[0]
    )


def load_lsd2_tree(population):

    tree_file = find_lsd2_tree(
        population
    )

    if tree_file is None:
        return None

    print(
        "  LSD2 tree:",
        tree_file
    )

    if tree_file.suffix == ".nexus":
        return Phylo.read(
            tree_file,
            "nexus"
        )

    return Phylo.read(
        tree_file,
        "newick"
    )


# ============================================================
# Extract TreeTime dates attached to clades
# ============================================================

def treetime_clade_dates(
    population,
    original_tree,
    tt_tree,
    tt_dates
):

    if tt_tree is None:
        return {}

    original_map = build_original_node_map(
        original_tree
    )

    output = {}

    for node in tt_tree.get_nonterminals():

        sig = descendant_signature(
            node
        )

        if sig not in original_map:
            continue

        original_node = original_map[sig]

        if node.name not in tt_dates:
            continue

        output[
            original_node.name
        ] = tt_dates[node.name]

    return output


# ============================================================
# Main
# ============================================================

rows = []


for tree_file in sorted(
    TREE_DIR.glob("*.tre")
):

    population = tree_file.stem

    print(
        f"\nProcessing {population}"
    )

    tree = load_original_tree(
        population
    )

    # --------------------------------------------------------
    # Assign deterministic names exactly as before
    # --------------------------------------------------------

    counter = 0

    for node in tree.get_nonterminals(
        order="preorder"
    ):

        if node.name is None:
            node.name = (
                f"{population}_internal_{counter}"
            )

            counter += 1


    # --------------------------------------------------------
    # TreeTime
    # --------------------------------------------------------

    tt_dates = load_treetime_dates(
        population
    )

    tt_tree = load_treetime_tree(
        population
    )

    tt_clade_dates = (
        treetime_clade_dates(
            population,
            tree,
            tt_tree,
            tt_dates
        )
    )


    # --------------------------------------------------------
    # Our existing inferred dates
    # --------------------------------------------------------

    old_file = Path(
        "LTEE_internal_node_inferred_times.csv"
    )

    if old_file.exists():

        old_df = pd.read_csv(
            old_file
        )

        old_pop = old_df[
            old_df["population"]
            == population
        ].copy()

        old_dates = dict(
            zip(
                old_pop["node"],
                old_pop[
                    "estimated_generation"
                ]
            )
        )

        hard_upper = dict(
            zip(
                old_pop["node"],
                old_pop[
                    "hard_upper_generation"
                ]
            )
        )

    else:

        old_dates = {}
        hard_upper = {}


    # --------------------------------------------------------
    # LSD2
    #
    # We first try to read node dates from
    # the dated tree branch annotations.
    # If unavailable, leave as NA.
    # --------------------------------------------------------

    lsd_tree = load_lsd2_tree(
        population
    )

    lsd_dates = {}

    if lsd_tree is not None:

        lsd_original_map = (
            build_original_node_map(
                tree
            )
        )

        for node in lsd_tree.get_nonterminals():

            sig = descendant_signature(
                node
            )

            if sig not in lsd_original_map:
                continue

            original_node = (
                lsd_original_map[sig]
            )

            # LSD2 may encode date in name
            # or branch annotations.
            #
            # Try numeric node name first.
            try:

                value = float(
                    node.name
                )

                lsd_dates[
                    original_node.name
                ] = value

            except Exception:

                pass


    # --------------------------------------------------------
    # Final comparison rows
    # --------------------------------------------------------

    for node in tree.get_nonterminals():

        rows.append({

            "population":
                population,

            "node":
                node.name,

            "root_distance":
                tree.distance(
                    tree.root,
                    node
                ),

            "our_isotonic_generation":
                old_dates.get(
                    node.name,
                    np.nan
                ),

            "hard_upper_generation":
                hard_upper.get(
                    node.name,
                    np.nan
                ),

            "treetime_generation":
                tt_clade_dates.get(
                    node.name,
                    np.nan
                ),

            "lsd2_generation":
                lsd_dates.get(
                    node.name,
                    np.nan
                )
        })


df = pd.DataFrame(rows)


# ============================================================
# Date differences
# ============================================================

df["treetime_minus_isotonic"] = (
    df["treetime_generation"]
    -
    df["our_isotonic_generation"]
)

df["lsd2_minus_isotonic"] = (
    df["lsd2_generation"]
    -
    df["our_isotonic_generation"]
)

df["treetime_minus_lsd2"] = (
    df["treetime_generation"]
    -
    df["lsd2_generation"]
)


df.to_csv(
    OUT,
    index=False
)


# ============================================================
# Summary
# ============================================================

print(
    "\n========================================"
)

print(
    "DATING METHOD COMPARISON"
)

print(
    "========================================"
)

print(
    "Total internal nodes:",
    len(df)
)

print(
    "\nTreeTime dates available:",
    df["treetime_generation"]
    .notna()
    .sum()
)

print(
    "LSD2 dates available:",
    df["lsd2_generation"]
    .notna()
    .sum()
)


for col in [
    "treetime_minus_isotonic",
    "lsd2_minus_isotonic",
    "treetime_minus_lsd2"
]:

    valid = df[col].dropna()

    print(
        f"\n{col}"
    )

    if len(valid) == 0:

        print(
            "  no comparable nodes"
        )

    else:

        print(
            valid.describe()
        )


print(
    "\nLargest TreeTime vs LSD2 disagreements:"
)

compare = df.dropna(
    subset=[
        "treetime_generation",
        "lsd2_generation"
    ]
).copy()

if len(compare) > 0:

    compare["absolute_difference"] = abs(
        compare["treetime_minus_lsd2"]
    )

    print(
        compare.sort_values(
            "absolute_difference",
            ascending=False
        )[
            [
                "population",
                "node",
                "treetime_generation",
                "lsd2_generation",
                "treetime_minus_lsd2"
            ]
        ]
        .head(30)
        .to_string(index=False)
    )

print(
    "\nSaved:",
    OUT
)
