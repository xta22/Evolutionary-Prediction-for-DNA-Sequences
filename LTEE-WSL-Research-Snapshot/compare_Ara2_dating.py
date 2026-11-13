from pathlib import Path
import re

import numpy as np
import pandas as pd
from Bio import Phylo


ORIGINAL_TREE = Path(
    "summary/LTEE_phylogenetic_trees/Ara+2.tre"
)

LSD2_TREE = Path(
    "dating_results/lsd2/Ara+2/Ara+2.date.nexus"
)

TREETIME_DATES = Path(
    "dating_results/treetime/"
    "Ara+2_alignment_relaxed/dates.tsv"
)

OLD_DATES = Path(
    "LTEE_internal_node_inferred_times.csv"
)

OUT = "Ara+2_dating_comparison.csv"


# ============================================================
# Helpers
# ============================================================

def descendant_signature(node):
    """
    Unique identity of a clade based on all descendant tips.
    """

    tips = sorted(
        tip.name
        for tip in node.get_terminals()
    )

    return "|".join(tips)


def read_original_tree():
    return Phylo.read(
        ORIGINAL_TREE,
        "newick"
    )


# ============================================================
# Name original internal nodes
# ============================================================

tree = read_original_tree()

counter = 0

for node in tree.get_nonterminals(
    order="preorder"
):

    node.name = (
        f"Ara+2_internal_{counter}"
    )

    counter += 1


original_nodes = {}

for node in tree.get_nonterminals():

    sig = descendant_signature(node)

    original_nodes[sig] = node


print(
    "Original internal nodes:",
    len(original_nodes)
)


# ============================================================
# Parse TreeTime dates
# ============================================================

tt_dates = {}

with open(TREETIME_DATES) as f:

    for line in f:

        line = line.strip()

        if (
            not line
            or line.startswith("#")
        ):
            continue

        fields = re.split(
            r"\s+",
            line
        )

        if len(fields) < 3:
            continue

        name = fields[0]

        numeric_date = float(
            fields[2]
        )

        tt_dates[name] = numeric_date


print(
    "TreeTime dated nodes:",
    len(tt_dates)
)


# ============================================================
# Load TreeTime tree
# ============================================================

treetime_tree = (
    Path(
        "dating_results/treetime/"
        "Ara+2_alignment_relaxed/"
        "timetree.nexus"
    )
)

tt_tree = Phylo.read(
    treetime_tree,
    "nexus"
)


# ============================================================
# Match TreeTime nodes
# to original clades
# ============================================================

treetime_clade_dates = {}

tt_matched = 0

for node in tt_tree.get_nonterminals():

    if node.name is None:
        continue

    if node.name not in tt_dates:
        continue

    sig = descendant_signature(
        node
    )

    if sig in original_nodes:

        original_node = (
            original_nodes[sig]
        )

        treetime_clade_dates[
            original_node.name
        ] = tt_dates[node.name]

        tt_matched += 1


print(
    "TreeTime matched internal nodes:",
    tt_matched
)


# ============================================================
# Parse LSD2 dated Nexus
#
# Internal nodes are unnamed, but their
# dates are stored in:
#
# [&date="1234.56"]
#
# We reconstruct node date by matching
# descendant-tip clade structure.
#
# ============================================================

lsd_tree = Phylo.read(
    LSD2_TREE,
    "nexus"
)


# ------------------------------------------------------------
# Unfortunately Bio.Phylo does not always retain Nexus
# [&date=...] annotations consistently.
#
# So parse the actual Newick line ourselves.
# ------------------------------------------------------------

with open(LSD2_TREE) as f:

    nexus_text = f.read()


# Get tree expression
m = re.search(
    r"tree\s+1\s*=\s*(.*?);",
    nexus_text,
    flags=re.IGNORECASE | re.DOTALL
)

if not m:
    raise RuntimeError(
        "Could not find tree expression "
        "in LSD2 Nexus file."
    )

lsd_newick = m.group(1).strip()


# ============================================================
# Custom Newick parser
# ============================================================

class Node:

    def __init__(
        self,
        name=None,
        date=None,
        children=None
    ):

        self.name = name
        self.date = date
        self.children = (
            children
            if children is not None
            else []
        )

    def is_leaf(self):
        return len(self.children) == 0


def parse_lsd_newick(text):

    pos = 0

    def skip_spaces():

        nonlocal pos

        while (
            pos < len(text)
            and text[pos].isspace()
        ):
            pos += 1


    def parse_token():

        nonlocal pos

        start = pos

        while (
            pos < len(text)
            and text[pos]
            not in ",():;"
        ):

            pos += 1

        token = text[
            start:pos
        ].strip()

        return token


    def parse_node():

        nonlocal pos

        skip_spaces()

        # ----------------------------------------------------
        # Internal node
        # ----------------------------------------------------

        if pos < len(text) and text[pos] == "(":

            pos += 1

            children = []

            while True:

                children.append(
                    parse_node()
                )

                skip_spaces()

                if (
                    pos < len(text)
                    and text[pos] == ","
                ):

                    pos += 1
                    continue

                if (
                    pos < len(text)
                    and text[pos] == ")"
                ):

                    pos += 1
                    break

                raise RuntimeError(
                    "Unexpected character "
                    f"at position {pos}"
                )

            skip_spaces()

            # ------------------------------------------------
            # Read optional internal annotation / branch
            # ------------------------------------------------

            date = None

            # [&date="..."]
            if text.startswith(
                "[&date=",
                pos
            ):

                end = text.find(
                    "]",
                    pos
                )

                annotation = text[
                    pos:end + 1
                ]

                md = re.search(
                    r'date="([^"]+)"',
                    annotation
                )

                if md:
                    date = float(
                        md.group(1)
                    )

                pos = end + 1

            # ------------------------------------------------
            # Ignore branch length
            # ------------------------------------------------

            skip_spaces()

            if (
                pos < len(text)
                and text[pos] == ":"
            ):

                pos += 1

                while (
                    pos < len(text)
                    and text[pos]
                    not in ",();"
                ):

                    pos += 1

            return Node(
                date=date,
                children=children
            )

        # ----------------------------------------------------
        # Leaf
        # ----------------------------------------------------

        name = parse_token()

        return Node(
            name=name
        )


    return parse_node()


lsd_root = parse_lsd_newick(
    lsd_newick
)


# ============================================================
# Calculate descendant signatures
# ============================================================

def lsd_tip_names(node):

    if node.is_leaf():
        return [node.name]

    tips = []

    for child in node.children:

        tips.extend(
            lsd_tip_names(child)
        )

    return tips


lsd_clade_dates = {}


def walk_lsd(node):

    if node.is_leaf():
        return

    tips = sorted(
        lsd_tip_names(node)
    )

    sig = "|".join(tips)

    if node.date is not None:

        if sig in original_nodes:

            original_node = (
                original_nodes[sig]
            )

            lsd_clade_dates[
                original_node.name
            ] = node.date

    for child in node.children:

        walk_lsd(child)


walk_lsd(lsd_root)


print(
    "LSD2 matched internal nodes:",
    len(lsd_clade_dates)
)


# ============================================================
# Load our isotonic dates
# ============================================================

old_dates = {}

hard_upper = {}

if OLD_DATES.exists():

    old_df = pd.read_csv(
        OLD_DATES
    )

    old_df = old_df[
        old_df["population"]
        == "Ara+2"
    ]

    for _, row in old_df.iterrows():

        old_dates[
            row["node"]
        ] = row[
            "estimated_generation"
        ]

        hard_upper[
            row["node"]
        ] = row[
            "hard_upper_generation"
        ]


# ============================================================
# Build comparison table
# ============================================================

rows = []

for sig, node in original_nodes.items():

    rows.append({

        "population":
            "Ara+2",

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

        "lsd2_generation":
            lsd_clade_dates.get(
                node.name,
                np.nan
            ),

        "treetime_generation":
            treetime_clade_dates.get(
                node.name,
                np.nan
            )
    })


df = pd.DataFrame(rows)


# ============================================================
# Differences
# ============================================================

df["lsd2_minus_treetime"] = (
    df["lsd2_generation"]
    -
    df["treetime_generation"]
)

df["lsd2_minus_isotonic"] = (
    df["lsd2_generation"]
    -
    df["our_isotonic_generation"]
)

df["treetime_minus_isotonic"] = (
    df["treetime_generation"]
    -
    df["our_isotonic_generation"]
)


# ============================================================
# Save
# ============================================================

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
    "ARA+2 DATING COMPARISON"
)

print(
    "========================================"
)

print(
    "\nInternal nodes:",
    len(df)
)

print(
    "LSD2 matched:",
    df["lsd2_generation"]
    .notna()
    .sum()
)

print(
    "TreeTime matched:",
    df["treetime_generation"]
    .notna()
    .sum()
)


# ------------------------------------------------------------
# Direct comparison
# ------------------------------------------------------------

compare = df.dropna(
    subset=[
        "lsd2_generation",
        "treetime_generation"
    ]
).copy()


if len(compare) > 0:

    print(
        "\n=== LSD2 vs TreeTime ==="
    )

    print(
        compare[
            [
                "lsd2_generation",
                "treetime_generation"
            ]
        ].describe()
    )

    print(
        "\nMedian absolute difference:",
        np.median(
            np.abs(
                compare[
                    "lsd2_generation"
                ]
                -
                compare[
                    "treetime_generation"
                ]
            )
        )
    )


# ------------------------------------------------------------
# Largest disagreements
# ------------------------------------------------------------

if len(compare) > 0:

    compare[
        "absolute_difference"
    ] = np.abs(
        compare["lsd2_generation"]
        -
        compare["treetime_generation"]
    )

    print(
        "\n=== LARGEST DISAGREEMENTS ==="
    )

    print(
        compare.sort_values(
            "absolute_difference",
            ascending=False
        )[
            [
                "node",
                "root_distance",
                "lsd2_generation",
                "treetime_generation",
                "absolute_difference"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


# ============================================================
# Root / temporal sanity check
# ============================================================

print(
    "\n=== ROOT / TIME SANITY ==="
)

print(
    "LSD2 root date:",
    lsd_root.date
)

print(
    "TreeTime root date:",
    tt_dates.get(
        "NODE_0000000",
        np.nan
    )
)


print(
    "\nSaved:",
    OUT
)
