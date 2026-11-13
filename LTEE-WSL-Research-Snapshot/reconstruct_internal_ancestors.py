from pathlib import Path
from collections import defaultdict
import re

import pandas as pd
from Bio import Phylo


# ============================================================
# Paths
# ============================================================

TREE_DIR = Path("summary/LTEE_phylogenetic_trees")
GD_DIR = Path("LTEE-clone-curated")

OUT_NODE = "LTEE_internal_node_SNP_states.csv"
OUT_BRANCH = "LTEE_phylogenetic_branch_SNP_changes.csv"
OUT_MISSING = "LTEE_missing_GD_files.csv"


# ============================================================
# Read SNPs from GenomeDiff
# ============================================================

def read_snps(gd_file):
    """
    Read SNP records from a GenomeDiff file.

    Expected format:
    SNP  id  parent  refseq  position  new_base

    Returns:
        set of tuples:
        (refseq, position, new_base)
    """

    snps = set()

    with open(gd_file) as f:

        for line in f:

            line = line.strip()

            if not line or line.startswith("#"):
                continue

            # Handles tabs or arbitrary whitespace
            fields = line.split()

            if len(fields) < 6:
                continue

            if fields[0] != "SNP":
                continue

            refseq = fields[3]
            position = int(fields[4])
            new_base = fields[5]

            snps.add(
                (refseq, position, new_base)
            )

    return snps


# ============================================================
# Convert tree tip name -> GD filename
# ============================================================

def clone_to_gd_file(clone_name):
    """
    Handle naming inconsistencies between tree tip names
    and GenomeDiff filenames.

    Example:

    tree:
        Ara-2_30000gen_30K_11832

    actual file:
        Ara-2_30000gen_11832.gd
    """

    gd_name = clone_name.replace("_30K_", "_")

    return GD_DIR / f"{gd_name}.gd"


# ============================================================
# Give internal nodes unique names
# ============================================================

def name_internal_nodes(tree, population):

    counter = 0

    for node in tree.get_nonterminals(order="preorder"):

        if node.name is None:

            node.name = (
                f"{population}_internal_{counter}"
            )

            counter += 1


# ============================================================
# Fitch parsimony
# ============================================================

def fitch_reconstruct(tree, leaf_states):
    """
    Binary Fitch reconstruction.

    leaf_states values:

        0    mutation absent
        1    mutation present
        None unknown

    Unknown leaves are represented as {0, 1}.
    """

    possible = {}

    # --------------------------------------------------------
    # Bottom-up pass
    # --------------------------------------------------------

    for node in tree.find_clades(order="postorder"):

        if node.is_terminal():

            value = leaf_states.get(
                node.name,
                None
            )

            if value is None:

                possible[node] = {0, 1}

            else:

                possible[node] = {value}

        else:

            child_sets = [
                possible[child]
                for child in node.clades
            ]

            intersection = set.intersection(
                *child_sets
            )

            if intersection:

                possible[node] = intersection

            else:

                possible[node] = set.union(
                    *child_sets
                )

    # --------------------------------------------------------
    # Top-down pass
    # --------------------------------------------------------

    state = {}

    root = tree.root

    # Mutations are defined relative to the founder/reference.
    # Prefer absence if root state is ambiguous.
    if 0 in possible[root]:

        state[root] = 0

    else:

        state[root] = 1

    def assign_children(parent):

        for child in parent.clades:

            # Prefer same state as parent if allowed
            if state[parent] in possible[child]:

                state[child] = state[parent]

            else:

                # deterministic choice
                state[child] = min(
                    possible[child]
                )

            assign_children(child)

    assign_children(root)

    return state


# ============================================================
# Main reconstruction
# ============================================================

node_rows = []
branch_rows = []
missing_rows = []


for tree_file in sorted(
    TREE_DIR.glob("*.tre")
):

    population = tree_file.stem

    print(
        f"\nProcessing {population}"
    )

    tree = Phylo.read(
        tree_file,
        "newick"
    )

    name_internal_nodes(
        tree,
        population
    )

    # --------------------------------------------------------
    # Read observed SNP sets at tree tips
    # --------------------------------------------------------

    leaf_snps = {}
    all_snps = set()

    for tip in tree.get_terminals():

        clone = tip.name

        # REL606 is the reference genome itself
        if clone == "Anc-_0gen_REL606":

            leaf_snps[clone] = set()

            continue

        gd_file = clone_to_gd_file(
            clone
        )

        if not gd_file.exists():

            print(
                f"WARNING: missing GD: {gd_file}"
            )

            missing_rows.append({
                "population": population,
                "clone": clone,
                "expected_file": str(gd_file)
            })

            # IMPORTANT:
            # Do NOT add empty set here.
            # Missing data should remain unknown.
            continue

        snps = read_snps(
            gd_file
        )

        leaf_snps[clone] = snps

        all_snps.update(snps)

        # Optional sanity check for REL607
        if clone == "Anc+_0gen_REL607":

            print(
                "  REL607 founder SNPs:",
                len(snps)
            )

    print(
        "  tree tips:",
        len(tree.get_terminals())
    )

    print(
        "  unique SNPs:",
        len(all_snps)
    )

    # --------------------------------------------------------
    # node -> inferred SNP set
    # --------------------------------------------------------

    node_mutations = defaultdict(set)

    # --------------------------------------------------------
    # Reconstruct every SNP independently
    # --------------------------------------------------------

    for mutation in all_snps:

        leaf_states = {}

        for tip in tree.get_terminals():

            clone = tip.name

            # Missing clone -> unknown state
            if clone not in leaf_snps:

                leaf_states[clone] = None

                continue

            leaf_states[clone] = int(
                mutation
                in leaf_snps[clone]
            )

        reconstructed = (
            fitch_reconstruct(
                tree,
                leaf_states
            )
        )

        for node, state in (
            reconstructed.items()
        ):

            if state == 1:

                node_mutations[
                    node.name
                ].add(
                    mutation
                )

    # --------------------------------------------------------
    # Save inferred SNPs for internal nodes
    # --------------------------------------------------------

    for node in tree.get_nonterminals():

        mutations = (
            node_mutations[
                node.name
            ]
        )

        for (
            refseq,
            position,
            new_base
        ) in sorted(
            mutations,
            key=lambda x: x[1]
        ):

            node_rows.append({
                "population": population,
                "node": node.name,
                "refseq": refseq,
                "position": position,
                "new_base": new_base,
            })

    # --------------------------------------------------------
    # Compare parent and child states on every branch
    # --------------------------------------------------------

    for parent in tree.find_clades(
        order="preorder"
    ):

        parent_set = (
            node_mutations[
                parent.name
            ]
        )

        for child in parent.clades:

            child_set = (
                node_mutations[
                    child.name
                ]
            )

            gained = (
                child_set
                - parent_set
            )

            lost = (
                parent_set
                - child_set
            )

            generation_match = (
                re.search(
                    r"_(\d+)gen_",
                    child.name
                )
            )

            if generation_match:

                child_generation = int(
                    generation_match.group(1)
                )

            else:

                child_generation = None

            branch_rows.append({
                "population":
                    population,

                "parent":
                    parent.name,

                "child":
                    child.name,

                "child_generation":
                    child_generation,

                "branch_length":
                    child.branch_length,

                "parent_snp_count":
                    len(parent_set),

                "child_snp_count":
                    len(child_set),

                "snps_gained":
                    len(gained),

                "snps_lost":
                    len(lost),

                "reconstructed_changes":
                    len(gained)
                    + len(lost),
            })


# ============================================================
# Save outputs
# ============================================================

node_df = pd.DataFrame(
    node_rows
)

branch_df = pd.DataFrame(
    branch_rows
)

missing_df = pd.DataFrame(
    missing_rows
)


node_df.to_csv(
    OUT_NODE,
    index=False
)

branch_df.to_csv(
    OUT_BRANCH,
    index=False
)

missing_df.to_csv(
    OUT_MISSING,
    index=False
)


# ============================================================
# Summary
# ============================================================

print(
    "\n========================================"
)

print(
    "DONE"
)

print(
    "========================================"
)


print(
    "\nInternal-node SNP-state rows:",
    len(node_df)
)


print(
    "Phylogenetic branches:",
    len(branch_df)
)


print(
    "Missing GD files:",
    len(missing_df)
)


if len(missing_df) > 0:

    print(
        "\n=== MISSING GD FILES ==="
    )

    print(
        missing_df.to_string(
            index=False
        )
    )


print(
    "\n=== BRANCH SNP CHANGES ==="
)

print(
    branch_df[
        [
            "snps_gained",
            "snps_lost",
            "reconstructed_changes"
        ]
    ].describe()
)


# ============================================================
# Correlation with official tree branch lengths
# ============================================================

valid = branch_df[
    branch_df[
        "branch_length"
    ].notna()
].copy()


if len(valid) > 1:

    corr = valid[
        [
            "branch_length",
            "reconstructed_changes"
        ]
    ].corr()

    print(
        "\n=== BRANCH LENGTH CORRELATION ==="
    )

    print(corr)


print(
    "\n=== BIGGEST SNP LOSSES ==="
)

print(
    branch_df.sort_values(
        "snps_lost",
        ascending=False
    )[
        [
            "population",
            "parent",
            "child",
            "branch_length",
            "snps_gained",
            "snps_lost",
            "reconstructed_changes"
        ]
    ].head(
        20
    ).to_string(
        index=False
    )
)


print(
    "\nSaved:"
)

print(
    OUT_NODE
)

print(
    OUT_BRANCH
)

print(
    OUT_MISSING
)
