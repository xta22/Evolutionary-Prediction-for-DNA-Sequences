from pathlib import Path
import re
import pandas as pd
from Bio import Phylo


TREE_DIR = Path("summary/LTEE_phylogenetic_trees")

OUT = "LTEE_internal_node_time_constraints.csv"


def get_generation(name):
    """
    Extract generation from a sampled clone tip name.

    Examples:
        Ara+2_500gen_769A       -> 500
        Ara-3_30000gen_ZDB16    -> 30000
        Anc+_0gen_REL607         -> 0
    """

    if name is None:
        return None

    m = re.search(r"_(\d+)gen_", name)

    if m:
        return int(m.group(1))

    return None


def name_internal_nodes(tree, population):

    counter = 0

    for node in tree.get_nonterminals(order="preorder"):

        if node.name is None:
            node.name = f"{population}_internal_{counter}"
            counter += 1


rows = []


for tree_file in sorted(TREE_DIR.glob("*.tre")):

    population = tree_file.stem

    print(f"Processing {population}")

    tree = Phylo.read(tree_file, "newick")

    name_internal_nodes(tree, population)


    # --------------------------------------------------------
    # For every node, collect descendant sampled generations
    # --------------------------------------------------------

    descendant_generations = {}

    for node in tree.find_clades(order="postorder"):

        if node.is_terminal():

            g = get_generation(node.name)

            if g is None:
                descendant_generations[node] = []
            else:
                descendant_generations[node] = [g]

        else:

            generations = []

            for child in node.clades:
                generations.extend(
                    descendant_generations[child]
                )

            descendant_generations[node] = generations


    # --------------------------------------------------------
    # Internal nodes only
    # --------------------------------------------------------

    for node in tree.get_nonterminals(order="preorder"):

        gens = descendant_generations[node]

        if len(gens) == 0:
            continue

        min_descendant_generation = min(gens)
        max_descendant_generation = max(gens)

        descendant_tips = node.get_terminals()

        sampled_descendants = [
            tip.name
            for tip in descendant_tips
        ]

        rows.append({
            "population":
                population,

            "node":
                node.name,

            "n_descendant_tips":
                len(descendant_tips),

            "min_descendant_generation":
                min_descendant_generation,

            "max_descendant_generation":
                max_descendant_generation,

            # Hard temporal fact:
            # ancestor must exist no later than its
            # earliest sampled descendant.
            "latest_possible_generation":
                min_descendant_generation,

            "root_distance":
                tree.distance(
                    tree.root,
                    node
                ),

            "descendant_tips":
                ";".join(sampled_descendants)
        })


df = pd.DataFrame(rows)

df = df.sort_values(
    [
        "population",
        "latest_possible_generation",
        "root_distance"
    ]
)

df.to_csv(
    OUT,
    index=False
)


print("\n========================================")
print("DONE")
print("========================================")

print(
    "Internal nodes:",
    len(df)
)

print("\nNodes per population:")
print(
    df.groupby("population").size()
)

print(
    "\nLatest possible generation summary:"
)

print(
    df["latest_possible_generation"]
    .describe()
)

print("\nExample rows:")

print(
    df[
        [
            "population",
            "node",
            "n_descendant_tips",
            "min_descendant_generation",
            "max_descendant_generation",
            "latest_possible_generation",
            "root_distance"
        ]
    ]
    .head(30)
    .to_string(index=False)
)

print(
    "\nSaved:",
    OUT
)
