from pathlib import Path
import re
import pandas as pd
from Bio import Phylo

TREE_DIR = Path("summary/LTEE_phylogenetic_trees")

rows = []

for tree_file in TREE_DIR.glob("*.tre"):

    population = tree_file.stem

    tree = Phylo.read(tree_file, "newick")

    # loop through all terminal nodes = sampled clones
    for terminal in tree.get_terminals():

        clone_name = terminal.name

        # Example:
        # Ara+2_500gen_769A
        # Ara-3_30000gen_ZDB16
        # Anc+_0gen_REL607

        match = re.search(r"_(\d+)gen_", clone_name)

        if match:
            generation = int(match.group(1))
        else:
            generation = None

        # distance from root to this clone
        root_distance = tree.distance(tree.root, terminal)

        rows.append({
            "population": population,
            "clone": clone_name,
            "generation": generation,
            "root_distance": root_distance
        })


df = pd.DataFrame(rows)

df = df.sort_values(
    ["population", "generation", "clone"]
)

df.to_csv(
    "LTEE_tree_clone_distances.csv",
    index=False
)

print("\n=== BASIC INFO ===")
print("Total tree tips:", len(df))

print("\nTips per population:")
print(df.groupby("population").size())

print("\n=== FIRST 30 ROWS ===")
print(df.head(30).to_string(index=False))

print("\n=== ROOT DISTANCE SUMMARY ===")
print(
    df.groupby("population")["root_distance"]
      .describe()[["count", "mean", "min", "max"]]
)

print("\nSaved: LTEE_tree_clone_distances.csv")
