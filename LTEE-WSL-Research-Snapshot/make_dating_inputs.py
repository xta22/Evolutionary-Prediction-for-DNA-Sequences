from pathlib import Path
import re

from Bio import Phylo


TREE_DIR = Path("summary/LTEE_phylogenetic_trees")
OUT_DIR = Path("dating_inputs")
OUT_DIR.mkdir(exist_ok=True)


def get_generation(name):
    m = re.search(r"_(\d+)gen_", name)
    if m:
        return int(m.group(1))
    return None


for tree_file in sorted(TREE_DIR.glob("*.tre")):

    population = tree_file.stem
    tree = Phylo.read(tree_file, "newick")

    # --------------------------------------------------------
    # Date file for LSD2
    # First line = number of dated tips
    # --------------------------------------------------------

    dated = []

    for tip in tree.get_terminals():

        generation = get_generation(tip.name)

        if generation is not None:
            dated.append(
                (tip.name, generation)
            )

    lsd_file = (
        OUT_DIR
        / f"{population}.dates"
    )

    with open(lsd_file, "w") as f:

        f.write(
            str(len(dated))
            + "\n"
        )

        for name, generation in dated:
            f.write(
                f"{name}\t{generation}\n"
            )

    # --------------------------------------------------------
    # Date file for TreeTime
    # TreeTime wants a header with name + date
    # --------------------------------------------------------

    tt_file = (
        OUT_DIR
        / f"{population}.tsv"
    )

    with open(tt_file, "w") as f:

        f.write("name\tdate\n")

        for name, generation in dated:

            f.write(
                f"{name}\t{generation}\n"
            )

    print(
        population,
        "dated tips:",
        len(dated)
    )

print(
    "\nSaved dating inputs to:",
    OUT_DIR
)
