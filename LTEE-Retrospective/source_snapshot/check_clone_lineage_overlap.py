import pandas as pd
from pathlib import Path
from itertools import product


METADATA_FILE = "tempo_264_clone_metadata.csv"


# --------------------------------------------------
# Read mutations from one GenomeDiff file
# --------------------------------------------------
def read_mutations(gd_file):
    mutations = set()

    with open(gd_file) as f:
        for line in f:

            line = line.strip()

            # skip metadata / empty lines
            if not line or line.startswith("#"):
                continue

            fields = line.split()

            # GenomeDiff:
            # mutation_type, id, parent_id, refseq, position, ...
            mutation_type = fields[0]

            # We only want actual mutation records.
            # These are the common breseq mutation types.
            mutation_types = {
                "SNP", "SUB", "DEL", "INS",
                "MOB", "AMP", "INV", "CON"
            }

            if mutation_type not in mutation_types:
                continue

            # Ignore the GenomeDiff record ID and parent ID
            # because they are file-specific.
            #
            # Keep:
            # mutation type + reference + position + mutation details
            mutation_key = tuple(
                [fields[0]] + fields[3:]
            )

            mutations.add(mutation_key)

    return mutations


# --------------------------------------------------
# Similarity metrics
# --------------------------------------------------
def compare_sets(earlier, later):

    shared = earlier & later

    if len(earlier) > 0:
        inheritance_fraction = len(shared) / len(earlier)
    else:
        inheritance_fraction = 0

    union = earlier | later

    if len(union) > 0:
        jaccard = len(shared) / len(union)
    else:
        jaccard = 0

    return {
        "earlier_mutations": len(earlier),
        "later_mutations": len(later),
        "shared_mutations": len(shared),
        "inheritance_fraction": inheritance_fraction,
        "jaccard": jaccard,
    }


# --------------------------------------------------
# Load metadata
# --------------------------------------------------
df = pd.read_csv(METADATA_FILE)

df = df.sort_values(
    ["population", "generation", "clone"]
)

# cache mutation sets so each file is only read once
mutation_cache = {}

for _, row in df.iterrows():
    mutation_cache[row["file"]] = read_mutations(row["file"])


results = []


# --------------------------------------------------
# Compare adjacent generations within each population
# --------------------------------------------------
for population, pop_df in df.groupby("population"):

    generations = sorted(pop_df["generation"].unique())

    for i in range(len(generations) - 1):

        g1 = generations[i]
        g2 = generations[i + 1]

        earlier_clones = pop_df[
            pop_df["generation"] == g1
        ]

        later_clones = pop_df[
            pop_df["generation"] == g2
        ]

        # compare all possible pairs
        for (_, earlier), (_, later) in product(
            earlier_clones.iterrows(),
            later_clones.iterrows()
        ):

            m1 = mutation_cache[earlier["file"]]
            m2 = mutation_cache[later["file"]]

            metrics = compare_sets(m1, m2)

            results.append({
                "population": population,

                "generation_1": g1,
                "clone_1": earlier["title"],

                "generation_2": g2,
                "clone_2": later["title"],

                **metrics
            })


result_df = pd.DataFrame(results)

result_df.to_csv(
    "tempo_adjacent_clone_overlap.csv",
    index=False
)


# --------------------------------------------------
# Print overall distribution
# --------------------------------------------------
print("\n=== ALL ADJACENT CLONE PAIRS ===")

print(
    result_df[
        [
            "inheritance_fraction",
            "jaccard"
        ]
    ].describe()
)


# --------------------------------------------------
# Find best later clone for each earlier clone
# --------------------------------------------------
best = (
    result_df
    .sort_values(
        "inheritance_fraction",
        ascending=False
    )
    .groupby(
        [
            "population",
            "generation_1",
            "clone_1"
        ],
        as_index=False
    )
    .first()
)

best.to_csv(
    "tempo_best_lineage_matches.csv",
    index=False
)


print("\n=== BEST MATCHES ===")

print(
    best[
        [
            "population",
            "generation_1",
            "clone_1",
            "generation_2",
            "clone_2",
            "earlier_mutations",
            "later_mutations",
            "shared_mutations",
            "inheritance_fraction",
            "jaccard"
        ]
    ].head(40).to_string(index=False)
)


# --------------------------------------------------
# Summary by population
# --------------------------------------------------
print("\n=== BEST-MATCH INHERITANCE BY POPULATION ===")

population_summary = (
    best.groupby("population")[
        "inheritance_fraction"
    ]
    .agg(["count", "mean", "median", "min", "max"])
)

print(population_summary)
