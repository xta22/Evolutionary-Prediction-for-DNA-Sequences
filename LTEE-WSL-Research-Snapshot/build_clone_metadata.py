from pathlib import Path
import pandas as pd

gd_dir = Path("LTEE-clone-curated")

rows = []

for gd_file in gd_dir.glob("*.gd"):

    metadata = {
        "file": str(gd_file),
        "title": None,
        "generation": None,
        "population": None,
        "clone": None,
        "mutator_status": None,
    }

    mutation_count = 0

    with open(gd_file) as f:
        for line in f:

            if line.startswith("#=TITLE"):
                metadata["title"] = line.strip().split(None, 1)[1]

            elif line.startswith("#=TIME"):
                metadata["generation"] = int(
                    line.strip().split(None, 1)[1]
                )

            elif line.startswith("#=POPULATION"):
                metadata["population"] = line.strip().split(None, 1)[1]

            elif line.startswith("#=CLONE"):
                metadata["clone"] = line.strip().split(None, 1)[1]

            elif line.startswith("#=MUTATOR_STATUS"):
                metadata["mutator_status"] = (
                    line.strip().split(None, 1)[1]
                )

            elif not line.startswith("#") and line.strip():
                mutation_count += 1

    metadata["mutation_count"] = mutation_count
    rows.append(metadata)


df = pd.DataFrame(rows)

df = df.sort_values(
    ["population", "generation", "clone"]
)

df.to_csv(
    "LTEE_clone_metadata.csv",
    index=False
)

print(df.head(20))

print("\nNumber of clone genomes:")
print(len(df))

print("\nPopulations:")
print(df["population"].value_counts().sort_index())

print("\nGenerations:")
print(
    sorted(df["generation"].dropna().unique())
)

print("\nNumber of genomes per generation:")
print(
    df.groupby("generation").size()
)

print("\nMutation counts:")
print(df["mutation_count"].describe())
