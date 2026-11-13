import pandas as pd

df = pd.read_csv("LTEE_clone_metadata.csv")

tempo_generations = [
    500, 1000, 1500, 2000,
    5000, 10000, 15000, 20000,
    30000, 40000, 50000
]

tempo_pops = [
    "Ara+1", "Ara+2", "Ara+3", "Ara+4", "Ara+5", "Ara+6",
    "Ara-1", "Ara-2", "Ara-3", "Ara-4", "Ara-5", "Ara-6"
]

sub = df[
    df["generation"].isin(tempo_generations)
    & df["population"].isin(tempo_pops)
].copy()

counts = (
    sub.groupby(["population", "generation"])
       .size()
       .reset_index(name="n_clones")
)

print("=== Candidate Tempo subset ===")
print("Total clone files:", len(sub))

print("\n=== Clone count distribution ===")
print(counts["n_clones"].value_counts().sort_index())

print("\n=== population × generation NOT equal to 2 ===")
problem = counts[counts["n_clones"] != 2]

if len(problem) == 0:
    print("All groups have exactly 2 clones.")
else:
    print(problem.to_string(index=False))

print("\n=== Extra clone details ===")

for _, row in problem.iterrows():
    tmp = sub[
        (sub["population"] == row["population"])
        & (sub["generation"] == row["generation"])
    ]

    print(
        f"\n{row['population']} "
        f"generation {row['generation']}:"
    )

    print(
        tmp[
            [
                "title",
                "clone",
                "mutation_count",
                "file"
            ]
        ].to_string(index=False)
    )
