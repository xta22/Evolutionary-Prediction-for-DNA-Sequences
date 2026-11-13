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

tempo = df[
    df["generation"].isin(tempo_generations)
    & df["population"].isin(tempo_pops)
].copy()

# Remove the extra Ara-3 X/Y clones
tempo = tempo[
    ~(
        (tempo["population"] == "Ara-3")
        & tempo["clone"].isin(["X", "Y"])
    )
]

tempo = tempo.sort_values(
    ["population", "generation", "clone"]
)

print("Total Tempo clones:", len(tempo))

print("\nClone counts per population-generation:")
print(
    tempo.groupby(
        ["population", "generation"]
    ).size().value_counts()
)

tempo.to_csv(
    "tempo_264_clone_metadata.csv",
    index=False
)

print("\nSaved: tempo_264_clone_metadata.csv")
