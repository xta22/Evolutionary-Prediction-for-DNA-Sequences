import pandas as pd

df = pd.read_csv("LTEE_tree_clone_distances.csv")

# Remove ancestor
df = df[df["generation"] > 0].copy()

print("=== CORRELATION BY POPULATION ===")

for pop, g in df.groupby("population"):
    corr = g["generation"].corr(g["root_distance"])
    print(f"{pop:6s}  r = {corr:.3f}")
