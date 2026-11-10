import pandas as pd

df = pd.read_csv("LTEE_phylogenetic_branch_SNP_changes.csv")

df["reconstructed_changes"] = (
    df["snps_gained"] + df["snps_lost"]
)

valid = df[df["branch_length"].notna()].copy()

print("=== CORRELATION ===")
print(
    valid[
        ["branch_length", "reconstructed_changes"]
    ].corr()
)

print("\n=== BIGGEST SNP LOSSES ===")
print(
    valid.sort_values(
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
    ].head(20).to_string(index=False)
)

print("\n=== BIGGEST DISAGREEMENT WITH TREE ===")

valid["difference"] = (
    valid["reconstructed_changes"]
    - valid["branch_length"]
).abs()

print(
    valid.sort_values(
        "difference",
        ascending=False
    )[
        [
            "population",
            "parent",
            "child",
            "branch_length",
            "snps_gained",
            "snps_lost",
            "reconstructed_changes",
            "difference"
        ]
    ].head(20).to_string(index=False)
)
