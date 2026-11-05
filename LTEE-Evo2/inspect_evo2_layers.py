import torch
from evo2 import Evo2

print("Loading Evo2 7B...", flush=True)
model = Evo2("evo2_7b_base")

print("\n=== MODEL MODULE NAMES ===")

candidates = []

for name, module in model.model.named_modules():
    if not name:
        continue

    depth = name.count(".")

    if depth <= 3:
        print(name, type(module).__name__)

    # Find likely transformer / Hyena blocks
    if any(
        keyword in name.lower()
        for keyword in ["block", "layer"]
    ):
        candidates.append(name)

print("\n=== CANDIDATE LAYERS ===")
for name in candidates[:100]:
    print(name)

print("\nTotal candidate layers:", len(candidates))

print("\nGPU allocated:",
      round(torch.cuda.memory_allocated() / 1024**3, 2),
      "GB")
