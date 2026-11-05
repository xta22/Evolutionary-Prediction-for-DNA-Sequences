from pathlib import Path
import torch
from evo2 import Evo2


def read_fasta(path):
    seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith(">"):
                seq.append(line.upper())
    return "".join(seq)


seq = read_fasta(
    Path("evo2_smoke_test/parent.fasta")
)

print("Sequence length:", len(seq))
print("GPU:", torch.cuda.get_device_name(0))

model = Evo2("evo2_7b_base")

tokens = model.tokenizer.tokenize(seq)

input_ids = torch.tensor(
    tokens,
    dtype=torch.long,
    device="cuda"
).unsqueeze(0)


with torch.inference_mode():
    output = model(input_ids)


def inspect(obj, name="output", depth=0):

    indent = "  " * depth

    if isinstance(obj, torch.Tensor):

        print(
            f"{indent}{name}: "
            f"Tensor "
            f"shape={tuple(obj.shape)} "
            f"dtype={obj.dtype} "
            f"device={obj.device}"
        )

        return

    if isinstance(obj, tuple):

        print(
            f"{indent}{name}: "
            f"tuple len={len(obj)}"
        )

        for i, x in enumerate(obj):
            inspect(
                x,
                f"{name}[{i}]",
                depth + 1
            )

        return

    if isinstance(obj, list):

        print(
            f"{indent}{name}: "
            f"list len={len(obj)}"
        )

        for i, x in enumerate(obj):
            inspect(
                x,
                f"{name}[{i}]",
                depth + 1
            )

        return

    if isinstance(obj, dict):

        print(
            f"{indent}{name}: "
            f"dict keys={list(obj.keys())}"
        )

        for k, v in obj.items():
            inspect(
                v,
                f"{name}[{k}]",
                depth + 1
            )

        return

    print(
        f"{indent}{name}: "
        f"{type(obj).__name__}"
    )


print("\n========================================")
print("FULL OUTPUT STRUCTURE")
print("========================================")

inspect(output)

print("\nPeak GPU memory:",
      round(
          torch.cuda.max_memory_allocated()
          / 1024**3,
          2
      ),
      "GB")
