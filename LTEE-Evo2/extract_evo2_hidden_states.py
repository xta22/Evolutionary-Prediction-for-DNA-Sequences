from pathlib import Path
import torch
from evo2 import Evo2

DATA = Path("evo2_smoke_test/parent.fasta")
OUT = Path("results/evo2_hidden_smoke_test.pt")

LAYERS = [
    "blocks.7",
    "blocks.15",
    "blocks.31",
]


def read_fasta(path):
    with open(path) as f:
        return "".join(
            line.strip().upper()
            for line in f
            if line.strip() and not line.startswith(">")
        )


def describe(name, tensor):
    print(
        name,
        "shape:", tuple(tensor.shape),
        "dtype:", tensor.dtype,
        "device:", tensor.device,
        flush=True
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    seq = read_fasta(DATA)

    print("Sequence length:", len(seq), flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    print("\nLoading Evo2 7B...", flush=True)
    model = Evo2("evo2_7b_base")

    # Confirm layer names before running
    for name in LAYERS:
        model.model.get_submodule(name)
        print("Found layer:", name, flush=True)

    tokens = model.tokenizer.tokenize(seq)

    input_ids = torch.tensor(
        tokens,
        dtype=torch.long,
        device="cuda"
    ).unsqueeze(0)

    print("\nExtracting hidden representations...", flush=True)

    with torch.inference_mode():
        logits, embeddings = model(
            input_ids,
            return_embeddings=True,
            layer_names=LAYERS
        )

    print("\n=== EMBEDDING SHAPES ===", flush=True)

    saved = {}

    for name in LAYERS:
        hidden = embeddings[name]
        describe(name, hidden)

        # Save the representation at the actual mutation site.
        # Unlike next-token logits, this hidden state can include
        # the current parent nucleotide.
        mutation_idx = 4096

        if hidden.ndim != 3:
            raise RuntimeError(
                f"Unexpected hidden shape for {name}: {hidden.shape}"
            )

        if hidden.shape[1] != len(seq):
            raise RuntimeError(
                f"Sequence length mismatch for {name}"
            )

        vector = hidden[0, mutation_idx, :].float().cpu()

        saved[name] = vector

        print(
            name,
            "mutation-site vector:",
            tuple(vector.shape),
            flush=True
        )

    OUT.parent.mkdir(exist_ok=True)

    torch.save(
        {
            "layers": saved,
            "mutation_index": 4096,
            "sequence_length": len(seq),
        },
        OUT
    )

    print("\nSaved:", OUT, flush=True)

    print(
        "Peak GPU memory:",
        round(
            torch.cuda.max_memory_allocated() / 1024**3,
            2
        ),
        "GB",
        flush=True
    )

    print("\nHIDDEN EXTRACTION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
