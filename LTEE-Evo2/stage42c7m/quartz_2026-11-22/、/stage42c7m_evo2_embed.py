#!/usr/bin/env python3
"""Extract frozen Evo2 alternate-minus-reference SNP-window embeddings on GPU."""

import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np


def read_fasta(path):
    records = {}
    name = None
    chunks = []
    with path.open(encoding="ascii") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records[name] = "".join(chunks).upper()
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(re.sub(r"\s+", "", line))
    if name is not None:
        records[name] = "".join(chunks).upper()
    if not records:
        raise ValueError(f"No FASTA records in {path}")
    return records


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing variant_index.csv and variant_windows.fasta")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="evo2_7b")
    parser.add_argument("--layer", default="blocks.28.mlp.l3")
    args = parser.parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    index_path = input_dir / "variant_index.csv"
    fasta_path = input_dir / "variant_windows.fasta"
    output = args.output.expanduser().resolve()
    for path in (index_path, fasta_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be absent or empty: {output}")
    with index_path.open(newline="", encoding="utf-8-sig") as handle:
        index_rows = list(csv.DictReader(handle))
    if not index_rows:
        raise ValueError("variant_index.csv contains no variants")
    sequences = read_fasta(fasta_path)

    try:
        import torch
        from evo2 import Evo2
    except Exception as exc:
        raise RuntimeError("Could not import Evo2/PyTorch; activate the server Evo2 environment.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this on an allocated GPU node")

    print(f"Loading {args.model} on {torch.cuda.get_device_name(0)}", flush=True)
    model = Evo2(args.model)
    if hasattr(model, "model") and hasattr(model.model, "eval"):
        model.model.eval()

    vectors = []
    started = time.time()
    embedding_dim = None
    for i, row in enumerate(index_rows, 1):
        ref_id = row["reference_fasta_id"]
        alt_id = row["alternate_fasta_id"]
        if ref_id not in sequences or alt_id not in sequences:
            raise ValueError(f"Missing FASTA pair for {row['variant_id']}")
        pooled = []
        for fasta_id in (ref_id, alt_id):
            dna = sequences[fasta_id]
            token_ids = model.tokenizer.tokenize(dna)
            input_ids = torch.tensor(token_ids, dtype=torch.int).unsqueeze(0).to("cuda:0")
            with torch.inference_mode():
                _logits, embedding_dict = model(
                    input_ids,
                    return_embeddings=True,
                    layer_names=[args.layer],
                )
            tensor = embedding_dict[args.layer]
            if tensor.ndim != 3 or tensor.shape[0] != 1:
                raise RuntimeError(f"Unexpected embedding shape for {fasta_id}: {tuple(tensor.shape)}")
            vector = tensor[0].float().mean(dim=0).cpu().numpy().astype(np.float32)
            if not np.isfinite(vector).all():
                raise RuntimeError(f"Non-finite embedding for {fasta_id}")
            if embedding_dim is None:
                embedding_dim = len(vector)
            elif len(vector) != embedding_dim:
                raise RuntimeError("Embedding dimension changed between sequences")
            pooled.append(vector)
        vectors.append(pooled[1] - pooled[0])
        if i % 100 == 0 or i == len(index_rows):
            elapsed = time.time() - started
            print(f"Embedded {i}/{len(index_rows)} variants in {elapsed:.1f}s", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(vectors, dtype=np.float32)
    np.save(output / "variant_embedding_delta.npy", matrix)
    with (output / "embedded_variant_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    report = {
        "status": "FROZEN_EVO2_VARIANT_CONTEXT_EMBEDDINGS_COMPLETE",
        "model": args.model,
        "layer": args.layer,
        "pooling": "mean over layer token axis",
        "variant_embedding": "alternate-window pooled embedding minus reference-window pooled embedding",
        "n_variants": len(index_rows),
        "embedding_dimension": int(matrix.shape[1]),
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "input_fasta_sha256": sha256(fasta_path),
        "input_index_sha256": sha256(index_path),
        "outputs": ["variant_embedding_delta.npy", "embedded_variant_index.csv", "embedding_report.json"],
        "limitations": [
            "Embeddings use REL606 local sequence context with only the focal SNP substituted; other clone-specific variants are omitted.",
            "This is a frozen representation; Evo2 weights are not fine-tuned.",
            "A mean-pooled local-window difference is an exploratory variant-context feature, not a causal effect estimate.",
        ],
    }
    (output / "embedding_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
