#!/usr/bin/env python3
"""Extract frozen Evo2 position-level features for LTEE temporal head V1."""
import argparse
import json
from pathlib import Path
import torch
from evo2 import Evo2

DNA = set('ACGT')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=['train','validation','test'], required=True)
    p.add_argument('--input-dir', default='LTEE_temporal_head_v1')
    p.add_argument('--output-dir', default='results/temporal_head_v1_embeddings')
    p.add_argument('--layer', default='blocks.15')
    p.add_argument('--chunk-size', type=int, default=100)
    args = p.parse_args()
    if args.chunk_size < 1:
        raise ValueError('chunk-size must be >= 1')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable; run via Slurm on GPU node')

    source = Path(args.input_dir) / (args.split + '.jsonl')
    if not source.is_file():
        raise FileNotFoundError(source)
    out = Path(args.output_dir) / args.split
    out.mkdir(parents=True, exist_ok=True)
    samples = [json.loads(s) for s in source.open() if s.strip()]
    if not samples:
        raise ValueError('Empty input')
    n_chunks = (len(samples) + args.chunk_size - 1) // args.chunk_size
    print(f'Split={args.split}, samples={len(samples)}, chunks={n_chunks}', flush=True)
    print(f'Layer={args.layer}', flush=True)

    model = Evo2('evo2_7b_base')
    model.model.get_submodule(args.layer)
    print('Evo2 loaded', flush=True)

    for chunk_id in range(n_chunks):
        lo = chunk_id * args.chunk_size
        hi = min(len(samples), lo + args.chunk_size)
        path = out / f'chunk_{chunk_id:05d}.pt'
        if path.exists():
            existing = torch.load(path, map_location='cpu', weights_only=True)
            if (existing.get('split') != args.split or
                existing.get('layer') != args.layer or
                existing.get('start_index') != lo or
                len(existing.get('metadata', [])) != hi - lo):
                raise RuntimeError(f'Existing chunk mismatch: {path}')
            print(f'Skipping existing {path}', flush=True)
            continue
        vectors = []
        metadata = []
        for i in range(lo, hi):
            row = samples[i]
            seq = row['sequence'].upper()
            idx = int(row['target_index'])
            if not 0 <= idx < len(seq):
                raise ValueError(f'Bad index for {row.get("sample_id", i)}')
            if seq[idx] != row['parent_base'] or row['future_base'] not in DNA:
                raise ValueError(f'Sequence/label mismatch for {row.get("sample_id", i)}')
            tokens = model.tokenizer.tokenize(seq)
            if len(tokens) != len(seq):
                raise ValueError(f'Token length mismatch at {i}')
            input_ids = torch.tensor(tokens, dtype=torch.long, device='cuda').unsqueeze(0)
            with torch.inference_mode():
                _, embs = model(input_ids, return_embeddings=True, layer_names=[args.layer])
                hidden = embs[args.layer]
                if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] != len(seq):
                    raise RuntimeError(f'Unexpected hidden shape {tuple(hidden.shape)}')
                vector = hidden[0, idx, :].to(dtype=torch.float16, device='cpu').clone()
            vectors.append(vector)
            metadata.append({k: v for k, v in row.items() if k != 'sequence'})
            del input_ids, hidden, embs
        payload = {
            'split': args.split, 'layer': args.layer, 'start_index': lo,
            'metadata': metadata, 'embeddings': torch.stack(vectors)
        }
        temp = path.with_suffix('.tmp')
        torch.save(payload, temp)
        temp.replace(path)
        print(f'Saved {path}: {hi}/{len(samples)}', flush=True)
    print('EXTRACTION COMPLETE', flush=True)
    print('Peak allocated GPU memory GiB:', round(torch.cuda.max_memory_allocated()/1024**3, 2), flush=True)

if __name__ == '__main__':
    main()

