#!/usr/bin/env python3
"""Train frozen-Evo2 temporal heads from precomputed feature chunks.

Runs on CPU or CUDA. Never loads Evo2 or touches the test split.
"""
import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss

BASES = 'ACGT'
BASE_TO_ID = {b: i for i, b in enumerate(BASES)}


def load_split(root, split):
    files = sorted((root / split).glob('chunk_*.pt'))
    if not files:
        raise FileNotFoundError(f'No chunks found for {split}: {root / split}')
    xs, metadata = [], []
    expected_start = 0
    for path in files:
        d = torch.load(path, map_location='cpu', weights_only=False)
        if d.get('split') != split or d.get('layer') != 'blocks.15':
            raise ValueError(f'Wrong split/layer in {path}')
        if 'start_index' in d and int(d['start_index']) != expected_start:
            raise ValueError(f'Non-contiguous start_index in {path}: {d["start_index"]} != {expected_start}')
        x = d['embeddings'].float()
        m = d['metadata']
        if x.ndim != 2 or x.shape[1] != 4096 or x.shape[0] != len(m):
            raise ValueError(f'Invalid embeddings/metadata in {path}: {tuple(x.shape)}, {len(m)}')
        if not torch.isfinite(x).all():
            raise ValueError(f'NaN/Inf in {path}')
        if any(z.get('split') != split for z in m):
            raise ValueError(f'Metadata split mismatch in {path}')
        xs.append(x)
        metadata.extend(m)
        expected_start += len(m)
    if len({z['sample_id'] for z in metadata}) != len(metadata):
        raise ValueError(f'Duplicate sample_id within {split}')
    parents = torch.tensor([BASE_TO_ID[z['parent_base']] for z in metadata], dtype=torch.long)
    future = torch.tensor([BASE_TO_ID[z['future_base']] for z in metadata], dtype=torch.long)
    mutation = torch.tensor([int(z['is_mutated']) for z in metadata], dtype=torch.long)
    if not torch.equal((parents != future).long(), mutation):
        raise ValueError(f'Mutation labels inconsistent in {split}')
    times = torch.tensor([float(z['delta_time']) for z in metadata], dtype=torch.float32)
    if not torch.isfinite(times).all() or (times <= 0).any():
        raise ValueError(f'Invalid delta_time in {split}')
    return torch.cat(xs), parents, future, times, metadata


class Head(nn.Module):
    def __init__(self, with_time, hidden=128, dropout=0.3):
        super().__init__()
        self.with_time = with_time
        self.net = nn.Sequential(
            nn.Linear(4096 + 4 + int(with_time), hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 4),
        )

    def forward(self, x, parent, log_time):
        onehot = torch.nn.functional.one_hot(parent, 4).float()
        pieces = [x, onehot]
        if self.with_time:
            pieces.append(log_time[:, None])
        return self.net(torch.cat(pieces, dim=1))


def metrics(logits, parent, future):
    probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    y = future.cpu().numpy()
    p = parent.cpu().numpy()
    mutation = (p != y).astype(int)
    mutation_score = 1.0 - probs[np.arange(len(p)), p]
    out = {
        'n': len(p),
        'mutation_fraction': float(mutation.mean()),
        'nucleotide_accuracy': float((probs.argmax(1) == y).mean()),
        'nucleotide_log_loss': float(log_loss(y, probs, labels=[0, 1, 2, 3])),
        'mutation_auprc': float(average_precision_score(mutation, mutation_score)),
        'mutation_auroc': float(roc_auc_score(mutation, mutation_score)) if len(set(mutation)) == 2 else None,
        'mutation_brier_sampled': float(np.mean((mutation_score - mutation) ** 2)),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='results/temporal_head_v1_embeddings')
    ap.add_argument('--output-dir', default='results/temporal_head_v1_training')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=12)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    print('Device:', device, flush=True)
    train = load_split(Path(args.data_dir), 'train')
    val = load_split(Path(args.data_dir), 'validation')
    if set(z['sample_id'] for z in train[4]) & set(z['sample_id'] for z in val[4]):
        raise ValueError('Train/validation sample_id overlap')
    print('Train:', len(train[4]), 'Validation:', len(val[4]), flush=True)
    print('NOTE: evaluation uses mutation-enriched validation; not natural-prevalence calibration.', flush=True)
    # Normalize embeddings using train statistics only.
    mean = train[0].mean(0)
    std = train[0].std(0).clamp_min(1e-4)
    train_x = ((train[0] - mean) / std).clamp(-10, 10)
    val_x = ((val[0] - mean) / std).clamp(-10, 10)
    # Log-time scaling fixed a priori; no validation/test fitting.
    train_t = torch.log1p(train[3]) / math.log1p(50000.0)
    val_t = torch.log1p(val[3]) / math.log1p(50000.0)
    train_data = TensorDataset(train_x, train[1], train[2], train_t)
    val_data = TensorDataset(val_x, val[1], val[2], val_t)
    results = {}
    for name, with_time in [('no_time', False), ('temporal', True)]:
        torch.manual_seed(args.seed)
        model = Head(with_time).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        vloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
        best_loss = float('inf')
        best_state = None
        best_epoch = 0
        history = []
        print(f'\n=== {name} ===', flush=True)
        for epoch in range(1, args.epochs + 1):
            model.train()
            total, count = 0., 0
            for x, parent, future, t in loader:
                x, parent, future, t = [v.to(device) for v in (x, parent, future, t)]
                optimizer.zero_grad(set_to_none=True)
                logits = model(x, parent, t)
                loss = criterion(logits, future)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(future)
                count += len(future)
            model.eval()
            val_logits = []
            with torch.inference_mode():
                for x, parent, future, t in vloader:
                    val_logits.append(model(x.to(device), parent.to(device), t.to(device)).cpu())
            val_logits = torch.cat(val_logits)
            val_loss = float(criterion(val_logits, val[2]).item())
            history.append({'epoch': epoch, 'train_loss': total/count, 'validation_loss': val_loss})
            print(f'Epoch {epoch:03d} train={total/count:.5f} val={val_loss:.5f}', flush=True)
            if val_loss < best_loss - 1e-5:
                best_loss, best_epoch = val_loss, epoch
                best_state = copy.deepcopy(model.state_dict())
            elif epoch - best_epoch >= args.patience:
                print('Early stopping', flush=True)
                break
        model.load_state_dict(best_state)
        model.eval()
        with torch.inference_mode():
            logits = torch.cat([model(x.to(device), p.to(device), t.to(device)).cpu()
                                for x, p, y, t in vloader])
        score = metrics(logits, val[1], val[2])
        score.update({'best_epoch': best_epoch, 'best_validation_loss': best_loss})
        print('Validation metrics:', json.dumps(score, indent=2), flush=True)
        torch.save({'model_state_dict': best_state, 'with_time': with_time,
                    'embedding_layer': 'blocks.15', 'embedding_mean': mean,
                    'embedding_std': std, 'config': vars(args), 'validation_metrics': score},
                   outdir / f'{name}_best.pt')
        (outdir / f'{name}_history.json').write_text(json.dumps(history, indent=2))
        results[name] = score
    # Identity baseline: always predict the current parent base.
    identity_accuracy = float((val[1] == val[2]).float().mean())
    results['identity_baseline'] = {'nucleotide_accuracy': identity_accuracy,
                                    'mutation_auprc': float((val[1] != val[2]).float().mean()),
                                    'note': 'Constant mutation score; deterministic nucleotide probabilities have undefined/infinite log loss for mutations.'}
    (outdir / 'validation_comparison.json').write_text(json.dumps(results, indent=2))
    print('\n=== VALIDATION COMPARISON ===', flush=True)
    print(json.dumps(results, indent=2), flush=True)
    print('Saved to:', outdir, flush=True)


if __name__ == '__main__':
    main()
