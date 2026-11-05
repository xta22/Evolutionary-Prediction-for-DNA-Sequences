#!/usr/bin/env python3
"""V2: frozen Evo2 embeddings + context-scaled CTMC nucleotide head.

Train/validation only; does not touch test. Sample-enriched outputs are NOT
calibrated genome-wide mutation probabilities or biochemical mutation rates.
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
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss

BASES = 'ACGT'
BASE_TO_ID = {b: i for i, b in enumerate(BASES)}


def load_split(root, split):
    files = sorted((root / split).glob('chunk_*.pt'))
    if not files:
        raise FileNotFoundError(f'No chunks found: {root / split}')
    xs, metadata = [], []
    expected_start = 0
    for path in files:
        d = torch.load(path, map_location='cpu', weights_only=False)
        if d.get('split') != split or d.get('layer') != 'blocks.15':
            raise ValueError(f'Wrong split/layer: {path}')
        if int(d.get('start_index', -1)) != expected_start:
            raise ValueError(f'Noncontiguous chunk: {path}')
        x = d['embeddings'].float()
        m = d['metadata']
        if x.ndim != 2 or x.shape[1] != 4096 or x.shape[0] != len(m):
            raise ValueError(f'Bad shape/metadata: {path}')
        if not torch.isfinite(x).all():
            raise ValueError(f'Nonfinite embeddings: {path}')
        if any(z.get('split') != split for z in m):
            raise ValueError(f'Split mismatch: {path}')
        xs.append(x)
        metadata.extend(m)
        expected_start += len(m)
    ids = [m['sample_id'] for m in metadata]
    if len(set(ids)) != len(ids):
        raise ValueError(f'Duplicate IDs in {split}')
    parent = torch.tensor([BASE_TO_ID[m['parent_base']] for m in metadata], dtype=torch.long)
    future = torch.tensor([BASE_TO_ID[m['future_base']] for m in metadata], dtype=torch.long)
    labels = torch.tensor([int(m['is_mutated']) for m in metadata])
    if not torch.equal((parent != future).long(), labels):
        raise ValueError(f'Labels mismatch: {split}')
    time = torch.tensor([float(m['delta_time']) for m in metadata], dtype=torch.float32)
    if not torch.isfinite(time).all() or (time <= 0).any():
        raise ValueError(f'Invalid time in {split}')
    return torch.cat(xs), parent, future, time, metadata


class CTMCHead(nn.Module):
    """Shared Q0 and per-site positive scalar rate multiplier.

    Q0 rows sum to zero, off-diagonal rates >= 0. Rate is per generation.
    For each site: P(t) = exp(t * r(x) * Q0).
    """
    def __init__(self, hidden=32, dropout=0.5, use_context=True):
        super().__init__()
        self.use_context = use_context
        self.net = (nn.Sequential(
            nn.Linear(4096 + 4, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        ) if use_context else None)
        # Start with equal off-diagonal rates, 1e-5 per generation.
        self.raw_offdiag = nn.Parameter(torch.full((4, 4), math.log(math.expm1(1e-5))))
        mask = torch.ones(4, 4) - torch.eye(4)
        self.register_buffer('offdiag_mask', mask)
        if self.net is not None:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, parent, delta_time):
        onehot = F.one_hot(parent, 4).float()
        if self.net is None:
            multiplier = torch.ones_like(delta_time)
        else:
            log_multiplier = self.net(torch.cat([x, onehot], dim=-1)).squeeze(-1)
            # Numerical stability; regularization also discourages extreme rates.
            multiplier = torch.exp(log_multiplier.clamp(-5, 5))
        offdiag = F.softplus(self.raw_offdiag) * self.offdiag_mask
        Q = offdiag - torch.diag(offdiag.sum(dim=-1))
        # Matrix exponential computes full nucleotide transition distribution.
        transition = torch.matrix_exp((delta_time * multiplier)[:, None, None] * Q)
        rows = transition[torch.arange(len(parent), device=parent.device), parent]
        return rows.clamp_min(1e-12), multiplier


def evaluate(model, data, device, batch_size):
    model.eval()
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    all_probs, all_parents, all_future, all_times = [], [], [], []
    with torch.inference_mode():
        for x, p, y, t in loader:
            probs, _ = model(x.to(device), p.to(device), t.to(device))
            all_probs.append(probs.cpu())
            all_parents.append(p)
            all_future.append(y)
            all_times.append(t)
    probs = torch.cat(all_probs).numpy()
    parents = torch.cat(all_parents).numpy()
    future = torch.cat(all_future).numpy()
    mutation = (parents != future).astype(int)
    score = 1.0 - probs[np.arange(len(parents)), parents]
    out = {
        'n': len(future),
        'mutation_fraction_sampled': float(mutation.mean()),
        'nucleotide_accuracy': float((probs.argmax(1) == future).mean()),
        'nucleotide_log_loss': float(log_loss(future, probs, labels=[0, 1, 2, 3])),
        'mutation_auroc': float(roc_auc_score(mutation, score)) if len(set(mutation)) == 2 else None,
        'mutation_auprc': float(average_precision_score(mutation, score)),
        'mutation_site_accuracy': float((probs.argmax(1)[mutation == 1] == future[mutation == 1]).mean()),
        'unchanged_site_accuracy': float((probs.argmax(1)[mutation == 0] == future[mutation == 0]).mean()),
        'mean_predicted_change_probability_sampled': float(score.mean()),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='results/temporal_head_v1_embeddings')
    ap.add_argument('--output-dir', default='results/temporal_head_v2_training')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=12)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=0.05)
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--rate-regularization', type=float, default=1e-3)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    train = load_split(Path(args.data_dir), 'train')
    val = load_split(Path(args.data_dir), 'validation')
    if set(m['sample_id'] for m in train[4]) & set(m['sample_id'] for m in val[4]):
        raise ValueError('Train/validation sample overlap')
    mean = train[0].mean(0)
    std = train[0].std(0).clamp_min(1e-4)
    train_x = ((train[0] - mean) / std).clamp(-10, 10)
    val_x = ((val[0] - mean) / std).clamp(-10, 10)
    train_ds = TensorDataset(train_x, train[1], train[2], train[3])
    val_ds = TensorDataset(val_x, val[1], val[2], val[3])
    print('Device:', device, 'Train:', len(train[4]), 'Validation:', len(val[4]), flush=True)
    print('IMPORTANT: mutation-enriched sample; rates/probabilities are NOT calibrated to genome-wide prevalence.', flush=True)
    results = {}
    for name, use_context in [('shared_ctmc', False), ('evo2_ctmc', True)]:
        torch.manual_seed(args.seed)
        model = CTMCHead(hidden=args.hidden, dropout=args.dropout, use_context=use_context).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        best_loss, best_epoch, best_state = float('inf'), 0, None
        history = []
        print(f'\n=== {name} ===', flush=True)
        for epoch in range(1, args.epochs + 1):
            model.train()
            total, n = 0., 0
            for x, p, y, t in loader:
                x, p, y, t = [z.to(device) for z in (x, p, y, t)]
                optimizer.zero_grad(set_to_none=True)
                probs, multiplier = model(x, p, t)
                nll = -torch.log(probs[torch.arange(len(y), device=device), y]).mean()
                reg = (torch.log(multiplier).square().mean() * args.rate_regularization
                       if use_context else nll.new_zeros(()))
                loss = nll + reg
                loss.backward()
                optimizer.step()
                total += nll.item() * len(y)
                n += len(y)
            # Select by unregularized validation NLL.
            model.eval()
            with torch.inference_mode():
                vloss, vn = 0., 0
                for x, p, y, t in DataLoader(val_ds, batch_size=args.batch_size):
                    x, p, y, t = [z.to(device) for z in (x, p, y, t)]
                    probs, _ = model(x, p, t)
                    vloss += -torch.log(probs[torch.arange(len(y), device=device), y]).sum().item()
                    vn += len(y)
            vloss /= vn
            history.append({'epoch': epoch, 'train_nll': total / n, 'validation_nll': vloss})
            print(f'Epoch {epoch:03d} train={total/n:.5f} val={vloss:.5f}', flush=True)
            if vloss < best_loss - 1e-5:
                best_loss, best_epoch, best_state = vloss, epoch, copy.deepcopy(model.state_dict())
            elif epoch - best_epoch >= args.patience:
                print('Early stopping', flush=True)
                break
        model.load_state_dict(best_state)
        score = evaluate(model, val_ds, device, args.batch_size)
        score.update({'best_epoch': best_epoch, 'best_validation_loss': best_loss})
        results[name] = score
        print('Validation metrics:', json.dumps(score, indent=2), flush=True)
        torch.save({'model_state_dict': best_state, 'model_name': name,
                    'embedding_layer': 'blocks.15', 'embedding_mean': mean,
                    'embedding_std': std, 'config': vars(args),
                    'validation_metrics': score}, outdir / f'{name}_best.pt')
        (outdir / f'{name}_history.json').write_text(json.dumps(history, indent=2))
    (outdir / 'validation_comparison.json').write_text(json.dumps(results, indent=2))
    print('\n=== V2 VALIDATION COMPARISON ===', flush=True)
    print(json.dumps(results, indent=2), flush=True)
    print('Saved to:', outdir, flush=True)


if __name__ == '__main__':
    main()
