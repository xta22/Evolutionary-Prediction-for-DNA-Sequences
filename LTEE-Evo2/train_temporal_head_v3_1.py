#!/usr/bin/env python3
"""V3.1: branch-count-regularized, population/time-aware CTMC.

Train-only branch holdout for within-training-era diagnostics; chronological
validation is evaluated, never used for gradient updates or model selection.
This is a sensitivity experiment, not a calibrated biological mutation model.
Requires train_temporal_head_v3.py and train_temporal_head_v2.py nearby.
"""
import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from train_temporal_head_v2 import load_split
from train_temporal_head_v3 import StageCTMC, BASE_RATE, load_weights, load_times

GENOME_LENGTH = 4629812.0


def make_branch_data(tab, branch_ids, populations):
    d = tab.loc[list(branch_ids)].copy()
    if d[['parent_time', 'child_time', 'actual_mutation_count']].isna().any().any():
        raise ValueError('Missing branch time/count fields')
    start = torch.tensor(d.parent_time.to_numpy(dtype=float), dtype=torch.float64)
    end = torch.tensor(d.child_time.to_numpy(dtype=float), dtype=torch.float64)
    pop = torch.tensor([populations.index(v) for v in d.population], dtype=torch.long)
    counts = torch.tensor(d.actual_mutation_count.to_numpy(dtype=float), dtype=torch.float64)
    if (end <= start).any() or (counts < 0).any():
        raise ValueError('Invalid branch intervals/counts')
    return start, end, pop, counts


def expected_branch_counts(model, branch):
    """Shared population/time expected counts, with local rate multiplier=1.

    This auxiliary model is an approximation: it does NOT integrate the
    sequence-dependent rate multiplier over all genome positions.
    """
    start, end, pop, _ = branch
    mid = (start[:, None] + end[:, None]) / 2
    half = (end - start)[:, None] / 2
    t = mid + half * model.nodes[None, :]
    integral = half.squeeze(1) * (model.quad_weights[None, :] *
                                torch.exp(model.stage_log(t))).sum(1)
    # Q0 has mean outgoing rate 1, so BASE_RATE is per-site outgoing rate.
    log_factor = model.log_global + model.pop[pop]
    return (GENOME_LENGTH * BASE_RATE * torch.exp(log_factor) * integral).clamp_min(1e-12)


def count_deviance(counts, mu):
    """Mean Poisson deviance / 2, rescaled by average count for optimization."""
    terms = torch.where(counts > 0, counts * torch.log(counts.clamp_min(1e-12) / mu),
                        torch.zeros_like(counts)) - (counts - mu)
    return terms.mean() / counts.mean().clamp_min(1.0)


def subset_data(d, ids):
    ix = torch.tensor([j for j, m in enumerate(d['metadata']) if m['branch_id'] in ids], dtype=torch.long)
    if len(ix) == 0: raise ValueError('Empty site subset')
    return {k: (v[ix] if isinstance(v, torch.Tensor) else [v[j] for j in ix.tolist()])
            for k, v in d.items()}


def predict(model, d):
    model.eval()
    with torch.no_grad():
        probs, _ = model(d['x'], d['parent'], d['start'], d['end'], d['pop'])
    return probs


def metrics(model, d):
    p = predict(model, d).numpy()
    a = d['parent'].numpy(); y = d['future'].numpy()
    w = d['weight'].numpy(); w = w / w.sum()
    changed = (a != y).astype(int)
    ix = np.arange(len(y))
    score = 1 - p[ix, a]
    nll = -np.log(np.clip(p[ix, y], 1e-300, 1))
    result = {
        'n': len(y), 'n_branches': len({m['branch_id'] for m in d['metadata']}),
        'sampled_change_fraction': float(changed.mean()),
        'weighted_change_fraction': float(np.dot(w, changed)),
        'predicted_change_weighted': float(np.dot(w, score)),
        'weighted_log_loss': float(np.dot(w, nll)),
        'sampled_log_loss': float(nll.mean()),
        'nucleotide_accuracy': float((p.argmax(1) == y).mean())
    }
    if changed.min() != changed.max():
        result['sampled_mutation_auroc'] = float(roc_auc_score(changed, score))
        result['sampled_mutation_auprc'] = float(average_precision_score(changed, score))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--embedding-dir', default='results/temporal_head_v1_embeddings')
    ap.add_argument('--weights-dir', default='LTEE_temporal_head_v1')
    ap.add_argument('--metadata', default='LTEE_temporal_head_v3/branch_time_metadata.csv')
    ap.add_argument('--output-dir', default='results/temporal_head_v3_1_training')
    ap.add_argument('--variants', nargs='+', choices=['A', 'B', 'C'], default=['A', 'B', 'C'])
    ap.add_argument('--epochs', type=int, default=180)
    ap.add_argument('--patience', type=int, default=25)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--holdout-fraction', type=float, default=0.2)
    ap.add_argument('--count-weight', type=float, default=1.0,
                    help='Weight on normalized Poisson branch-count deviance')
    ap.add_argument('--site-weight', type=float, default=1.0,
                    help='Weight on site-level weighted NLL divided by initial NLL')
    ap.add_argument('--lr-regime', type=float, default=0.02)
    ap.add_argument('--lr-sequence', type=float, default=0.001)
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--regularization', type=float, default=0.0001)
    args = ap.parse_args()
    if not 0 < args.holdout_fraction < 0.5: raise ValueError('Invalid holdout fraction')
    torch.set_num_threads(4)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    tab = pd.read_csv(args.metadata)
    required = {'branch_id', 'split', 'population', 'parent_time', 'child_time', 'actual_mutation_count'}
    if not required.issubset(tab.columns) or tab.branch_id.duplicated().any():
        raise ValueError('Invalid branch metadata')
    tab = tab.set_index('branch_id')
    pops = sorted(tab.loc[tab['split'] == 'train', 'population'].unique())
    raw_train = load_split(Path(args.embedding_dir), 'train')
    raw_val = load_split(Path(args.embedding_dir), 'validation')
    if set(m['sample_id'] for m in raw_train[4]) & set(m['sample_id'] for m in raw_val[4]):
        raise ValueError('Train/validation sample overlap')
    mu_x = raw_train[0].mean(0); sd_x = raw_train[0].std(0).clamp_min(1e-4)

    def pack(raw, split):
        s, e, pop = load_times(raw[4], tab, split, pops)
        w = load_weights(raw[4], Path(args.weights_dir) / (split + '_weights.csv'), split)
        return {'x': ((raw[0] - mu_x) / sd_x).clamp(-10, 10),
                'parent': raw[1], 'future': raw[2], 'start': s, 'end': e,
                'pop': pop, 'weight': w, 'metadata': raw[4]}

    train_all = pack(raw_train, 'train'); val = pack(raw_val, 'validation')
    # Split at branch level within the original TRAIN set, stratified by population.
    rng = np.random.default_rng(args.seed)
    fit_ids, hold_ids = set(), set()
    for p in pops:
        ids = sorted({m['branch_id'] for m in train_all['metadata'] if m['population'] == p})
        rng.shuffle(ids)
        n_hold = max(1, int(round(len(ids) * args.holdout_fraction)))
        hold_ids.update(ids[:n_hold]); fit_ids.update(ids[n_hold:])
    if fit_ids & hold_ids: raise AssertionError('Branch holdout leakage')
    fit = subset_data(train_all, fit_ids)
    hold = subset_data(train_all, hold_ids)
    fit_br = make_branch_data(tab, sorted(fit_ids), pops)
    hold_br = make_branch_data(tab, sorted(hold_ids), pops)
    val_ids = sorted({m['branch_id'] for m in val['metadata']})
    val_br = make_branch_data(tab, val_ids, pops)
    print('TRAIN branches for fitting:', len(fit_ids), 'TRAIN-era held-out branches:', len(hold_ids),
          'Chronological validation branches:', len(val_ids), flush=True)
    print('No test split accessed. Held-out training branches excluded from gradient updates.', flush=True)
    print('CAVEAT: branch holdout does not remove phylogenetic relatedness.', flush=True)
    print('CAVEAT: auxiliary branch-count model approximates genome-wide context average.', flush=True)
    print('CAVEAT: stage function plateaus beyond 10,000 generations.', flush=True)

    results = {}
    for variant in args.variants:
        torch.manual_seed(args.seed)
        model = StageCTMC(variant, len(pops), args.hidden, args.dropout, use_context=True)
        regime = [model.pop, model.log_global, model.raw_q]
        if model.linear is not None: regime.append(model.linear)
        if model.nonlinear is not None: regime.append(model.nonlinear)
        opt = torch.optim.AdamW([
            {'params': regime, 'lr': args.lr_regime},
            {'params': model.net.parameters(), 'lr': args.lr_sequence}
        ], weight_decay=0)
        with torch.no_grad():
            initial_prob, _ = model(fit['x'], fit['parent'], fit['start'], fit['end'], fit['pop'])
            ix = torch.arange(len(fit['parent']))
            initial_site_loss = float((fit['weight'] * -torch.log(initial_prob[ix, fit['future']])).sum())
        scale = max(initial_site_loss, 1e-8)
        best = math.inf; best_epoch = 0; best_state = None; history = []
        print('\n=== V3.1-' + variant + ' === initial site NLL:', initial_site_loss, flush=True)
        for epoch in range(1, args.epochs + 1):
            model.train(); opt.zero_grad(set_to_none=True)
            prob, local = model(fit['x'], fit['parent'], fit['start'], fit['end'], fit['pop'])
            nll = -torch.log(prob[torch.arange(len(fit['parent'])), fit['future']])
            site_loss = (fit['weight'] * nll).sum() / fit['weight'].sum()
            count_loss = count_deviance(fit_br[3], expected_branch_counts(model, fit_br))
            reg = args.regularization * (model.pop.square().mean() + model.log_global.square())
            if model.linear is not None: reg = reg + args.regularization * model.linear.square()
            if model.nonlinear is not None: reg = reg + args.regularization * model.nonlinear.square().mean()
            reg = reg + args.regularization * local.square().mean()
            objective = args.site_weight * site_loss / scale + args.count_weight * count_loss + reg
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            opt.step()
            model.eval()
            with torch.no_grad():
                hp = predict(model, hold)
                hold_nll = -torch.log(hp[torch.arange(len(hold['parent'])), hold['future']])
                held_site = float((hold['weight'] * hold_nll).sum() / hold['weight'].sum())
                held_count = float(count_deviance(hold_br[3], expected_branch_counts(model, hold_br)))
                selection = args.site_weight * held_site / scale + args.count_weight * held_count
            history.append({'epoch': epoch, 'train_site_nll': float(site_loss.detach()),
                            'train_count_deviance_scaled': float(count_loss.detach()),
                            'heldout_site_nll': held_site, 'heldout_count_deviance_scaled': held_count,
                            'heldout_selection': selection})
            if epoch == 1 or epoch % 10 == 0:
                print(f'Epoch {epoch:03d} train_site={float(site_loss):.5g} '
                      f'train_count={float(count_loss):.5g} hold_site={held_site:.5g} '
                      f'hold_count={held_count:.5g}', flush=True)
            if selection < best - 1e-6:
                best = selection; best_epoch = epoch; best_state = copy.deepcopy(model.state_dict())
            elif epoch - best_epoch >= args.patience:
                print('Early stopping', flush=True); break
        model.load_state_dict(best_state)
        result = {'variant': variant, 'best_epoch': best_epoch,
                  'train_era_branch_holdout': metrics(model, hold),
                  'chronological_validation': metrics(model, val),
                  'heldout_branch_count_deviance_scaled': float(count_deviance(hold_br[3], expected_branch_counts(model, hold_br))),
                  'validation_branch_count_deviance_scaled': float(count_deviance(val_br[3], expected_branch_counts(model, val_br))),
                  'population_rate_multipliers': {p: math.exp(float(v)) for p, v in zip(pops, model.pop.detach())},
                  'linear_coefficient': None if model.linear is None else float(model.linear.detach()),
                  'nonlinear_coefficients': None if model.nonlinear is None else model.nonlinear.detach().tolist()}
        results[variant] = result
        print('RESULT', json.dumps(result, indent=2), flush=True)
        torch.save({'state_dict': best_state, 'variant': variant, 'populations': pops,
                    'embedding_mean': mu_x, 'embedding_std': sd_x, 'config': vars(args),
                    'base_rate': BASE_RATE, 'results': result}, out / f'v3_1_{variant}_best.pt')
        (out / f'v3_1_{variant}_history.json').write_text(json.dumps(history, indent=2))
    (out / 'validation_comparison.json').write_text(json.dumps(results, indent=2))
    print('\n=== V3.1 VALIDATION COMPARISON ===', flush=True)
    print(json.dumps(results, indent=2), flush=True)
    print('Saved:', out, flush=True)

if __name__ == '__main__':
    main()
