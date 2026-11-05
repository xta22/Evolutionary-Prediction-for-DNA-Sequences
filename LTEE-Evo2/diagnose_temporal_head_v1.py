#!/usr/bin/env python3
"""Validation-only diagnostics for trained Evo2 temporal heads (CPU, no Evo2 load)."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             confusion_matrix, log_loss)
from train_temporal_head_v1 import Head, load_split, BASES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='results/temporal_head_v1_embeddings')
    parser.add_argument('--model-dir', default='results/temporal_head_v1_training')
    parser.add_argument('--output-dir', default='results/temporal_head_v1_diagnostics')
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    features, parent, future, delta, metadata = load_split(Path(args.data_dir), 'validation')
    assert len(metadata) == 3234 or len(metadata) > 0
    print('Validation samples:', len(metadata))
    print('NOTE: mutation-enriched validation; scores are not natural mutation probabilities.')
    p = parent.numpy()
    y = future.numpy()
    true_mut = (p != y).astype(int)
    log_time = torch.log1p(delta) / math.log1p(50000.0)
    base = pd.DataFrame([{k: v for k, v in m.items() if k != 'sequence'} for m in metadata])
    base['true_mutation'] = true_mut
    base['identity_correct'] = (p == y).astype(int)
    summary = {}

    for name, with_time in [('no_time', False), ('temporal', True)]:
        ckpt_path = Path(args.model_dir) / f'{name}_best.pt'
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if bool(ckpt['with_time']) != with_time:
            raise ValueError(f'Checkpoint model type mismatch: {ckpt_path}')
        if ckpt.get('embedding_layer') != 'blocks.15':
            raise ValueError('Unexpected embedding layer')
        mean, std = ckpt['embedding_mean'], ckpt['embedding_std']
        x = ((features - mean) / std.clamp_min(1e-4)).clamp(-10, 10)
        model = Head(with_time).eval()
        model.load_state_dict(ckpt['model_state_dict'])
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(x), args.batch_size):
                stop = start + args.batch_size
                logits = model(x[start:stop], parent[start:stop], log_time[start:stop])
                outputs.append(torch.softmax(logits.float(), dim=-1).numpy())
        probs = np.concatenate(outputs)
        pred = probs.argmax(axis=1)
        mutation_prob = 1.0 - probs[np.arange(len(p)), p]
        predicted_mut = (pred != p).astype(int)
        correct = (pred == y).astype(int)
        print(f'\n=== {name.upper()} ===')
        print('Observed mutation fraction:', true_mut.mean())
        print('Predicted mutation fraction (argmax):', predicted_mut.mean())
        print('Mean predicted mutation score:', mutation_prob.mean())
        print('Nucleotide accuracy:', correct.mean())
        print('Accuracy on mutated sites:', correct[true_mut == 1].mean())
        print('Accuracy on unchanged sites:', correct[true_mut == 0].mean())
        print('Mutation AUROC:', roc_auc_score(true_mut, mutation_prob))
        print('Mutation AUPRC:', average_precision_score(true_mut, mutation_prob))
        print('Nucleotide log loss:', log_loss(y, probs, labels=[0, 1, 2, 3]))
        print('Future nucleotide confusion matrix (rows=true, columns=predicted):')
        print(pd.DataFrame(confusion_matrix(y, pred, labels=range(4)), index=list(BASES), columns=list(BASES)).to_string())
        for j, b in enumerate(BASES):
            base[f'{name}_p_{b}'] = probs[:, j]
        base[f'{name}_mutation_score'] = mutation_prob
        base[f'{name}_predicted_base'] = [BASES[z] for z in pred]
        base[f'{name}_predicted_mutation'] = predicted_mut
        base[f'{name}_correct'] = correct
        summary[name] = {
            'observed_mutation_fraction': float(true_mut.mean()),
            'predicted_mutation_fraction_argmax': float(predicted_mut.mean()),
            'mean_predicted_mutation_score': float(mutation_prob.mean()),
            'nucleotide_accuracy': float(correct.mean()),
            'accuracy_mutated_sites': float(correct[true_mut == 1].mean()),
            'accuracy_unchanged_sites': float(correct[true_mut == 0].mean()),
            'mutation_auroc': float(roc_auc_score(true_mut, mutation_prob)),
            'mutation_auprc': float(average_precision_score(true_mut, mutation_prob)),
            'nucleotide_log_loss': float(log_loss(y, probs, labels=[0, 1, 2, 3])),
        }

    # Time-stratified diagnostic: association is descriptive, not proof of time use.
    base['time_bin'] = pd.qcut(base['delta_time'], q=4, duplicates='drop')
    time_table = base.groupby('time_bin', observed=True).agg(
        n=('true_mutation', 'size'),
        median_delta_time=('delta_time', 'median'),
        observed_mutation_fraction=('true_mutation', 'mean'),
        no_time_mean_mutation_score=('no_time_mutation_score', 'mean'),
        temporal_mean_mutation_score=('temporal_mutation_score', 'mean'),
        no_time_accuracy=('no_time_correct', 'mean'),
        temporal_accuracy=('temporal_correct', 'mean'),
    )
    print('\n=== BY EVOLUTIONARY TIME QUARTILE ===')
    print(time_table.to_string())
    time_table.to_csv(out / 'time_quartiles.csv')

    # Counterfactual sensitivity: same embeddings and parent, varying only time.
    ckpt = torch.load(Path(args.model_dir) / 'temporal_best.pt', map_location='cpu', weights_only=False)
    model = Head(True).eval()
    model.load_state_dict(ckpt['model_state_dict'])
    x = ((features - ckpt['embedding_mean']) / ckpt['embedding_std'].clamp_min(1e-4)).clamp(-10, 10)
    time_sensitivity = {}
    with torch.inference_mode():
        for generations in [125, 1000, 10000, 30000]:
            fixed_t = torch.full((len(x),), math.log1p(generations) / math.log1p(50000.0))
            probs = []
            for start in range(0, len(x), args.batch_size):
                stop = start + args.batch_size
                logits = model(x[start:stop], parent[start:stop], fixed_t[start:stop])
                probs.append(torch.softmax(logits.float(), dim=-1).numpy())
            probs = np.concatenate(probs)
            scores = 1 - probs[np.arange(len(p)), p]
            time_sensitivity[str(generations)] = scores
    for generations, scores in time_sensitivity.items():
        base[f'counterfactual_mutation_score_t{generations}'] = scores
    print('\n=== TIME COUNTERFACTUAL (same DNA features; only delta_time changed) ===')
    for generations, scores in time_sensitivity.items():
        print(f'{generations:>6} generations: mean predicted mutation score = {scores.mean():.6f}')
    delta_scores = time_sensitivity['30000'] - time_sensitivity['125']
    print('Mean score change (30000 - 125):', delta_scores.mean())
    print('Median absolute per-site change:', np.median(np.abs(delta_scores)))
    summary['counterfactual'] = {
        'mean_score_by_generations': {k: float(v.mean()) for k, v in time_sensitivity.items()},
        'mean_change_30000_vs_125': float(delta_scores.mean()),
        'median_absolute_change_30000_vs_125': float(np.median(np.abs(delta_scores))),
    }

    base.to_csv(out / 'validation_predictions.csv', index=False)
    (out / 'diagnostic_summary.json').write_text(json.dumps(summary, indent=2))
    print('\nSaved:', out / 'validation_predictions.csv')
    print('Saved:', out / 'time_quartiles.csv')
    print('Saved:', out / 'diagnostic_summary.json')
    print('DIAGNOSTIC COMPLETE')


if __name__ == '__main__':
    main()
