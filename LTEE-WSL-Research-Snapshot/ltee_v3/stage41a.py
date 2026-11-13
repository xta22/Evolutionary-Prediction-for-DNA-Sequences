#!/usr/bin/env python3
"""Stage 4.1A: matched historical pairing intervention, validation-only.

Imports the original stage4.py and does not modify it or refit parameters.
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import numpy as np


def load_production(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location('stage4_production', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ('load_folds', 'predict', 'score', 'frequencies', 'independent'):
        if not hasattr(mod, name):
            raise RuntimeError(f'Production stage4 missing {name}')
    return mod


def matched_shuffled_distribution(prod, row, alpha, rng):
    """Shuffle second allele within each pair; preserve both empirical marginals."""
    mat = row['latest_matrix']
    shuffled = np.empty_like(row['latest'])
    for k, (i, j) in enumerate(row['pairs']):
        a = mat[:, i]
        b = mat[rng.permutation(len(mat)), j]
        shuffled[k] = np.bincount(2 * a + b, minlength=4) / len(mat)
    base = prod.independent(row['latest'])
    q = prod.normalize((1 - alpha) * shuffled + alpha * base)
    return q, shuffled


def intervention(prod, row, alpha, rng, trend_strength=0.5):
    latest = row['latest']
    previous = row['previous']
    base = prod.independent(latest)
    ratio = row['horizon'] / row['history_interval'] if row['history_interval'] else 0.0
    original = prod.predict(latest, previous, 'M5_shrunk', alpha=alpha)
    independent = prod.predict(latest, previous, 'M1_independent')
    empirical = prod.predict(latest, previous, 'M5_empirical')
    shuffled, shuf_emp = matched_shuffled_distribution(prod, row, alpha, rng)
    dynamics = prod.predict(latest, previous, 'M5_dynamics', alpha=alpha,
                            trend=trend_strength, horizon_ratio=ratio)
    # Marginals of unnormalized empirical/shuffled distributions are exactly preserved.
    marginal_diff = float(np.max(np.abs(prod.independent(shuf_emp) - base)))
    assert marginal_diff < 1e-12, 'shuffle did not preserve marginals'
    return {'independent': independent, 'original_shrunk': original,
            'shuffled_matched': shuffled, 'empirical': empirical,
            'forced_dynamics': dynamics}, marginal_diff


def evaluate(prod, rows, alpha, permutations, seed, trend_strength):
    out = []
    per_rep = []
    for k, r in enumerate(rows):
        baseline_rng = np.random.default_rng(seed + k * 100003)
        preds, diff = intervention(prod, r, alpha, baseline_rng, trend_strength)
        scores = {name: prod.score(q, r['truth']) for name, q in preds.items()}
        q_orig = preds['original_shrunk']
        q_shuf = preds['shuffled_matched']
        row = dict(scenario=r['scenario'], seed=r['seed'], fold_id=r['fold_id'],
                   cutoff=r['cutoff'], target_generation=r['target_generation'],
                   n_pairs=len(r['pairs']), n_target_clones=r['n_target_clones'],
                   marginal_max_abs_difference=diff,
                   original_vs_shuffled_max_abs_probability=float(np.max(np.abs(q_orig-q_shuf))),
                   empirical_near_zero_fraction=float(np.mean(preds['empirical'] <= 1e-6)),
                   empirical_min_probability=float(np.min(preds['empirical'])))
        for name, score in scores.items():
            for metric, value in score.items():
                row[f'{name}_{metric}'] = value
        out.append(row)
        for rep in range(permutations):
            rng = np.random.default_rng(seed + k * 100003 + rep + 1)
            q, _ = matched_shuffled_distribution(prod, r, alpha, rng)
            per_rep.append(dict(scenario=r['scenario'], fold_id=r['fold_id'],
                                repetition=rep, original_nll=scores['original_shrunk']['joint_nll'],
                                shuffled_nll=prod.score(q, r['truth'])['joint_nll']))
    return out, per_rep


def write_csv(path, rows):
    if not rows:
        raise ValueError('No output rows')
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--original-results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--validation-seeds', nargs='+', type=int, default=[42])
    p.add_argument('--alpha', type=float, default=None)
    p.add_argument('--trend-strength', type=float, default=0.5)
    p.add_argument('--permutations', type=int, default=200)
    p.add_argument('--shuffle-seed', type=int, default=413)
    p.add_argument('--pair-mode', choices=['adjacent', 'all_adjacent'], default='adjacent')
    p.add_argument('--max-pairs', type=int, default=0)
    args = p.parse_args()
    if args.permutations < 1:
        p.error('permutations must be >= 1')
    if args.trend_strength < 0:
        p.error('trend strength must be >= 0')
    if not args.input.is_dir():
        p.error('input simulation directory missing')
    prod = load_production(args.source)
    selection = json.loads((args.original_results/'selected_models.json').read_text())
    selected_alpha = float(selection['M5_shrunk']['alpha'])
    alpha = selected_alpha if args.alpha is None else args.alpha
    if not 0 <= alpha <= 1:
        p.error('alpha must be in [0,1]')
    rows = prod.load_folds(args.input, args.validation_seeds, None, args.pair_mode, args.max_pairs)
    detail, permutations = evaluate(prod, rows, alpha, args.permutations, args.shuffle_seed, args.trend_strength)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output/'fold_diagnostics.csv', detail)
    write_csv(args.output/'permutation_diagnostics.csv', permutations)
    metrics = ['joint_nll','marginal_nll','double_mutant_mae','ld_D_mae','joint_brier']
    names = ['independent','original_shrunk','shuffled_matched','empirical','forced_dynamics']
    summary = {}
    for name in names:
        summary[name] = {metric: float(np.mean([r[f'{name}_{metric}'] for r in detail])) for metric in metrics}
    # Average NLL within folds first, then over folds; repetitions are not independent trajectories.
    shuffled_per_fold = {}
    for row in permutations:
        shuffled_per_fold.setdefault((row['scenario'], row['fold_id']), []).append(row['shuffled_nll'])
    rep_avg = np.mean([np.mean(v) for v in shuffled_per_fold.values()])
    original_nll = summary['original_shrunk']['joint_nll']
    report = {
        'status':'EXPLORATORY_VALIDATION_ONLY', 'n_folds':len(rows),
        'validation_seeds':args.validation_seeds, 'alpha_matched':alpha,
        'alpha_from_original_validation_selection':selected_alpha,
        'forced_trend_strength':args.trend_strength, 'permutations_per_fold':args.permutations,
        'summary':summary,
        'mean_permuted_nll':float(rep_avg),
        'mean_permuted_minus_original_nll':float(rep_avg-original_nll),
        'max_marginal_difference':float(max(r['marginal_max_abs_difference'] for r in detail)),
        'mean_original_vs_shuffled_max_probability_difference':float(np.mean([r['original_vs_shuffled_max_abs_probability'] for r in detail])),
        'mean_empirical_near_zero_fraction':float(np.mean([r['empirical_near_zero_fraction'] for r in detail])),
        'limitations':[
            'No trainable M5 fitted; Stage 4 original alpha selected on same validation set.',
            'Permutation repeats are not independent evolutionary trajectories.',
            'Shuffling preserves empirical marginals exactly, but does not force D=0 in finite samples.',
            'Original and shuffled have identical empirical marginals before EPS normalization.',
            'Neutral and linkage_only trajectories may be identical.',
            'Only validation labels are used; no simulator truth or test seeds loaded.',
            'Forced dynamics is a diagnostic, not a validation-selected forecasting model.'
        ]
    }
    (args.output/'stage41a_report.json').write_text(json.dumps(report,indent=2))
    print('=== MATCHED DEPENDENCE AUDIT ===')
    print('folds:',len(rows),'alpha:',alpha)
    for name in names:
        print(f'{name:19s} joint_nll={summary[name]["joint_nll"]:.6f} ld_D_mae={summary[name]["ld_D_mae"]:.6f}')
    print('mean_permuted_minus_original_nll:',round(report['mean_permuted_minus_original_nll'],8))
    print('max_marginal_difference:',report['max_marginal_difference'])
    print('empirical_near_zero_fraction:',round(report['mean_empirical_near_zero_fraction'],6))
    print('Saved:',args.output)

if __name__=='__main__':
    main()
