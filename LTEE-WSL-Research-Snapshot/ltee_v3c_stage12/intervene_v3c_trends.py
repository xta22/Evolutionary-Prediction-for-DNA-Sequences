#!/usr/bin/env python3
"""Read-only production V3-C B2/B3 trend intervention (validation-only).

Imports the actual local v3c_forecast.py; does not replace its prediction formula.
Never opens test-seed directories, future simulation truth, or original test metrics.
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import numpy as np

MODELS = ('B1_latest', 'B2_shrunk_trend', 'B3_site_trend')

def load_production(path):
    spec = importlib.util.spec_from_file_location('v3c_forecast_production', str(path.resolve()))
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot import production source')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ('predict', 'parse_history', 'read_csv', 'score'):
        if not hasattr(module, name):
            raise RuntimeError(f'Missing production function: {name}')
    return module

def fixture_checks(prod):
    # Deliberately unequal site slopes; test the real production function.
    arrays = {
        0: np.array([[0, 1, 0], [0, 1, 0]], dtype=float),
        10: np.array([[0, 1, 0], [0, 1, 0]], dtype=float),
        20: np.array([[1, 1, 0], [1, 1, 0]], dtype=float),
    }
    alpha = 1.0
    intervention = 0.25
    b1 = prod.predict(arrays, 20, 30, 'B1_latest', alpha, 0.0)
    b2 = prod.predict(arrays, 20, 30, 'B2_shrunk_trend', alpha, intervention)
    b3 = prod.predict(arrays, 20, 30, 'B3_site_trend', alpha, intervention)
    b2zero = prod.predict(arrays, 20, 30, 'B2_shrunk_trend', alpha, 0.0)
    b3zero = prod.predict(arrays, 20, 30, 'B3_site_trend', alpha, 0.0)
    if not (np.array_equal(b1, b2zero) and np.array_equal(b1, b3zero)):
        raise AssertionError('Zero-strength production trend differs from B1')
    if np.array_equal(b1, b2) or np.array_equal(b1, b3):
        raise AssertionError('Production nonzero trend did not change fixture predictions')
    # Horizon scaling checked before clipping (chosen fixture avoids clipping for B2).
    b2half = prod.predict(arrays, 20, 25, 'B2_shrunk_trend', alpha, intervention)
    if not np.allclose(b2half - b1, (b2 - b1) * 0.5, atol=1e-10):
        raise AssertionError('Production B2 horizon scaling failed')
    return {'zero_strength_exact_identity': True,
            'B2_nonzero_max_difference': float(np.max(np.abs(b2-b1))),
            'B3_nonzero_max_difference': float(np.max(np.abs(b3-b1))),
            'B2_horizon_scaling': True}

def fold_predictions(prod, run_dir, model, alpha, strength, min_cutoff):
    folds = prod.read_csv(run_dir/'folds.csv')
    history = prod.read_csv(run_dir/'historical_inputs.csv')
    targets = prod.read_csv(run_dir/'evaluation_targets_RESTRICTED.csv')
    records=[]
    for fold in folds:
        cutoff=int(fold['cutoff']); target=int(fold['target_generation']); fid=fold['fold_id']
        if cutoff < min_cutoff:
            continue
        h=[r for r in history if r['fold_id']==fid]
        y=[r for r in targets if r['fold_id']==fid]
        if not h or not y:
            raise ValueError(f'Empty fold {fid}')
        if any(int(r['generation'])>cutoff for r in h):
            raise ValueError('History contains future records')
        if any(int(r['generation'])!=target for r in y):
            raise ValueError('Target time mismatch')
        arrays=prod.parse_history(h, cutoff)
        p=prod.predict(arrays,cutoff,target,model,alpha,strength)
        yy=np.array([[int(c) for c in r['sequence']] for r in y],dtype=float)
        metrics=prod.score(p,yy)
        records.append((fid,cutoff,target,p,metrics))
    return records

def run(args):
    prod=load_production(Path(args.source))
    fixture=fixture_checks(prod)
    if set(args.validation_seeds) & set(args.test_seeds):
        raise ValueError('Validation/test seeds overlap')
    if args.test_seeds and any(s in args.validation_seeds for s in args.test_seeds):
        raise ValueError('Seed overlap')
    root=Path(args.input)
    out=Path(args.output)
    if out.resolve() == root.resolve() or out.resolve() == Path(args.original_results).resolve():
        raise ValueError('Output must be separate from input/original results')
    original=json.loads((Path(args.original_results)/'selected_hyperparameters.json').read_text())
    scenarios=args.scenarios or list(prod.SCENARIOS)
    rows=[]
    for scenario in scenarios:
        for seed in args.validation_seeds:
            run_dir=root/scenario/f'seed_{seed}'
            if not run_dir.is_dir():
                raise FileNotFoundError(run_dir)
            for model in MODELS:
                alpha=float(original[model]['alpha'])
                strengths=[0.0] if model=='B1_latest' else args.strengths
                baseline=fold_predictions(prod,run_dir,'B1_latest',float(original['B1_latest']['alpha']),0.0,args.min_cutoff)
                base_by_id={r[0]:r for r in baseline}
                for strength in strengths:
                    for fid,cutoff,target,p,metric in fold_predictions(prod,run_dir,model,alpha,strength,args.min_cutoff):
                        base=base_by_id[fid][3]
                        if p.shape != base.shape:
                            raise ValueError('Probability shape mismatch')
                        rows.append({'split':'validation','scenario':scenario,'seed':seed,
                                     'fold_id':fid,'cutoff':cutoff,'target_generation':target,
                                     'model':model,'alpha':alpha,'trend_strength':strength,
                                     'nll':metric['nll'],'brier':metric['brier'],
                                     'abs_count_error':metric['abs_count_error'],
                                     'max_abs_prob_diff_from_B1':float(np.max(np.abs(p-base))),
                                     'mean_abs_prob_diff_from_B1':float(np.mean(np.abs(p-base))),
                                     'fraction_sites_changed_from_B1':float(np.mean(np.abs(p-base)>1e-12))})
    if not rows:
        raise ValueError('No validation folds')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'validation_interventions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    groups={}
    for row in rows:
        key=(row['scenario'],row['model'],row['trend_strength'])
        groups.setdefault(key,[]).append(row)
    summary=[]
    for (scenario,model,strength),items in sorted(groups.items()):
        summary.append({'scenario':scenario,'model':model,'trend_strength':strength,
                        'folds':len(items),'mean_validation_nll':float(np.mean([r['nll'] for r in items])),
                        'max_probability_difference':max(r['max_abs_prob_diff_from_B1'] for r in items),
                        'fraction_folds_with_changed_predictions':float(np.mean([r['max_abs_prob_diff_from_B1']>1e-12 for r in items]))})
    with (out/'validation_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]))
        writer.writeheader();writer.writerows(summary)
    report={'status':'EXPLORATORY_VALIDATION_ONLY','production_source':str(Path(args.source).resolve()),
            'original_selection_file':str(Path(args.original_results)/'selected_hyperparameters.json'),
            'fixture':fixture,'validation_seeds':args.validation_seeds,
            'test_seeds_not_accessed':args.test_seeds,'scenarios':scenarios,
            'strength_grid':args.strengths,'n_validation_rows':len(rows),
            'notes':['Actual production predict(), parse_history(), score() called.',
                     'No test data or simulation_truth_RESTRICTED files accessed.',
                     'This diagnostic must not be used to retune original held-out test performance.',
                     'Original selected hyperparameters and original metrics remain unchanged.']}
    (out/'intervention_report.json').write_text(json.dumps(report,indent=2))
    print('PRODUCTION FIXTURE:',json.dumps(fixture))
    print('VALIDATION INTERVENTIONS')
    for row in summary:
        print(f"{row['scenario']:16s} {row['model']:16s} strength={row['trend_strength']:4.2f} NLL={row['mean_validation_nll']:.6f} max|dP|={row['max_probability_difference']:.6g} changed_folds={row['fraction_folds_with_changed_predictions']:.2f}")
    print('Saved',out)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',default='v3c_forecast.py')
    ap.add_argument('--input',default='../LTEE_prospective/v3b_simulation')
    ap.add_argument('--original-results',default='../LTEE_prospective/v3c_stage1')
    ap.add_argument('--output',default='../LTEE_prospective/v3c_stage12_intervention')
    ap.add_argument('--validation-seeds',nargs='+',type=int,default=[42])
    ap.add_argument('--test-seeds',nargs='*',type=int,default=[67,101])
    ap.add_argument('--scenarios',nargs='*',default=None)
    ap.add_argument('--strengths',nargs='+',type=float,default=[0.0,0.25,0.5,1.0])
    ap.add_argument('--min-cutoff',type=int,default=10)
    args=ap.parse_args()
    if any(s<0 for s in args.strengths):
        ap.error('Strengths must be nonnegative')
    run(args)

if __name__=='__main__':
    main()
