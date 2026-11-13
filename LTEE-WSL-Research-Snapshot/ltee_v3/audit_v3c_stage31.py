#!/usr/bin/env python3
"""Read-only post-fit M4 gate/expert audit against the actual v3c_stage3.py."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit


def load_source(path):
    spec = importlib.util.spec_from_file_location('v3c_stage3_production', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def nll(p, y, n):
    p = np.clip(np.asarray(p), 1e-5, 1-1e-5)
    return float(np.sum(n * (-y*np.log(p) - (1-y)*np.log1p(-p))) / np.sum(n))


def evaluate(prod, rows, theta, hs, vs, seed, permutations):
    rng = np.random.default_rng(seed)
    details = []
    for r in rows:
        d = prod.design([r], hs, vs)
        X, G, base = d['X'], d['G'], d['base']
        z = logit(base)
        pa = expit(np.clip(z + X @ theta[:4], -30, 30))
        pb = expit(np.clip(z + X @ theta[4:8], -30, 30))
        gate = expit(np.clip(G @ theta[8:11], -30, 30))
        orig = (1-gate)*pa + gate*pb
        actual = prod.predictions(theta, d, 'M4_latent')
        if not np.allclose(actual, orig, atol=1e-12, rtol=1e-12):
            raise AssertionError('Expert decomposition does not match production predictions')
        details.append(dict(row=r, pa=pa, pb=pb, gate=gate, orig=orig, n=d['n'], y=d['y']))

    # Crucial: shuffle only the historical volatility across forecast folds;
    # never shuffle labels, latest frequencies, horizon, or fitted parameters.
    # Same-scenario permutation avoids trivially exchanging scenario identity.
    by_scenario = {}
    for i, x in enumerate(details):
        by_scenario.setdefault(x['row']['scenario'], []).append(i)
    shuffled_losses = []
    changed = []
    for _ in range(permutations):
        mapping = np.arange(len(details))
        for inds in by_scenario.values():
            if len(inds)>1:
                mapping[inds] = rng.permutation(inds)
        pp=[]; diffs=[]
        for i,x in enumerate(details):
            r=x['row']; source=details[mapping[i]]['row']
            altered={**r,'volatility':source['volatility']}
            dd=prod.design([altered],hs,vs)
            gp=expit(np.clip(dd['G'] @ theta[8:11],-30,30))
            p=(1-gp)*x['pa']+gp*x['pb']
            pp.append(p)
            diffs.append(float(np.max(np.abs(p-x['orig']))))
        shuffled_losses.append(nll(np.concatenate(pp),np.concatenate([x['y'] for x in details]),np.concatenate([x['n'] for x in details])))
        changed.append(max(diffs))

    y=np.concatenate([x['y'] for x in details]);n=np.concatenate([x['n'] for x in details])
    def stack(key):return np.concatenate([x[key] for x in details])
    pa,pb,gate,orig=[stack(k) for k in ('pa','pb','gate','orig')]
    orig_loss=nll(orig,y,n)
    report={
        'n_folds':len(rows),'n_scenarios':len(by_scenario),
        'expert_separation_mean_abs':float(np.mean(np.abs(pa-pb))),
        'expert_separation_max_abs':float(np.max(np.abs(pa-pb))),
        'gate_mean':float(np.mean(gate)),'gate_std':float(np.std(gate)),
        'gate_min':float(np.min(gate)),'gate_max':float(np.max(gate)),
        'nll_original':orig_loss,
        'nll_forced_expert_a':nll(pa,y,n),
        'nll_forced_expert_b':nll(pb,y,n),
        'nll_forced_half_gate':nll(.5*pa+.5*pb,y,n),
        'permutation_n':permutations,
        'permutation_mean_nll':float(np.mean(shuffled_losses)),
        'permutation_delta_nll_mean':float(np.mean(shuffled_losses)-orig_loss),
        'permutation_delta_nll_std':float(np.std(np.asarray(shuffled_losses)-orig_loss,ddof=1)) if permutations>1 else 0.,
        'permutation_delta_nll_quantiles':np.quantile(np.asarray(shuffled_losses)-orig_loss,[.025,.5,.975]).tolist(),
        'permutation_max_prediction_change_mean':float(np.mean(changed)),
        'note':'Validation-only exploratory diagnostics; folds correlated; permutation distribution is not a formal p-value.'
    }
    per_fold=[]
    for x in details:
        r=x['row'];v=float(r['volatility'])
        per_fold.append({'scenario':r['scenario'],'seed':r['seed'],'fold_id':r['fold_id'],
            'cutoff':r['cutoff'],'target_generation':r['target_generation'],
            'volatility':v,'gate_mean':float(x['gate'].mean()),
            'expert_mean_abs_difference':float(np.mean(np.abs(x['pa']-x['pb']))),
            'nll_original':nll(x['orig'],x['y'],x['n']),
            'nll_expert_a':nll(x['pa'],x['y'],x['n']),
            'nll_expert_b':nll(x['pb'],x['y'],x['n'])})
    return report,pd.DataFrame(per_fold)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--original-results',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--validation-seeds',nargs='+',type=int,default=[42])
    ap.add_argument('--permutations',type=int,default=200)
    ap.add_argument('--seed',type=int,default=17031)
    args=ap.parse_args()
    if args.permutations<1:ap.error('permutations must be positive')
    prod=load_source(args.source)
    selected=json.loads((args.original_results/'selected_models.json').read_text())
    protocol=json.loads((args.original_results/'protocol.json').read_text())
    if sorted(args.validation_seeds)!=sorted(protocol['validation_seeds']):
        ap.error('validation seeds do not match saved protocol')
    if set(args.validation_seeds)&(set(protocol['train_seeds'])|set(protocol['test_seeds'])):
        ap.error('validation seeds overlap train/test')
    theta=np.asarray(selected['M4_latent']['coefficients'],dtype=float)
    if theta.shape!=(11,):raise ValueError('expected exactly 11 M4_latent coefficients')
    hs=float(protocol['horizon_scale_train_only']);vs=float(protocol['volatility_scale_train_only'])
    scenarios=set(protocol['scenarios'])
    rows=prod.make_rows(args.input,args.validation_seeds,scenarios)
    report,folds=evaluate(prod,rows,theta,hs,vs,args.seed,args.permutations)
    report.update({'source':str(args.source),'validation_seeds':args.validation_seeds,
        'gate_coefficients':theta[8:11].tolist(),
        'shuffle_scope':'within scenario, across validation folds; frozen model',
        'protocol_scales':{'horizon':hs,'volatility':vs}})
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'stage31_report.json').write_text(json.dumps(report,indent=2))
    folds.to_csv(args.output/'fold_diagnostics.csv',index=False)
    print('=== POST-FIT M4 AUDIT ===')
    for k,v in report.items():
        if k not in ('source','note'):print(f'{k}: {v}')
    print('\n=== GATE BY SCENARIO ===')
    print(folds.groupby('scenario')[['volatility','gate_mean','expert_mean_abs_difference','nll_original']].mean().round(6).to_string())
    if 'regime_shift' in set(folds.scenario):
        rr=folds[folds.scenario=='regime_shift'].copy()
        rr['period']=np.where(rr.cutoff<50,'before_cutoff_50','at_or_after_cutoff_50')
        print('\n=== REGIME SHIFT EXPLORATORY ===')
        print(rr.groupby('period')[['volatility','gate_mean','nll_original']].agg(['mean','count']).to_string())
    print('\nSaved',args.output)

if __name__=='__main__':main()
