#!/usr/bin/env python3
"""Stage 4.2C.5A: development-only cutoff-marginal smoothing audit.

Requires stage42c3_matched.py and stage42c4_audit.py alongside this file.
Never uses evaluation targets to construct predictions or select parameters.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from stage42c3_matched import load, make_rows
from stage42c4_audit import marginals, dependence, from_marginals_and_D, nll_tail

SCENARIOS=('neutral','recombination','selection_only','epistasis')

def smooth_marginals(latest_matrix,pairs,alpha):
    """Jeffreys/Beta-binomial pseudocounts from CURRENT cutoff sample only."""
    if alpha<0: raise ValueError('negative alpha')
    m=np.asarray(latest_matrix)
    if m.ndim!=2 or m.shape[0]<1: raise ValueError('empty cutoff sample')
    n=len(m)
    ix=np.array(pairs,dtype=int)
    return np.column_stack(((m[:,ix[:,0]].sum(axis=0)+alpha)/(n+2*alpha),
                            (m[:,ix[:,1]].sum(axis=0)+alpha)/(n+2*alpha)))

def predict_smoothed(row,theta,model,prod,alpha,uniform_mix):
    """Reconstruct joint from smoothed marginals and historical D; then mix uniform."""
    if not 0<=uniform_mix<1: raise ValueError('uniform mix must be in [0,1)')
    p=smooth_marginals(row['latest_matrix'],row['pairs'],alpha)
    d=dependence(row['latest'])
    logits=model.features(row,'history')@theta
    rho=1/(1+np.exp(-np.clip(logits,-30,30)))
    q,_=from_marginals_and_D(p,rho*d)
    q=(1-uniform_mix)*q+uniform_mix/4
    return prod.normalize(q)

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run(a):
    if a.output.exists() and any(a.output.iterdir()):raise ValueError('Output must be absent or empty')
    if not a.train_seeds or not a.eval_seeds or set(a.train_seeds)&set(a.eval_seeds):
        raise ValueError('Train/eval seeds missing or overlapping')
    if a.sample_size<1 or a.sample_size>a.individuals:raise ValueError('invalid sample size')
    if any(c%a.observe_step or c+h>a.generations for c in a.cutoffs for h in a.horizons):
        raise ValueError('unavailable cutoff/target')
    ref=load(a.reference,'ref42c5a');prod=load(a.source,'prod42c5a');model=load(a.model_source,'model42c5a')
    train=make_rows(ref,prod,a,a.scenarios,a.train_seeds,a.cutoffs,a.horizons)
    evaluation=make_rows(ref,prod,a,a.scenarios,a.eval_seeds,a.cutoffs,a.horizons)
    theta,fit_info=model.fit(train,'history',prod,a.penalty)
    print('FROZEN HISTORY theta',theta.tolist(),'iterations',fit_info['iterations'],flush=True)
    rows=[]
    for row in evaluation:
        truth=row['truth']
        original=model.predict(row,theta,'history',prod)
        baseline=prod.score(original,truth)['joint_nll']
        for alpha in a.alphas:
            for mix in a.mixes:
                q=predict_smoothed(row,theta,model,prod,alpha,mix)
                s=prod.score(q,truth)
                tail=nll_tail(q,truth,1e-4,prod.EPS)
                rows.append(dict(scenario=row['scenario'],seed=row['seed'],cutoff=row['cutoff'],
                    horizon=row['horizon'],fold_id=row['fold_id'],alpha=alpha,uniform_mix=mix,
                    n_pairs=len(row['pairs']),n_target_clones=row['n_target_clones'],
                    baseline_joint_nll=baseline,**s,
                    near_zero_true_event_fraction=tail['near_zero_true_event_fraction'],
                    near_zero_loss_contribution=tail['near_zero_loss_contribution']))
        print('AUDITED',row['scenario'],row['seed'],row['fold_id'],flush=True)
    a.output.mkdir(parents=True,exist_ok=True)
    write_csv(a.output/'fold_metrics.csv',rows)
    summary=[]
    for alpha in a.alphas:
        for mix in a.mixes:
            rr=[r for r in rows if r['alpha']==alpha and r['uniform_mix']==mix]
            metrics=('joint_nll','marginal_nll','joint_brier','ld_D_mae',
                     'near_zero_true_event_fraction','near_zero_loss_contribution')
            summary.append(dict(alpha=alpha,uniform_mix=mix,n_folds=len(rr),
                **{m:float(np.mean([r[m] for r in rr])) for m in metrics},
                mean_delta_joint_nll_vs_original=float(np.mean([r['joint_nll']-r['baseline_joint_nll'] for r in rr]))))
    write_csv(a.output/'grid_summary.csv',summary)
    # The grid is descriptive only; no chosen configuration is certified or fitted to eval labels.
    report=dict(status='EXPLORATORY_SMOOTHING_GRID_NOT_LOCKED',
        train_seeds=a.train_seeds,eval_seeds=a.eval_seeds,scenarios=a.scenarios,
        cutoffs=a.cutoffs,horizons=a.horizons,alphas=a.alphas,mixes=a.mixes,
        sample_size=a.sample_size,penalty=a.penalty,history_theta=theta.tolist(),
        n_eval_folds=len(evaluation),
        source_sha256={k:hashlib.sha256(getattr(a,k).read_bytes()).hexdigest() for k in ('reference','source','model_source')},
        limitations=['Grid inspected on development evaluation labels; do not call best row test performance.',
          'Smoothing changes marginals; dependence persistence is frozen.',
          'No future targets enter prediction construction.',
          'Original simulator observes first N individuals, not a random sample.',
          'Folds/horizons share trajectories and are correlated.',
          'For independent confirmation, choose smoothing using separate validation trajectories.'])
    (a.output/'report.json').write_text(json.dumps(report,indent=2))
    print('=== SMOOTHING GRID ===')
    for r in summary:
        print('alpha',r['alpha'],'mix',r['uniform_mix'],'NLL',round(r['joint_nll'],6),
              'delta',round(r['mean_delta_joint_nll_vs_original'],6),
              'near-zero',round(r['near_zero_true_event_fraction'],6))
    print('SAVED',a.output)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('reference','source','model-source','output'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--train-seeds',nargs='+',type=int,default=[311,313,317,331])
    p.add_argument('--eval-seeds',nargs='+',type=int,default=[211,223])
    p.add_argument('--scenarios',nargs='+',choices=SCENARIOS,default=list(SCENARIOS))
    p.add_argument('--cutoffs',nargs='+',type=int,default=[20,50])
    p.add_argument('--horizons',nargs='+',type=int,default=[10,20,30])
    p.add_argument('--alphas',nargs='+',type=float,default=[0,.1,.5,1,2,5])
    p.add_argument('--mixes',nargs='+',type=float,default=[0,.001,.01,.05])
    p.add_argument('--penalty',type=float,default=.01)
    for name,typ,default in [('individuals',int,120),('sites',int,24),('generations',int,100),
        ('observe-step',int,10),('sample-size',int,8),('rate',float,.001),
        ('initial-frequency',float,.25),('context-sigma',float,1.),
        ('heterogeneity-multiplier',float,3.),('shift-generation',int,50),
        ('shift-multiplier',float,4.),('recombination-probability',float,.5),
        ('selection-strength',float,.12),('epistasis-strength',float,.25)]:
        p.add_argument('--'+name,type=typ,default=default)
    a=p.parse_args()
    if not a.alphas or any(x<0 for x in a.alphas) or len(set(a.alphas))!=len(a.alphas):p.error('invalid alphas')
    if not a.mixes or any(not 0<=x<1 for x in a.mixes) or len(set(a.mixes))!=len(a.mixes):p.error('invalid mixes')
    if a.penalty<0:p.error('negative penalty')
    run(a)
if __name__=='__main__':main()
