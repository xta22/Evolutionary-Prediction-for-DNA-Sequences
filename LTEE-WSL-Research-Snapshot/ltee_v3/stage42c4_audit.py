#!/usr/bin/env python3
"""V3-C Stage 4.2C.4: matched sampling-information and NLL-tail audit.

Development only. Uses the same simulator seeds, cutoffs, targets and adjacent
pairs as Stage 4.2C.3. Predictions never use future labels. The full-state
hybrid is privileged and diagnostic, not a deployable predictor.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

from stage42c3_matched import load, config, make_rows, checkpoint

SCENARIOS = ('neutral', 'recombination', 'selection_only', 'epistasis')


def marginals(q):
    q=np.asarray(q,float)
    return np.column_stack((q[:,2]+q[:,3],q[:,1]+q[:,3]))


def dependence(q):
    p=marginals(q)
    return q[:,3]-p[:,0]*p[:,1]


def from_marginals_and_D(p,D):
    """Project requested dependence to exact Frechet bounds, preserving marginals."""
    p=np.asarray(p,float)
    D=np.asarray(D,float)
    a,b=p[:,0],p[:,1]
    low=np.maximum(0.,a+b-1.)-a*b
    high=np.minimum(a,b)-a*b
    d=np.clip(D,low,high)
    q11=a*b+d
    q=np.column_stack((1-a-b+q11,b-q11,a-q11,q11))
    if np.min(q)<-1e-10 or np.max(np.abs(marginals(q)-p))>1e-9:
        raise AssertionError('Invalid joint probability reconstruction')
    return np.maximum(q,0.), d


def nll_tail(q,truth,threshold=1e-4,eps=1e-9):
    """Average log loss and near-zero event contribution, per pair and clone."""
    q=np.asarray(q,float);truth=np.asarray(truth,float)
    if q.shape!=truth.shape or q.ndim!=2 or q.shape[1]!=4:
        raise ValueError('Expected matching [pairs,4] matrices')
    if np.max(np.abs(truth.sum(axis=1)-1))>1e-8:
        raise ValueError('Invalid truth distribution')
    losses=-truth*np.log(np.clip(q,eps,1.))
    mask=q<threshold
    return {
        'joint_nll':float(losses.sum(axis=1).mean()),
        'near_zero_true_event_fraction':float((truth*mask).sum(axis=1).mean()),
        'near_zero_loss_contribution':float((losses*mask).sum(axis=1).mean()),
        'near_zero_loss_share':float((losses*mask).sum()/max(losses.sum(),1e-15)),
        'mean_true_state_probability':float((truth*q).sum(axis=1).mean()),
    }


def write_csv(path,rows):
    if not rows:raise ValueError('No rows')
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)


def run(a):
    if a.output.exists() and any(a.output.iterdir()):
        raise ValueError('Output must be absent or empty; keep previous results')
    if set(a.train_seeds)&set(a.eval_seeds):
        raise ValueError('Train and evaluation seeds overlap')
    if not a.train_seeds or not a.eval_seeds:
        raise ValueError('Nonempty train and eval seeds required')
    if a.sample_size>a.individuals or a.sample_size<1:
        raise ValueError('Invalid sample size')
    if any(c%a.observe_step or c+h>a.generations for c in a.cutoffs for h in a.horizons):
        raise ValueError('Cutoff or horizon is unavailable')
    ref=load(a.reference,'ref42c4')
    prod=load(a.source,'prod42c4')
    model=load(a.model_source,'model42c4')
    oracle=load(a.oracle_source,'oracle42c4')
    train=make_rows(ref,prod,a,a.scenarios,a.train_seeds,a.cutoffs,a.horizons)
    evaluation=make_rows(ref,prod,a,a.scenarios,a.eval_seeds,a.cutoffs,a.horizons)
    theta,info=model.fit(train,'history',prod,a.penalty)
    print('FROZEN M5_HISTORY theta',theta.tolist(),'iterations',info['iterations'],flush=True)
    fold_metrics=[];tail_metrics=[];pair_metrics=[]
    for scenario in a.scenarios:
      for seed in a.eval_seeds:
       for cutoff in a.cutoffs:
        pop,context=checkpoint(oracle,scenario,seed,cutoff,a)
        matching=[r for r in evaluation if r['scenario']==scenario and r['seed']==seed and r['cutoff']==cutoff]
        if not matching:raise RuntimeError('Missing evaluation folds')
        if not np.array_equal(pop[:a.sample_size],matching[0]['latest_matrix']):
            raise RuntimeError('Cutoff sampled genomes do not match full-state checkpoint')
        for row in matching:
            pairs=row['pairs'];truth=row['truth']
            q_sample=prod.normalize(prod.independent(row['latest']))
            q_m5=model.predict(row,theta,'history',prod)
            q_full=oracle.pair_freq(pop,pairs)
            q_full_ind=prod.normalize(prod.independent(q_full))
            p_sample=marginals(row['latest'])
            p_full=marginals(q_full)
            rho=model.features(row,'history')@theta
            rho=1/(1+np.exp(-np.clip(rho,-30,30)))
            sampled_D=dependence(row['latest'])
            # Privileged marginal control: full population marginals plus the
            # dependence signal and rho estimated from sampled history.
            q_hybrid,projected_D=from_marginals_and_D(p_full,rho*sampled_D)
            q_hybrid=prod.normalize(q_hybrid)
            pred={'sampled_independent':q_sample,'M5_history':q_m5,
                  'full_state_independent':q_full_ind,
                  'full_marginals_sampled_dependence_DIAGNOSTIC':q_hybrid}
            key=dict(scenario=scenario,seed=seed,cutoff=cutoff,horizon=row['horizon'],fold_id=row['fold_id'])
            marginal_abs=np.abs(p_sample-p_full)
            for name,q in pred.items():
                metrics=prod.score(q,truth)
                fold_metrics.append(dict(**key,model=name,n_pairs=len(pairs),n_target_clones=row['n_target_clones'],
                    cutoff_marginal_mae=float(marginal_abs.mean()),cutoff_marginal_rmse=float(np.sqrt(np.mean((p_sample-p_full)**2))),
                    **metrics))
                for threshold in (1e-4,1e-3,1e-2):
                    tail_metrics.append(dict(**key,model=name,threshold=threshold,
                                             **nll_tail(q,truth,threshold,prod.EPS)))
            for k,(i,j) in enumerate(pairs):
                pair_metrics.append(dict(**key,pair_index=k,site_i=i,site_j=j,
                    sampled_p_i=p_sample[k,0],sampled_p_j=p_sample[k,1],
                    full_p_i=p_full[k,0],full_p_j=p_full[k,1],
                    sampled_D=sampled_D[k],full_D=dependence(q_full)[k],
                    rho=rho[k],hybrid_D=projected_D[k],
                    abs_marginal_error=float(marginal_abs[k].mean())))
            print('AUDITED',scenario,seed,cutoff,row['horizon'],flush=True)
    a.output.mkdir(parents=True,exist_ok=True)
    write_csv(a.output/'fold_metrics.csv',fold_metrics)
    write_csv(a.output/'tail_metrics.csv',tail_metrics)
    write_csv(a.output/'pair_diagnostics.csv',pair_metrics)
    models=sorted({r['model'] for r in fold_metrics})
    summary={name:{metric:float(np.mean([r[metric] for r in fold_metrics if r['model']==name]))
                   for metric in ('joint_nll','joint_brier','ld_D_mae','marginal_nll')}
             for name in models}
    hashes={k:hashlib.sha256(getattr(a,k).read_bytes()).hexdigest() for k in ('reference','source','model_source','oracle_source')}
    report=dict(status='DEVELOPMENT_SAMPLING_CALIBRATION_AUDIT_ONLY',
                train_seeds=a.train_seeds,eval_seeds=a.eval_seeds,scenarios=a.scenarios,
                cutoffs=a.cutoffs,horizons=a.horizons,n_eval_folds=len(evaluation),
                sample_size=a.sample_size,individuals=a.individuals,
                history_theta=theta.tolist(),penalty=a.penalty,summary=summary,
                source_sha256=hashes,limitations=[
                    'Full-state controls have privileged cutoff information.',
                    'Hybrid full-marginal predictor is a diagnostic, not a deployable model.',
                    'Target distributions only enter scoring and error diagnostics.',
                    'Samples are the first N individuals, not random population subsamples.',
                    'Repeated horizons and folds are correlated; exploratory only.',
                    'No oracle rollouts here; compare to prior matched oracle separately.',
                    'All model scores use the same production Stage 4 scoring function.'])
    (a.output/'report.json').write_text(json.dumps(report,indent=2))
    print('=== MATCHED INFORMATION CONTROLS ===')
    for name,metrics in summary.items():
        print(name,'NLL',round(metrics['joint_nll'],6),'marginal NLL',round(metrics['marginal_nll'],6))
    print('=== NEAR-ZERO EVENT AUDIT (threshold 1e-4) ===')
    for name in models:
        x=[r for r in tail_metrics if r['model']==name and r['threshold']==1e-4]
        print(name,'event_fraction',round(float(np.mean([r['near_zero_true_event_fraction'] for r in x])),6),
              'NLL_contribution',round(float(np.mean([r['near_zero_loss_contribution'] for r in x])),6))
    print('SAVED',a.output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('reference','source','model-source','oracle-source','output'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--train-seeds',nargs='+',type=int,default=[311,313,317,331])
    p.add_argument('--eval-seeds',nargs='+',type=int,default=[211,223])
    p.add_argument('--scenarios',nargs='+',choices=SCENARIOS,default=list(SCENARIOS))
    p.add_argument('--cutoffs',nargs='+',type=int,default=[20,50])
    p.add_argument('--horizons',nargs='+',type=int,default=[10,20,30])
    p.add_argument('--penalty',type=float,default=.01)
    for name,typ,default in [('individuals',int,120),('sites',int,24),('generations',int,100),
        ('observe-step',int,10),('sample-size',int,8),('rate',float,.001),
        ('initial-frequency',float,.25),('context-sigma',float,1.),
        ('heterogeneity-multiplier',float,3.),('shift-generation',int,50),
        ('shift-multiplier',float,4.),('recombination-probability',float,.5),
        ('selection-strength',float,.12),('epistasis-strength',float,.25)]:
        p.add_argument('--'+name,type=typ,default=default)
    a=p.parse_args()
    if a.penalty<0:p.error('negative penalty')
    run(a)

if __name__=='__main__':main()
