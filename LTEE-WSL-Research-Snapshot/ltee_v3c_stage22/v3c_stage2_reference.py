#!/usr/bin/env python3
"""V3-C Stage 2: trainable binary marginal forecasting baselines.

M1: shared log-odds correction; M2: shared history-conditioned dynamics;
M3: adds observable local genomic-frequency context.

No individual ancestry is assumed. Fits shared coefficients ONLY on train seeds,
chooses L2 on validation seeds, evaluates test seeds once. Restricted target
files are read by the evaluator, never by the feature builder.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

SCENARIOS = ('neutral','heterogeneous','context','regime_shift','linkage_only',
             'recombination','selection_only','epistasis')
MODELS = ('B1_latest','M1_shared','M2_history','M3_local_context')
EPS = 1e-5


def read_rows(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('No rows to save')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def sigmoid(z):
    z = np.clip(z, -40, 40)
    return 1. / (1. + np.exp(-z))


def logit(p):
    p = np.clip(p, EPS, 1-EPS)
    return np.log(p) - np.log1p(-p)


def parse_history(rows, cutoff):
    if not rows:
        raise ValueError('Empty historical inputs')
    times = sorted({int(r['generation']) for r in rows})
    if times[-1] > cutoff:
        raise ValueError('FUTURE HISTORY LEAKAGE')
    out = {}
    n_sites = None
    for t in times:
        ss = [r['sequence'] for r in rows if int(r['generation']) == t]
        if not ss or any(set(s) - {'0','1'} for s in ss):
            raise ValueError('Invalid binary sequences')
        if n_sites is None:
            n_sites = len(ss[0])
        if any(len(s) != n_sites for s in ss):
            raise ValueError('Inconsistent sequence length')
        out[t] = np.asarray([[int(x) for x in s] for s in ss], dtype=float)
    return out


def make_features(history, cutoff, target, alpha=5.):
    """Feature extraction from historical observations only. Never takes labels."""
    if target <= cutoff:
        raise ValueError('Target must be after cutoff')
    times = sorted(history)
    if not times or times[-1] > cutoff:
        raise ValueError('History extends past cutoff')
    last = history[times[-1]]
    n_sites = last.shape[1]
    pooled = np.concatenate([history[t] for t in times], axis=0)
    global_p = (pooled.sum() + alpha) / (pooled.size + 2*alpha)
    p0 = (last.sum(axis=0) + alpha*global_p) / (len(last)+alpha)
    p0 = np.clip(p0, EPS, 1-EPS)
    horizon = (target-cutoff)/max(1, (times[-1]-times[-2]) if len(times)>1 else target-cutoff)
    if len(times)>1:
        prev = history[times[-2]]
        pp = (prev.sum(axis=0) + alpha*global_p)/(len(prev)+alpha)
        delta = p0-pp
    else:
        delta = np.zeros(n_sites)
    # Historical population-wide trajectory summaries; do not assume sample_index lineage.
    mean_delta = float(np.mean(delta))
    mean_abs_delta = float(np.mean(np.abs(delta)))
    centered = p0 - float(np.mean(p0))
    neighbor = (np.roll(p0, 1) + np.roll(p0, -1))/2 - p0
    local_heterozygosity = (np.roll(p0*(1-p0), 1)+np.roll(p0*(1-p0), -1))/2
    h = np.log1p(horizon)
    x1 = np.column_stack([np.ones(n_sites), np.full(n_sites, h)])
    x2 = np.column_stack([x1, np.full(n_sites, mean_delta*h),
                          np.full(n_sites, mean_abs_delta*h), centered*h])
    x3 = np.column_stack([x2, neighbor*h, local_heterozygosity*h,
                          centered*neighbor*h])
    return p0, {'M1_shared':x1, 'M2_history':x2, 'M3_local_context':x3}


def load_fold(root, scenario, seed, fold, include_target):
    run = root/scenario/f'seed_{seed}'
    fid = fold['fold_id']
    cutoff, target = int(fold['cutoff']), int(fold['target_generation'])
    history_rows = [r for r in read_rows(run/'historical_inputs.csv') if r['fold_id']==fid]
    if not history_rows:
        raise ValueError(f'Empty history {run} {fid}')
    hist = parse_history(history_rows, cutoff)
    p0, feats = make_features(hist, cutoff, target)
    if not include_target:
        return p0, feats, None
    # Evaluator-only path. No target information enters make_features.
    target_rows = [r for r in read_rows(run/'evaluation_targets_RESTRICTED.csv') if r['fold_id']==fid]
    if not target_rows or any(int(r['generation']) != target for r in target_rows):
        raise ValueError('TARGET TIME LEAKAGE or missing targets')
    y = np.asarray([[int(x) for x in r['sequence']] for r in target_rows],dtype=float)
    if y.shape[1] != len(p0):
        raise ValueError('Target length mismatch')
    return p0, feats, y


def collect(root, scenarios, seeds, min_cutoff):
    data=[]
    for scenario in scenarios:
        for seed in seeds:
            run = root/scenario/f'seed_{seed}'
            folds = read_rows(run/'folds.csv')
            for fold in folds:
                if int(fold['cutoff']) < min_cutoff:
                    continue
                p0, features, y = load_fold(root,scenario,seed,fold,True)
                data.append({'scenario':scenario,'seed':seed,'fold_id':fold['fold_id'],
                             'cutoff':int(fold['cutoff']), 'target_generation':int(fold['target_generation']),
                             'p0':p0,'features':features,'y':y})
    if not data:
        raise ValueError('No folds available')
    return data


def prepare(data, model):
    xs=[]; offsets=[]; positives=[]; totals=[]
    for d in data:
        xs.append(d['features'][model]); offsets.append(logit(d['p0']))
        positives.append(d['y'].sum(axis=0)); totals.append(np.full(len(d['p0']),len(d['y'])))
    return np.concatenate(xs), np.concatenate(offsets), np.concatenate(positives), np.concatenate(totals)


def fit_model(data, model, penalty):
    X, offset, successes, n = prepare(data, model)
    if np.any(~np.isfinite(X)):
        raise ValueError('Nonfinite features')
    def objective(w):
        z = offset + X@w
        p = sigmoid(z)
        loss = np.sum(n*np.logaddexp(0,z)-successes*z)/n.sum()
        # Penalize all coefficients; this is regularized shared regression, not Bayesian hierarchy.
        val = loss + 0.5*penalty*np.dot(w,w)
        grad = (X.T@(n*p-successes))/n.sum() + penalty*w
        return val, grad
    res = minimize(objective, np.zeros(X.shape[1]), method='L-BFGS-B',jac=True,
                   options={'maxiter':300,'ftol':1e-11})
    return {'coef':res.x,'success':bool(res.success),'message':str(res.message),
            'iterations':int(res.nit),'objective':float(res.fun)}


def predict(d, model, fitted=None):
    if model=='B1_latest':
        return d['p0']
    if fitted is None:
        raise ValueError('Missing fitted coefficients')
    return np.clip(sigmoid(logit(d['p0']) + d['features'][model]@fitted['coef']),EPS,1-EPS)


def metrics(p, y):
    nll = -np.mean(y*np.log(p[None,:])+(1-y)*np.log1p(-p[None,:]))
    brier = np.mean((y-p[None,:])**2)
    expected = len(y)*p.sum()
    observed = y.sum()
    return {'nll':float(nll),'brier':float(brier),
            'abs_count_error':float(abs(expected-observed)),
            'observed_allele1_count':float(observed),
            'predicted_allele1_count':float(expected)}


def mean_nll(data, model, fit):
    return float(np.mean([metrics(predict(d,model,fit),d['y'])['nll'] for d in data]))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default='../LTEE_prospective/v3b_simulation')
    ap.add_argument('--output',default='../LTEE_prospective/v3c_stage2')
    ap.add_argument('--train-seeds',nargs='+',type=int,default=[11,23])
    ap.add_argument('--validation-seeds',nargs='+',type=int,default=[42])
    ap.add_argument('--test-seeds',nargs='+',type=int,default=[67,101])
    ap.add_argument('--scenarios',nargs='+',choices=SCENARIOS,default=list(SCENARIOS))
    ap.add_argument('--penalties',nargs='+',type=float,default=[0.0001,0.001,0.01,0.1])
    ap.add_argument('--min-cutoff',type=int,default=10)
    args=ap.parse_args()
    groups=[set(args.train_seeds),set(args.validation_seeds),set(args.test_seeds)]
    if not all(groups) or any(groups[i]&groups[j] for i in range(3) for j in range(i+1,3)):
        ap.error('train/validation/test trajectory seeds must be nonempty and disjoint')
    if any(x<0 for x in args.penalties):
        ap.error('penalties must be nonnegative')
    root=Path(args.input);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    train=collect(root,args.scenarios,args.train_seeds,args.min_cutoff)
    val=collect(root,args.scenarios,args.validation_seeds,args.min_cutoff)
    selected={}
    for model in MODELS[1:]:
        choices=[]
        for penalty in args.penalties:
            fit=fit_model(train,model,penalty)
            choices.append((mean_nll(val,model,fit),penalty,fit))
        loss,penalty,fit=min(choices,key=lambda z:(z[0],z[1]))
        selected[model]={'penalty':penalty,'validation_nll':loss,
                         'fit_success':fit['success'],'optimizer_message':fit['message'],
                         'iterations':fit['iterations'],'coefficients':fit['coef'].tolist()}
        print('SELECTED',model,'penalty',penalty,'validation NLL',round(loss,6),flush=True)
    # Evaluate test only after all hyperparameters and train fits have been frozen.
    test=collect(root,args.scenarios,args.test_seeds,args.min_cutoff)
    rows=[]
    for split,data in [('validation',val),('test',test)]:
        for d in data:
            for model in MODELS:
                fit={'coef':np.asarray(selected[model]['coefficients'])} if model!='B1_latest' else None
                r=metrics(predict(d,model,fit),d['y'])
                rows.append({'split':split,'scenario':d['scenario'],'seed':d['seed'],
                             'fold_id':d['fold_id'],'cutoff':d['cutoff'],
                             'target_generation':d['target_generation'],'model':model,**r})
    write_rows(out/'metrics.csv',rows)
    (out/'selected_models.json').write_text(json.dumps(selected,indent=2))
    (out/'protocol.json').write_text(json.dumps({
        'status':'EXPLORATORY_STAGE2','train_seeds':args.train_seeds,
        'validation_seeds':args.validation_seeds,'test_seeds':args.test_seeds,
        'scenarios':args.scenarios,'penalties':args.penalties,
        'limitations':['Binary sampled-population marginal allele forecasts, not nucleotide CTMC',
          'M2 uses shared history-conditioned coefficients, not fitted per-population random effects',
          'M3 uses observable neighborhood allele-frequency features, not Evo2 or true simulator context',
          'No joint haplotype model, M4 or M5 yet',
          'Previously inspected seeds: exploratory test only',
          'Correlated folds within trajectories; aggregate per trajectory for uncertainty',
          'Across-scenario training; no hidden mechanism labels supplied as model features']},indent=2))
    print('SAVED',out,'rows',len(rows))
    for model in MODELS:
        a=[r['nll'] for r in rows if r['split']=='test' and r['model']==model]
        print(model,'test NLL',round(float(np.mean(a)),6))

if __name__=='__main__':
    main()
