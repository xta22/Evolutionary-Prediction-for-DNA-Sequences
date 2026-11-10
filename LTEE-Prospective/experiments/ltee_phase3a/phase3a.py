#!/usr/bin/env python3
"""Phase 3A: synthetic context-aware future nucleotide-change forecasting.
No LTEE files, Evo2 embeddings, or future labels are used for training.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def make_data(seed, scenario, n_pop=8, n_sites=1800, dim=8, train_exposure=1.0, test_exposure=1.0):
    rng = np.random.default_rng(seed)
    # Same genomic sites across historical and future periods; independent stochastic events.
    X = rng.normal(size=(n_sites, dim))
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    pop_effect = rng.normal(0, 0.55, n_pop)
    pop_effect -= pop_effect.mean()
    base_rate = 0.035
    if scenario == 'null':
        signal = np.zeros(n_sites)
    elif scenario == 'linear':
        signal = 0.8 * X[:, 0] - 0.55 * X[:, 1]
    elif scenario == 'nonlinear':
        signal = 0.85 * (X[:, 0] * X[:, 1]) + 0.65 * (X[:, 2] ** 2 - 1)
    else:
        raise ValueError(scenario)
    # Center true susceptibility multiplier to mean one, separating global rate and context.
    signal = np.clip(signal, -4, 4)
    multiplier = np.exp(signal)
    multiplier /= multiplier.mean()
    rate = base_rate * np.exp(pop_effect[:, None]) * multiplier[None, :]
    def sample(exposure):
        p = -np.expm1(-exposure * rate)
        return rng.binomial(1, p).astype(np.float64), p
    train_y, train_p = sample(train_exposure)
    val_y, val_p = sample(1.0)
    test_y, test_p = sample(test_exposure)
    return X, pop_effect, rate, (train_y, val_y, test_y), (train_p, val_p, test_p)


def features(X, kind):
    if kind == 'M2': return np.zeros((len(X), 0))
    if kind == 'linear': return X
    if kind == 'mlp':
        # Fixed nonlinear random-feature basis; only the small readout is fitted.
        # Includes interactions, permitting nonlinear recovery without a large neural network.
        return np.column_stack([X, X[:, 0]*X[:, 1], X[:, 2]**2 - 1, np.tanh(X[:, 0]+X[:, 1])])
    raise ValueError(kind)


def probabilities(theta, F, n_pop, exposure):
    # theta: intercept, centered population deviations, context readout
    a = theta[0]
    u = theta[1:1+n_pop]
    u = u-u.mean()
    if F.shape[1]:
        raw = F @ theta[1+n_pop:]
        raw = np.clip(raw, -12, 12)
        context = np.exp(raw)
        context /= context.mean()
    else:
        context = np.ones(len(F))
    lam = np.exp(np.clip(a+u, -16, 5))[:,None] * context[None,:]
    return -np.expm1(-exposure*lam)


def fit(y, F, penalty=2.0, maxiter=200):
    n_pop = y.shape[0]
    theta0 = np.zeros(1+n_pop+F.shape[1]); theta0[0] = np.log(max(y.mean(), 1e-5))
    def objective(theta):
        p = np.clip(probabilities(theta,F,n_pop,1.0), 1e-10, 1-1e-10)
        nll = -np.mean(y*np.log(p)+(1-y)*np.log1p(-p))
        u = theta[1:1+n_pop]; v = theta[1+n_pop:]
        reg = penalty * (np.mean((u-u.mean())**2) / n_pop + np.sum(v*v)/max(1,F.shape[1])) / y.size
        return nll+reg
    res = minimize(objective, theta0, method='L-BFGS-B', options={'maxiter':maxiter,'ftol':1e-10})
    return res.x, {'success':bool(res.success), 'message':str(res.message), 'nit':int(res.nit), 'objective':float(res.fun)}


def metrics(y, p, true_p):
    p = np.clip(p, 1e-10, 1-1e-10)
    return {'nll':float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))),
            'brier':float(np.mean((y-p)**2)),
            'true_probability_rmse':float(np.sqrt(np.mean((p-true_p)**2))),
            'observed_count':float(y.sum()),'predicted_count':float(p.sum()),
            'true_expected_count':float(true_p.sum())}


def run(seed, scenario, n_sites, dim, penalty, maxiter):
    X, _, _, ys, ps = make_data(seed, scenario, n_sites=n_sites, dim=dim)
    ytr,yval,ytest = ys; _,pval,ptest=ps
    rng=np.random.default_rng(seed+10000)
    rows=[]; fits={}
    for model in ['M2','M3_linear','M3_nonlinear','M3_shuffled']:
        kind={'M2':'M2','M3_linear':'linear','M3_nonlinear':'mlp','M3_shuffled':'mlp'}[model]
        F=features(X,kind)
        if model=='M3_shuffled':
            # Fixed site permutation for historical and future features, destroys true alignment.
            F=F[rng.permutation(len(F))]
        theta,info=fit(ytr,F,penalty=penalty,maxiter=maxiter)
        fits[model]=info
        for split,y,tp in [('validation',yval,pval),('test',ytest,ptest)]:
            pred=probabilities(theta,F,len(y),1.0)
            rows.append({'seed':seed,'scenario':scenario,'model':model,'split':split,
                         'n_sites':n_sites,'fit_success':info['success'],**metrics(y,pred,tp)})
    return rows,fits


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',nargs='+',type=int,default=[11,23,42,67,101])
    ap.add_argument('--scenarios',nargs='+',choices=['null','linear','nonlinear'],default=['null','linear','nonlinear'])
    ap.add_argument('--sites',type=int,default=1800)
    ap.add_argument('--dim',type=int,default=8)
    ap.add_argument('--penalty',type=float,default=2.0)
    ap.add_argument('--maxiter',type=int,default=200)
    ap.add_argument('--output',default='results/phase3a_synthetic')
    args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    all_rows=[];fit_log={}
    for scenario in args.scenarios:
        for seed in args.seeds:
            rows,fits=run(seed,scenario,args.sites,args.dim,args.penalty,args.maxiter)
            all_rows.extend(rows);fit_log[f'{scenario}_{seed}']=fits
            print(f'DONE {scenario} seed={seed}',flush=True)
    with (out/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    (out/'fits.json').write_text(json.dumps(fit_log,indent=2))
    (out/'config.json').write_text(json.dumps(vars(args),indent=2))
    print('Saved',out)

if __name__=='__main__':main()
