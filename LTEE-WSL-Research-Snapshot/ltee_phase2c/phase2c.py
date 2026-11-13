#!/usr/bin/env python3
"""Phase 2C: strictly synthetic held-out rare-count forecasting.
No LTEE files are read. Python 3.9+, numpy, scipy, pandas.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

SCENARIOS = ('stationary', 'rate_shift')

def simulate(seed, rate, exposure, n_pop=12, n_train=4, n_test=3, scenario='stationary'):
    rng = np.random.default_rng(seed)
    alpha = rng.normal(0, 0.65, n_pop)
    alpha -= alpha.mean()
    global_rate = rate
    E_train = np.full((n_pop, n_train), exposure, dtype=float)
    E_test = np.full((n_pop, n_test), exposure, dtype=float)
    shift = np.zeros(n_pop)
    if scenario == 'rate_shift':
        shift = rng.normal(0.5, 0.35, n_pop)  # future unobserved regime shift
    mu_train = E_train * global_rate * np.exp(alpha[:, None])
    mu_test = E_test * global_rate * np.exp((alpha + shift)[:, None])
    return {'train': rng.poisson(mu_train), 'test': rng.poisson(mu_test),
            'e_train': E_train, 'e_test': E_test, 'true_train': mu_train,
            'true_test': mu_test, 'alpha': alpha, 'shift': shift}


def fit_counts(y, e, penalty=1., hierarchical=True):
    """Fit Poisson likelihood on training intervals only. Center alpha for identifiability."""
    n_pop = y.shape[0]
    assert np.all(e > 0) and np.all(y >= 0)
    pooled = max(y.sum() / e.sum(), 1e-12)
    if not hierarchical:
        return {'log_rate': float(np.log(pooled)), 'alpha': np.zeros(n_pop),
                'success': True, 'message': 'closed_form', 'grad_norm': 0., 'iterations': 0}

    def objective(x):
        b = x[0]
        raw = x[1:]
        a = raw - raw.mean()
        eta = np.clip(b + a[:, None], -35, 15)
        mu = e * np.exp(eta)
        loss = np.sum(mu - y * eta) + 0.5 * penalty * np.dot(a, a)
        grad_eta = np.sum(mu - y, axis=1)
        grad_b = grad_eta.sum()
        grad_raw = grad_eta - grad_eta.mean() + penalty * a
        return float(loss), np.r_[grad_b, grad_raw]

    x0 = np.r_[np.log(pooled), np.zeros(n_pop)]
    res = minimize(objective, x0, jac=True, method='L-BFGS-B',
                   options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
    a = res.x[1:] - res.x[1:].mean()
    return {'log_rate': float(res.x[0]), 'alpha': a, 'success': bool(res.success),
            'message': str(res.message), 'grad_norm': float(np.linalg.norm(objective(res.x)[1])),
            'iterations': int(res.nit)}


def predict(fit, e):
    return e * np.exp(np.clip(fit['log_rate'] + fit['alpha'][:, None], -35, 15))


def metrics(y, mu, truth):
    mu = np.maximum(mu, 1e-12)
    dev = 2 * np.sum(np.where(y > 0, y * np.log(np.maximum(y, 1e-12) / mu) - (y - mu), mu))
    return {'heldout_poisson_nll': float(np.mean(mu - y * np.log(mu) + gammaln(y + 1))),
            'heldout_deviance_per_cell': float(dev / y.size),
            'true_intensity_mae': float(np.mean(np.abs(mu - truth))),
            'observed_total': int(y.sum()), 'true_expected_total': float(truth.sum()),
            'predicted_total': float(mu.sum()),
            'total_error_vs_truth': float(mu.sum() - truth.sum()),
            'total_error_vs_observed': float(mu.sum() - y.sum())}


def corr(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12: return None
    return float(np.corrcoef(a, b)[0, 1])


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows, fits = [], []
    for scenario in SCENARIOS:
        for rate in args.rates:
            for exposure in args.exposures:
                for seed in args.seeds:
                    data = simulate(seed, rate, exposure, args.populations,
                                    args.train_intervals, args.test_intervals, scenario)
                    for model, penalty in [('M1_shared', None),
                                           ('M2_unregularized', 0.),
                                           ('M2_shrink_1', 1.),
                                           ('M2_shrink_10', 10.)]:
                        fit = fit_counts(data['train'], data['e_train'],
                                         penalty or 0., hierarchical=model != 'M1_shared')
                        pred = predict(fit, data['e_test'])
                        met = metrics(data['test'], pred, data['true_test'])
                        met.update(scenario=scenario, rate=rate, exposure=exposure,
                                   seed=seed, model=model,
                                   train_observed_total=int(data['train'].sum()),
                                   train_expected_total=float(data['true_train'].sum()),
                                   fit_success=fit['success'], grad_norm=fit['grad_norm'],
                                   iterations=fit['iterations'],
                                   population_rate_corr=corr(fit['alpha'], data['alpha']),
                                   centered_rate_rmse=float(np.sqrt(np.mean((fit['alpha']-data['alpha'])**2))))
                        rows.append(met)
                        fits.append({'scenario':scenario,'rate':rate,'exposure':exposure,
                                     'seed':seed,'model':model,'optimizer_message':fit['message'],
                                     'estimated_log_rate':fit['log_rate'],
                                     'estimated_centered_population_effects':fit['alpha'].tolist(),
                                     'true_centered_population_effects':data['alpha'].tolist(),
                                     'unobserved_future_shift':data['shift'].tolist()})
    pd.DataFrame(rows).to_csv(out/'heldout_metrics.csv',index=False)
    (out/'fit_diagnostics.json').write_text(json.dumps(fits,indent=2))
    (out/'config.json').write_text(json.dumps(vars(args),indent=2))
    print('Saved', out)
    df = pd.DataFrame(rows)
    print(df.groupby(['scenario','rate','model'])[['heldout_poisson_nll','true_intensity_mae']].mean().to_string())
    print('IMPORTANT: all evaluation is synthetic; future shifts are not supplied to fitted models.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',type=int,nargs='+',default=[11,23,42,67,101])
    p.add_argument('--rates',type=float,nargs='+',default=[1e-8,1e-7,1e-6])
    p.add_argument('--exposures',type=float,nargs='+',default=[1e5,1e6,1e7])
    p.add_argument('--populations',type=int,default=12)
    p.add_argument('--train-intervals',type=int,default=4)
    p.add_argument('--test-intervals',type=int,default=3)
    p.add_argument('--output',default='../LTEE_prospective/phase2c_synthetic')
    a=p.parse_args()
    if min(a.rates)<=0 or min(a.exposures)<=0 or a.populations<2 or a.train_intervals<1 or a.test_intervals<1:
        p.error('rates/exposures must be positive; populations >=2; intervals >=1')
    run(a)
if __name__=='__main__':main()
