#!/usr/bin/env python3
"""Synthetic four-state CTMC; NumPy/SciPy edition. No torch, no LTEE data."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import expm
from sklearn.metrics import roc_auc_score, average_precision_score

DNA = 'ACGT'
COMP = {0: 3, 1: 2, 2: 1, 3: 0}
PAIRS = [(a, b) for a in range(4) for b in range(4) if a != b]
CLASSES = {}
for a, b in PAIRS:
    key = min((a, b), (COMP[a], COMP[b]))
    if key not in CLASSES:
        CLASSES[key] = len(CLASSES)
CLASS_INDEX = {(a, b): CLASSES[min((a, b), (COMP[a], COMP[b]))] for a, b in PAIRS}
assert len(CLASSES) == 6


def generator(log_spectrum):
    """Mean exit rate normalized to one; last two axes are 4x4."""
    v = np.asarray(log_spectrum, dtype=float)
    rates = np.exp(np.clip(v - v.mean(axis=-1, keepdims=True), -15, 15))
    q = np.zeros(v.shape[:-1] + (4, 4))
    for (a, b), k in CLASS_INDEX.items():
        q[..., a, b] = rates[..., k]
    scale = np.maximum(q.sum(axis=-1).mean(axis=-1), 1e-12)
    q /= scale[..., None, None]
    for a in range(4):
        q[..., a, a] = -q[..., a, :].sum(axis=-1)
    return q


def transition(q, exposure):
    """Batched 4x4 CTMC exp via uniformization, stable for moderate exposure.

    For unusually large exposure, fall back to scipy.linalg.expm.
    """
    q = np.asarray(q, float)
    e = np.asarray(exposure, float)
    q = np.broadcast_to(q, np.broadcast_shapes(q.shape[:-2], e.shape) + (4, 4))
    e = np.broadcast_to(e, q.shape[:-2])
    shape = e.shape
    q = q.reshape(-1, 4, 4)
    e = e.reshape(-1)
    if np.any(e < 0):
        raise ValueError('negative exposure')
    result = np.empty_like(q)
    eye = np.eye(4)
    a = q * e[:, None, None]
    nu = np.max(-np.diagonal(a, axis1=-2, axis2=-1), axis=-1)
    small = nu <= 24
    if np.any(small):
        aa = a[small]
        nn = np.maximum(nu[small], 1e-12)
        r = eye + aa / nn[:, None, None]
        term = np.broadcast_to(eye, r.shape).copy()
        weight = np.exp(-nn)
        total = weight[:, None, None] * term
        cumulative = weight.copy()
        for k in range(1, 150):
            term = term @ r
            weight = weight * nn / k
            total += weight[:, None, None] * term
            cumulative += weight
            if np.all(1 - cumulative < 1e-13):
                break
        total = np.maximum(total, 0)
        total /= total.sum(axis=-1, keepdims=True)
        result[small] = total
    for j in np.flatnonzero(~small):
        result[j] = expm(a[j])
    return result.reshape(shape + (4, 4))


def make_data(seed, scenario, n_pop, n_sites, dim, n_train, n_val, n_test):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_sites, dim))
    origin = rng.integers(0, 4, size=n_sites)
    pop_log = rng.normal(0, .48, size=n_pop)
    spectra = rng.normal(0, .32, size=(n_pop, 6))
    if scenario == 'homogeneous':
        pop_log *= 0
        spectra *= 0
    if scenario == 'rate_only':
        spectra *= 0
    beta = rng.normal(0, .6, size=dim)
    beta = beta / np.linalg.norm(beta) * .85
    context = np.zeros(n_sites)
    if scenario in ('linear', 'nonlinear'):
        context = x @ beta
    if scenario == 'nonlinear':
        context += .9 * x[:, 0] * x[:, 1]
    context -= np.log(np.exp(context).mean())
    base_rate = .065
    q = generator(spectra)
    datasets = {}
    for split, n in [('train', n_train), ('validation', n_val), ('test', n_test)]:
        p = rng.integers(n_pop, size=n)
        s = rng.integers(n_sites, size=n)
        dt = rng.uniform(.5, 1.8, size=n)
        expo = base_rate * np.exp(pop_log[p] + context[s]) * dt
        probs = transition(q[p], expo)[np.arange(n), origin[s]]
        y = np.array([rng.choice(4, p=v) for v in probs])
        datasets[split] = dict(p=p, s=s, dt=dt, y=y, true_probs=probs)
    truth = dict(population_log_rates=pop_log.tolist(), spectrum=spectra.tolist(), base_rate=base_rate)
    return x, origin, datasets, truth


class Model:
    def __init__(self, n_pop, dim, model):
        self.n_pop, self.dim, self.model = n_pop, dim, model
        self.names = ['global_log_rate', 'global_spectrum']
        self.sizes = [1, 6]
        if model != 'M1':
            self.names += ['pop_rate', 'pop_spectrum']
            self.sizes += [n_pop, n_pop * 6]
        if model.startswith('M3'):
            self.names += ['linear']
            self.sizes += [dim]
            if model in ('M3_nonlinear', 'M3_shuffled'):
                self.names += ['interaction']
                self.sizes += [1]
        self.theta = np.zeros(sum(self.sizes))
        self.theta[0] = -2.8

    def unpack(self, theta=None):
        theta = self.theta if theta is None else theta
        parts = {}
        j = 0
        for name, size in zip(self.names, self.sizes):
            parts[name] = theta[j:j + size]
            j += size
        if 'pop_spectrum' in parts:
            parts['pop_spectrum'] = parts['pop_spectrum'].reshape(self.n_pop, 6)
        return parts

    def predict(self, data, x, origin, theta=None):
        z = self.unpack(theta)
        p, s, dt = data['p'], data['s'], data['dt']
        lr = np.full(len(p), z['global_log_rate'][0])
        spec = np.broadcast_to(z['global_spectrum'], (len(p), 6)).copy()
        if self.model != 'M1':
            lr += (z['pop_rate'] - z['pop_rate'].mean())[p]
            spec += (z['pop_spectrum'] - z['pop_spectrum'].mean(axis=0))[p]
        if self.model.startswith('M3'):
            raw = x @ z['linear']
            if 'interaction' in z:
                raw += z['interaction'][0] * x[:, 0] * x[:, 1]
            # Full-site normalization: no sampled-label prevalence in normalization.
            center = np.log(np.exp(np.clip(raw - raw.max(), -100, 0)).mean()) + raw.max()
            lr += (raw - center)[s]
        expo = np.exp(np.clip(lr, -15, 8)) * dt
        q = generator(spec)
        return transition(q, expo)[np.arange(len(s)), origin[s]]


def objective(theta, model, data, x, origin, penalty):
    prob = model.predict(data, x, origin, theta)
    nll = -np.log(np.clip(prob[np.arange(len(data['y'])), data['y']], 1e-12, 1)).mean()
    z = model.unpack(theta)
    if 'pop_rate' in z:
        nll += penalty * (np.mean(z['pop_rate'] ** 2) + np.mean(z['pop_spectrum'] ** 2))
    if 'linear' in z:
        nll += .002 * (np.sum(z['linear'] ** 2) + np.sum(z.get('interaction', np.zeros(1)) ** 2))
    return float(nll)


def fit(model, train, val, x, origin, epochs, penalty, seed=None, device=None):
    # L-BFGS-B with finite-difference gradients; no torch dependency.
    best = [float('inf'), model.theta.copy()]
    iterations = [0]
    max_grad = [0.]
    def callback(theta):
        iterations[0] += 1
        v = objective(theta, model, val, x, origin, 0.)
        if v < best[0]:
            best[0], best[1] = v, theta.copy()
    result = minimize(objective, model.theta, args=(model, train, x, origin, penalty),
                      method='L-BFGS-B', jac='2-point', callback=callback,
                      options={'maxiter': epochs, 'ftol': 1e-9, 'maxls': 20})
    callback(result.x)
    model.theta = best[1]
    max_grad[0] = float(np.linalg.norm(result.jac)) if result.jac is not None else np.nan
    return dict(best_validation_nll=best[0], epochs_ran=int(result.nit),
                max_gradient_norm=max_grad[0], fit_success=bool(result.success),
                optimizer_message=str(result.message))


def evaluate(model, data, x, origin, device=None):
    probs = model.predict(data, x, origin)
    y, s = data['y'], data['s']
    p_y = np.clip(probs[np.arange(len(y)), y], 1e-12, 1)
    changed = (y != origin[s]).astype(int)
    p_change = np.clip(1 - probs[np.arange(len(y)), origin[s]], 0, 1)
    mutated = np.flatnonzero(changed)
    cond = [-np.log(np.clip(probs[i, y[i]] / max(p_change[i], 1e-12), 1e-12, 1)) for i in mutated]
    return dict(nll=float(-np.log(p_y).mean()),
                brier_change=float(np.mean((p_change - changed) ** 2)),
                true_probability_rmse=float(np.sqrt(np.mean((probs - data['true_probs']) ** 2))),
                mutation_auroc=float(roc_auc_score(changed, p_change)) if len(set(changed)) == 2 else np.nan,
                mutation_auprc=float(average_precision_score(changed, p_change)) if len(set(changed)) == 2 else np.nan,
                conditional_substitution_nll=float(np.mean(cond)) if cond else np.nan,
                observed_count=int(changed.sum()), predicted_count=float(p_change.sum()),
                true_expected_count=float((1 - data['true_probs'][np.arange(len(y)), origin[s]]).sum()))


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows, fits = [], []
    for seed in args.seeds:
        for scenario in args.scenarios:
            x, origin, data, truth = make_data(seed, scenario, args.populations, args.sites,
                                               args.dim, args.train, args.validation, args.test)
            for name in ['M1', 'M2', 'M3_linear', 'M3_nonlinear', 'M3_shuffled']:
                xuse = x.copy()
                if name == 'M3_shuffled':
                    xuse = xuse[np.random.default_rng(seed + 998).permutation(len(xuse))]
                model = Model(args.populations, args.dim, name)
                diag = fit(model, data['train'], data['validation'], xuse, origin,
                           args.epochs, args.penalty, seed)
                metrics = evaluate(model, data['test'], xuse, origin)
                rows.append(dict(seed=seed, scenario=scenario, model=name, **metrics, **diag))
                z = model.unpack()
                fits.append(dict(seed=seed, scenario=scenario, model=name, fit=diag,
                                 true_parameters=truth, fitted_global_log_rate=float(z['global_log_rate'][0]),
                                 fitted_pop_rates=z.get('pop_rate', np.zeros(args.populations)).tolist()))
                print(seed, scenario, name, 'NLL', round(metrics['nll'], 5), flush=True)
    pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)
    (out / 'fits.json').write_text(json.dumps(fits, indent=2))
    (out / 'config.json').write_text(json.dumps(vars(args), indent=2))
    print('Saved:', out)


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--seeds', nargs='+', type=int, default=[11, 23, 42])
    a.add_argument('--scenarios', nargs='+', choices=['homogeneous', 'rate_only', 'rate_spectrum', 'linear', 'nonlinear'],
                   default=['homogeneous', 'rate_spectrum', 'linear', 'nonlinear'])
    a.add_argument('--populations', type=int, default=4)
    a.add_argument('--sites', type=int, default=240)
    a.add_argument('--dim', type=int, default=6)
    a.add_argument('--train', type=int, default=900)
    a.add_argument('--validation', type=int, default=240)
    a.add_argument('--test', type=int, default=400)
    a.add_argument('--epochs', type=int, default=110)
    a.add_argument('--penalty', type=float, default=.003)
    a.add_argument('--output', default='../LTEE_prospective/phase3b_synthetic')
    args = a.parse_args()
    if min(args.sites, args.populations, args.train, args.validation, args.test) <= 0:
        a.error('sizes must be positive')
    if args.dim < 2:
        a.error('dim must be >= 2 for nonlinear interaction')
    run(args)


if __name__ == '__main__':
    main()
