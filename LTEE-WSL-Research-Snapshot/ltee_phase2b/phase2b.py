#!/usr/bin/env python3
"""Phase 2B synthetic-only: recovery, shrinkage, rare-event stress tests.
No LTEE files are read. Rates in synthetic units, not biochemical estimates.
"""
import argparse, csv, json, math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
import phase2a as core


def truth_prob(truth, p, a, dt):
    return core.transition(truth['log_rate']+truth['population_log_rates'][p],
                           np.array(truth['global_spectrum'])+np.array(truth['population_spectrum'][p]),dt)[a]


def recovery(fitted, truth, n_pop, horizon=1.0):
    if fitted is None: return {}
    params=np.asarray(fitted['parameters'])
    effects=params[7:7+n_pop] if len(params)>7 else np.zeros(n_pop)
    true=np.asarray(truth['population_log_rates'])
    # Relative rates are identifiable; absolute decomposition into global/population intercept is not.
    centered_true=true-true.mean();centered_est=effects-effects.mean()
    corr=float(np.corrcoef(centered_true,centered_est)[0,1]) if np.std(centered_true)>1e-8 and np.std(centered_est)>1e-8 else None
    matrix_errors=[]
    for p in range(n_pop):
        for a in range(4):
            row=np.array([[p,a,a,horizon]],float)
            estimated=core.predict(row,'M2' if len(params)>7 else 'M1',fitted,n_pop)[0]
            actual=truth_prob(truth,p,a,horizon)
            matrix_errors.extend((estimated-actual)**2)
    return {'centered_population_rate_corr':corr,
            'transition_probability_rmse':float(np.sqrt(np.mean(matrix_errors))),
            'population_rate_centered_rmse':float(np.sqrt(np.mean((centered_est-centered_true)**2)))}


def run_recovery(args,out):
    records=[]; fits=[]
    settings=[('homogeneous',0.,0.,args.train_n),('rate_only',0.8,0.,args.train_n),
              ('rate_spectrum',0.8,0.7,args.train_n),('low_sample',0.8,0.7,args.low_n)]
    for seed in args.seeds:
        for j,(setting,rs,ss,n) in enumerate(settings):
            rng=np.random.default_rng(seed*1009+j)
            train,truth=core.simulate(rng,args.populations,n,args.rate,rs,ss)
            valid,_=core.simulate_from_truth(rng,args.populations,args.validation_n,truth)
            test,_=core.simulate_from_truth(rng,args.populations,args.test_n,truth)
            for model in ('M1','M2_unregularized','M2_shrinkage'):
                candidates=[0.] if model!='M2_shrinkage' else args.penalties
                best=None
                for penalty in candidates:
                    fit=core.fit(train,'M1' if model=='M1' else 'M2',args.populations,penalty,args.maxiter)
                    v=core.metrics(valid,core.predict(valid,'M1' if model=='M1' else 'M2',fit,args.populations))
                    if best is None or v['nucleotide_nll']<best[0]:best=(v['nucleotide_nll'],penalty,fit)
                _,penalty,fit=best
                pred=core.predict(test,'M1' if model=='M1' else 'M2',fit,args.populations)
                metrics=core.metrics(test,pred)
                rec=recovery(fit,truth,args.populations)
                records.append(dict(seed=seed,scenario=setting,model=model,train_n_per_pop=n,
                                    penalty=penalty,fit_success=fit['success'],**metrics,**rec))
                fits.append({'seed':seed,'scenario':setting,'model':model,'selected_penalty':penalty,
                             'fit':fit,'truth':truth})
                print('RECOVERY',seed,setting,model,'nll',round(metrics['nucleotide_nll'],5),flush=True)
    save_csv(out/'recovery_metrics.csv',records)
    (out/'recovery_fits.json').write_text(json.dumps(fits,indent=2))


def rare_simulate(rng,n_pop,n_sites,rate,heterogeneity,horizon):
    """Exact Bernoulli count simulation for independent sites, using CTMC stay probabilities.
    All sites are A initially; count events correspond to final A != A, not number of jumps.
    """
    effects=rng.normal(0,heterogeneity,n_pop)
    counts=[];truth=[]
    for p in range(n_pop):
        q=core.generator(np.log(rate)+effects[p],np.zeros(6))
        change=1.-expm(q*horizon)[0,0]
        counts.append(int(rng.binomial(n_sites,change)))
        truth.append(float(change))
    return np.asarray(counts),np.asarray(truth),effects


def fit_rare(counts,n_sites,horizon,penalty,initial_log_rate):
    """Aggregate count likelihood avoids vanishing per-site gradient and artificial enrichment.
    Poisson approximation for expected rare changes; penalized population log effects.
    """
    counts=np.asarray(counts,float); n_pop=len(counts)
    def objective(z):
        global_log=z[0];effects=z[1:]; effects=effects-effects.mean()
        expected=n_sites*horizon*np.exp(global_log+effects)
        return float(np.sum(expected-counts*np.log(np.maximum(expected,1e-200))) +
                     penalty*np.sum(effects**2))
    z0=np.r_[initial_log_rate,np.zeros(n_pop)]
    result=minimize(objective,z0,method='L-BFGS-B',bounds=[(-25,0)]+[(-8,8)]*n_pop,
                    options={'maxiter':250,'ftol':1e-12})
    e=result.x[1:]-np.mean(result.x[1:]);return result,e


def run_rare(args,out):
    rows=[]
    for seed in args.seeds:
        rng=np.random.default_rng(seed+19001)
        for rate in args.rare_rates:
            counts,true_probs,true_effects=rare_simulate(rng,args.populations,args.genome_sites,rate,args.rare_heterogeneity,args.rare_horizon)
            for pen in [0.,1.,10.,100.]:
                result,est_effects=fit_rare(counts,args.genome_sites,args.rare_horizon,pen,np.log(rate))
                estimated=np.exp(result.x[0]+est_effects)*args.rare_horizon
                truth_center=true_effects-true_effects.mean()
                corr=float(np.corrcoef(est_effects,truth_center)[0,1]) if np.std(est_effects)>1e-8 and np.std(truth_center)>1e-8 else None
                rows.append(dict(seed=seed,rate=rate,penalty=pen,n_pop=args.populations,
                    genome_sites=args.genome_sites,observed_total=int(counts.sum()),
                    expected_true_total=float((true_probs*args.genome_sites).sum()),
                    expected_pred_total=float((estimated*args.genome_sites).sum()),
                    mean_abs_expected_count_error=float(np.mean(np.abs(estimated*args.genome_sites-true_probs*args.genome_sites))),
                    population_rate_corr=corr,fit_success=bool(result.success),
                    gradient_norm_at_opt=float(np.linalg.norm(result.jac)),
                    max_abs_effect=float(np.max(np.abs(est_effects)))))
            print('RARE',seed,rate,'observed',int(counts.sum()),flush=True)
    save_csv(out/'rare_metrics.csv',rows)


def save_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',default='LTEE_prospective/phase2b_synthetic')
    ap.add_argument('--seeds',nargs='+',type=int,default=[11,23,42])
    ap.add_argument('--populations',type=int,default=4)
    ap.add_argument('--train-n',type=int,default=120)
    ap.add_argument('--low-n',type=int,default=25)
    ap.add_argument('--validation-n',type=int,default=60)
    ap.add_argument('--test-n',type=int,default=120)
    ap.add_argument('--rate',type=float,default=0.08)
    ap.add_argument('--penalties',nargs='+',type=float,default=[0.001,0.01,0.1])
    ap.add_argument('--maxiter',type=int,default=80)
    ap.add_argument('--genome-sites',type=int,default=4629812)
    ap.add_argument('--rare-rates',nargs='+',type=float,default=[1e-8,1e-7,1e-6])
    ap.add_argument('--rare-horizon',type=float,default=1.)
    ap.add_argument('--rare-heterogeneity',type=float,default=1.)
    ap.add_argument('--mode',choices=['all','recovery','rare'],default='all')
    args=ap.parse_args()
    if args.populations<2 or args.train_n<1 or args.low_n<1 or args.genome_sites<1 or any(x<=0 for x in args.rare_rates):ap.error('Invalid simulation settings')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(vars(args),indent=2))
    if args.mode in ('all','recovery'):run_recovery(args,out)
    if args.mode in ('all','rare'):run_rare(args,out)
    print('SYNTHETIC ONLY; no LTEE training or test access. Saved:',out)

if __name__=='__main__':main()
