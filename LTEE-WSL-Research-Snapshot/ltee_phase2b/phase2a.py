#!/usr/bin/env python3
"""Phase 2A: synthetic CTMC recovery. No LTEE data is read.
Requires numpy, scipy. Outputs synthetic-only diagnostics.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

DNA = 'ACGT'
PAIRS = [(a,b) for a in range(4) for b in range(4) if a != b]
# Six reversible-complement classes, normalized so average off-diagonal rate is one.
RC = lambda a: 3-a  # A<->T, C<->G with A,C,G,T indices
CLASSES = sorted({min((a,b),(RC(a),RC(b))) for a,b in PAIRS})
CLASS_INDEX = np.array([CLASSES.index(min((a,b),(RC(a),RC(b)))) for a,b in PAIRS])
assert len(CLASSES)==6

def generator(log_rate, log_spectrum):
    rates=np.exp(np.asarray(log_spectrum)[CLASS_INDEX]); rates=rates/rates.mean()
    q=np.zeros((4,4))
    for (a,b),r in zip(PAIRS,rates): q[a,b]=np.exp(log_rate)*r/3.0
    np.fill_diagonal(q,-q.sum(axis=1))
    return q

def transition(log_rate, log_spectrum, dt):
    return expm(generator(log_rate,log_spectrum)*dt)

def simulate(rng, n_pop, n_per_pop, rate, heterogeneity, spectrum_heterogeneity, horizon_min=0.5, horizon_max=2.0):
    global_spectrum=np.array([0.35,-0.25,0.15,-0.15,0.20,-0.30])
    pop_rate=rng.normal(0,heterogeneity,n_pop)
    pop_spec=rng.normal(0,spectrum_heterogeneity,(n_pop,6))
    # Center spectrum: remove nonidentifiable common shift.
    pop_spec-=pop_spec.mean(axis=1,keepdims=True)
    rows=[]
    for p in range(n_pop):
        x=rng.integers(0,4,n_per_pop)
        dt=rng.choice([0.5,1.0,2.0],n_per_pop)
        y=np.empty(n_per_pop,dtype=int)
        for i in range(n_per_pop):
            mat=transition(np.log(rate)+pop_rate[p],global_spectrum+pop_spec[p],dt[i])
            y[i]=rng.choice(4,p=mat[x[i]])
        rows.extend((p,int(a),int(b),float(t)) for a,b,t in zip(x,y,dt))
    return np.array(rows),{'log_rate':np.log(rate),'population_log_rates':pop_rate.tolist(),
                           'global_spectrum':global_spectrum.tolist(), 'population_spectrum':pop_spec.tolist()}

def nll(params, rows, model, n_pop, penalty=0):
    log_rate=params[0]
    spectrum=params[1:7]
    if model=='M2':
        pop_rates=params[7:7+n_pop]
        effects=params[7+n_pop:].reshape(n_pop,6)
        effects=effects-effects.mean(axis=1,keepdims=True)
    else:
        pop_rates=np.zeros(n_pop);effects=np.zeros((n_pop,6))
    # Aggregate identical (population, source, target, horizon) observations.
    # Matrix exponentials are computed once per population/horizon.
    loss=0.
    for p in range(n_pop):
        subset=rows[rows[:,0]==p]
        for t in np.unique(subset[:,3]):
            block=subset[subset[:,3]==t]
            mat=transition(log_rate+pop_rates[p],spectrum+effects[p],t)
            src=block[:,1].astype(int);dst=block[:,2].astype(int)
            loss-=np.log(np.maximum(mat[src,dst],1e-300)).sum()
    loss/=len(rows)
    if model=='M2':
        loss+=penalty*(np.mean(pop_rates**2)+np.mean(effects**2))
    return loss

def fit(rows,model,n_pop,penalty,maxiter):
    if model=='M0':return None
    initial=np.r_[np.log(0.05),np.zeros(6)]
    if model=='M2':initial=np.r_[initial,np.zeros(n_pop),np.zeros(n_pop*6)]
    bounds=[(-12,1)]+[(-3,3)]*6
    if model=='M2':bounds += [(-4,4)]*n_pop+[(-3,3)]*(6*n_pop)
    result=minimize(nll,initial,args=(rows,model,n_pop,penalty),method='L-BFGS-B',
                    bounds=bounds,options={'maxiter':maxiter,'ftol':1e-9})
    return {'parameters':result.x.tolist(),'success':bool(result.success),'message':str(result.message),
            'iterations':int(result.nit),'train_nll':float(result.fun),
            'max_abs_parameter_change':float(np.max(np.abs(result.x-initial)))}

def predict(rows,model,fit_result,n_pop):
    if model=='M0':
        return np.eye(4)[rows[:,1].astype(int)]
    params=np.array(fit_result['parameters']);log_rate=params[0];spectrum=params[1:7]
    rates=np.zeros(n_pop);effects=np.zeros((n_pop,6))
    if model=='M2':
        rates=params[7:7+n_pop]
        effects=params[7+n_pop:].reshape(n_pop,6)
        effects-=effects.mean(axis=1,keepdims=True)
    return np.array([transition(log_rate+rates[int(p)],spectrum+effects[int(p)],t)[int(a)]
                     for p,a,_,t in rows])

def metrics(rows,probs):
    y=rows[:,2].astype(int);x=rows[:,1].astype(int)
    true_prob=probs[np.arange(len(y)),y]
    changed=(x!=y)
    pred_change=1-probs[np.arange(len(y)),x]
    return {'n':len(y),'change_fraction':float(changed.mean()),
            'predicted_change_fraction':float(pred_change.mean()),
            'nucleotide_nll':float(-np.log(np.maximum(true_prob,1e-300)).mean()),
            'nucleotide_brier':float(np.mean(np.sum((probs-np.eye(4)[y])**2,axis=1))),
            'mutation_brier':float(np.mean((pred_change-changed)**2)),
            'nucleotide_accuracy':float((probs.argmax(axis=1)==y).mean())}

def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    scenarios={'homogeneous':(0.,0.),'rate_heterogeneity':(1.0,0.),
               'rate_and_spectrum':(1.0,0.8),'low_sample':(1.0,0.8)}
    results=[];fit_records=[]
    for seed in args.seeds:
        for scenario,(rate_sd,spec_sd) in scenarios.items():
            rng=np.random.default_rng(seed*100+list(scenarios).index(scenario))
            n=args.low_sample_n if scenario=='low_sample' else args.samples_per_population
            train,truth=simulate(rng,args.populations,n,args.rate,rate_sd,spec_sd)
            valid,_=simulate_from_truth(rng,args.populations,args.validation_n,truth)
            test,_=simulate_from_truth(rng,args.populations,args.test_n,truth)
            # Select shrinkage using validation ONLY, never test.
            for model in ['M0','M1','M2']:
                candidates=[(0.,None)] if model=='M0' else [(0.,None)] if model=='M1' else [(v,None) for v in args.penalties]
                options=[]
                for pen,_ in candidates:
                    fitted=fit(train,model,args.populations,pen,args.maxiter)
                    val_metric=metrics(valid,predict(valid,model,fitted,args.populations))
                    options.append((val_metric['nucleotide_nll'],pen,fitted))
                _,best_pen,best_fit=min(options,key=lambda x:x[0])
                for split,rows in [('validation',valid),('test',test)]:
                    m=metrics(rows,predict(rows,model,best_fit,args.populations))
                    results.append(dict(seed=seed,scenario=scenario,model=model,split=split,
                                        penalty=best_pen,**m))
                fit_records.append(dict(seed=seed,scenario=scenario,model=model,selected_penalty=best_pen,
                                        fit=best_fit,truth=truth if model=='M2' else None))
                print(seed,scenario,model,'penalty',best_pen,'test_nll',round(results[-1]['nucleotide_nll'],5),flush=True)
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    (out/'fits.json').write_text(json.dumps(fit_records,indent=2))
    (out/'experiment.json').write_text(json.dumps(vars(args),indent=2))
    print('Saved',out)

def simulate_from_truth(rng,n_pop,n_per_pop,truth):
    rows=[]
    for p in range(n_pop):
        x=rng.integers(0,4,n_per_pop)
        dt=rng.choice([0.5,1.0,2.0],n_per_pop)
        y=[]
        for a,t in zip(x,dt):
            mat=transition(truth['log_rate']+truth['population_log_rates'][p],
                           np.array(truth['global_spectrum'])+np.array(truth['population_spectrum'][p]),t)
            y.append(rng.choice(4,p=mat[a]))
        rows.extend((p,int(a),int(b),float(t)) for a,b,t in zip(x,y,dt))
    return np.array(rows),truth

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='LTEE_prospective/phase2a_synthetic')
    parser.add_argument('--seeds',type=int,nargs='+',default=[11,23,42])
    parser.add_argument('--populations',type=int,default=4)
    parser.add_argument('--samples-per-population',type=int,default=120)
    parser.add_argument('--low-sample-n',type=int,default=25)
    parser.add_argument('--validation-n',type=int,default=60)
    parser.add_argument('--test-n',type=int,default=120)
    parser.add_argument('--rate',type=float,default=0.08)
    parser.add_argument('--penalties',type=float,nargs='+',default=[0.001,0.01,0.1])
    parser.add_argument('--maxiter',type=int,default=80)
    args=parser.parse_args()
    assert args.populations>=2 and args.rate>0 and args.low_sample_n>0
    run(args)
