#!/usr/bin/env python3
"""Development-only, leakage-separated calibration confirmation for two-site forecasts.
Uses the exact V3-B simulator; does not use future targets as features.
"""
import argparse, csv, hashlib, importlib.util, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar

EPS=1e-12

def load(path):
    spec=importlib.util.spec_from_file_location('v3b_reference',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def pair_counts(pop):
    x=pop[:,0::2];y=pop[:,1::2]
    return np.stack([((x==0)&(y==0)).sum(0),((x==0)&(y==1)).sum(0),((x==1)&(y==0)).sum(0),((x==1)&(y==1)).sum(0)],axis=1).astype(float)

def probs(counts,alpha,mix,rho):
    n=counts.sum(1)
    a=(counts[:,2]+counts[:,3]+alpha)/(n+2*alpha)
    b=(counts[:,1]+counts[:,3]+alpha)/(n+2*alpha)
    indep=np.stack([(1-a)*(1-b),(1-a)*b,a*(1-b),a*b],axis=1)
    # Smooth joint counts with the independence prior; preserves smoothed marginals.
    raw=counts/counts.sum(1,keepdims=True)
    # Historical dependence D from raw counts; clamp to feasible smoothed margins.
    d=raw[:,3]-(raw[:,2]+raw[:,3])*(raw[:,1]+raw[:,3])
    lower=np.maximum(0,a+b-1)-a*b
    upper=np.minimum(a,b)-a*b
    d=np.clip(rho*d,lower,upper)
    out=indep+np.stack([d,-d,-d,d],axis=1)
    out=(1-mix)*out+mix/4
    return np.clip(out,EPS,1)

def score(pred,targets):
    counts=pair_counts(targets)
    return float(-(counts*np.log(pred)).sum()/counts.sum())

def make_examples(sim,args,seeds,scenarios):
    examples=[]
    for scenario in scenarios:
        for seed in seeds:
            data=sim.simulate(scenario,seed,args)
            by_time={}
            for row in data['observations']:
                by_time.setdefault(int(row['generation']),[]).append(row['sequence'])
            times=sorted(by_time)
            for cutoff in args.cutoffs:
                if cutoff not in by_time:continue
                hist=np.asarray([list(map(int,s)) for s in by_time[cutoff]],dtype=np.uint8)
                for h in args.horizons:
                    if cutoff+h not in by_time:continue
                    target=np.asarray([list(map(int,s)) for s in by_time[cutoff+h]],dtype=np.uint8)
                    examples.append((scenario,seed,cutoff,h,pair_counts(hist),target))
    return examples

def mean_nll(examples,alpha,mix,rho):
    return float(np.mean([score(probs(c,alpha,mix,rho),t) for _,_,_,_,c,t in examples]))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train-seeds',type=int,nargs='+',default=[311,313,317,331])
    p.add_argument('--calibration-seeds',type=int,nargs='+',default=[401,409,419])
    p.add_argument('--confirmation-seeds',type=int,nargs='+',default=[503,509,521])
    p.add_argument('--scenarios',nargs='+',default=['neutral','recombination','selection_only','epistasis'])
    p.add_argument('--cutoffs',type=int,nargs='+',default=[20,50])
    p.add_argument('--horizons',type=int,nargs='+',default=[10,20,30])
    p.add_argument('--alphas',type=float,nargs='+',default=[0,.1,.5,1,2,5])
    p.add_argument('--mixes',type=float,nargs='+',default=[0,.001,.01,.05])
    p.add_argument('--individuals',type=int,default=120)
    p.add_argument('--sites',type=int,default=24)
    p.add_argument('--generations',type=int,default=100)
    p.add_argument('--observe-step',type=int,default=10)
    p.add_argument('--sample-size',type=int,default=8)
    p.add_argument('--rate',type=float,default=.001)
    p.add_argument('--initial-frequency',type=float,default=.25)
    p.add_argument('--context-sigma',type=float,default=1.)
    p.add_argument('--heterogeneity-multiplier',type=float,default=3.)
    p.add_argument('--shift-generation',type=int,default=50)
    p.add_argument('--shift-multiplier',type=float,default=4.)
    p.add_argument('--recombination-probability',type=float,default=.5)
    p.add_argument('--selection-strength',type=float,default=.12)
    p.add_argument('--epistasis-strength',type=float,default=.25)
    a=p.parse_args()
    splits=[a.train_seeds,a.calibration_seeds,a.confirmation_seeds]
    if any(set(splits[i])&set(splits[j]) for i in range(3) for j in range(i+1,3)):p.error('seed splits overlap')
    if a.sites%2:p.error('sites must be even')
    if any(x<0 for x in a.alphas) or any(x<0 or x>=1 for x in a.mixes):p.error('invalid smoothing grid')
    if a.output.exists() and any(a.output.iterdir()):p.error('output must be absent or empty')
    sim=load(a.reference)
    train=make_examples(sim,a,a.train_seeds,a.scenarios)
    cal=make_examples(sim,a,a.calibration_seeds,a.scenarios)
    if not train or not cal:raise ValueError('empty train or calibration folds')
    # Fit persistence on training only with a predeclared stable alpha=1, mix=0.
    opt=minimize_scalar(lambda rho:mean_nll(train,1.,0.,rho),bounds=(0,1),method='bounded',options={'xatol':1e-5})
    if not opt.success:raise RuntimeError('rho optimization failed')
    rho=float(opt.x)
    grid=[]
    for alpha in a.alphas:
        for mix in a.mixes:
            grid.append({'alpha':alpha,'mix':mix,'calibration_nll':mean_nll(cal,alpha,mix,rho)})
    chosen=min(grid,key=lambda x:(x['calibration_nll'],x['alpha'],x['mix']))
    # Confirmation outcomes are generated and scored ONLY after rho and calibration choice frozen.
    confirm=make_examples(sim,a,a.confirmation_seeds,a.scenarios)
    if not confirm:raise ValueError('empty confirmation folds')
    rows=[]
    for scenario,seed,cutoff,h,c,t in confirm:
        configs=[('sampled_independent_unsmoothed',0.,0.,0.),('M5_unsmoothed',0.,0.,rho),
                 ('sampled_independent_calibrated',chosen['alpha'],chosen['mix'],0.),
                 ('M5_calibrated',chosen['alpha'],chosen['mix'],rho),
                 ('M5_calibrated_pairing_removed',chosen['alpha'],chosen['mix'],0.)]
        for name,alpha,mix,r in configs:
            pred=probs(c,alpha,mix,r)
            rows.append({'scenario':scenario,'seed':seed,'cutoff':cutoff,'horizon':h,'model':name,
                         'joint_nll':score(pred,t),'min_probability':float(pred.min())})
    a.output.mkdir(parents=True)
    with (a.output/'confirmation_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    with (a.output/'calibration_grid.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=grid[0]);w.writeheader();w.writerows(grid)
    summary={name:float(np.mean([r['joint_nll'] for r in rows if r['model']==name])) for name in sorted({r['model'] for r in rows})}
    report={'status':'EXPLORATORY_HELDOUT_DEVELOPMENT','reference_sha256':hashlib.sha256(a.reference.read_bytes()).hexdigest(),
            'train_seeds':a.train_seeds,'calibration_seeds':a.calibration_seeds,'confirmation_seeds':a.confirmation_seeds,
            'rho_train_only':rho,'chosen_calibration':chosen,'n_confirmation_folds':len(confirm),
            'confirmation_mean_joint_nll':summary,'limitations':['Previously explored simulator family; not a locked confirmatory test',
            'Independent baseline is sampled-current-marginal, not full-state','M5 is a single learned global dependence coefficient, not original M5 History']}
    (a.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
