#!/usr/bin/env python3
"""Stage 4.1B: train-only fitting of pairwise dependence persistence.

Imports the original stage4.py. Never reads simulation truth; target files are
read only by the training/evaluation loader for their assigned split.
"""
import argparse, csv, hashlib, importlib.util, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

MODELS = ('M1_independent','M5_fixed','M5_global','M5_horizon','M5_history','M5_history_shuffled')
EPS=1e-9

def load_source(path):
    spec=importlib.util.spec_from_file_location('production_stage4',str(path.resolve()))
    if spec is None or spec.loader is None: raise ValueError('Cannot import stage4')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    for name in ('load_folds','independent','score','normalize'):
        if not hasattr(m,name): raise ValueError('Missing production function '+name)
    return m

def features(row,kind):
    h=float(row['horizon'])/10.0
    q=row['latest']; prev=row['previous']
    d=q[:,3]-(q[:,2]+q[:,3])*(q[:,1]+q[:,3])
    previous_d=(prev[:,3]-(prev[:,2]+prev[:,3])*(prev[:,1]+prev[:,3])) if prev is not None else np.zeros(len(q))
    v=np.abs(d)
    change=np.abs(d-previous_d)
    if kind=='global': return np.ones((len(q),1))
    if kind=='horizon': return np.column_stack([np.ones(len(q)),np.full(len(q),h)])
    if kind=='history': return np.column_stack([np.ones(len(q)),np.full(len(q),h),v,change])
    raise ValueError(kind)

def shuffle_joint(row,seed):
    mat=row['latest_matrix'];pairs=row['pairs'];rng=np.random.default_rng(seed)
    out=[]
    for i,j in pairs:
        x=mat[:,i];y=mat[rng.permutation(len(mat)),j]
        out.append(np.bincount(2*x+y,minlength=4)/len(x))
    q=np.asarray(out,float)
    if not np.allclose(row['latest'].sum(axis=1),1): raise ValueError('Invalid original')
    if not np.allclose(q[:,2]+q[:,3],row['latest'][:,2]+row['latest'][:,3]):raise ValueError('Marginal 1 changed')
    if not np.allclose(q[:,1]+q[:,3],row['latest'][:,1]+row['latest'][:,3]):raise ValueError('Marginal 2 changed')
    return q

def predict(row,theta,kind,prod,shuffle=False,shuffle_seed=0):
    latest=shuffle_joint(row,shuffle_seed) if shuffle else row['latest']
    base=prod.independent(latest)
    if kind=='independent':return prod.normalize(base)
    if kind=='fixed':rho=np.full(len(latest),0.5)
    else:rho=expit(np.clip(features(row,kind)@theta,-30,30))
    # The two inputs share marginals. Mixture preserves them exactly.
    return prod.normalize(base+rho[:,None]*(latest-base))

def fold_nll(row,q,prod):return prod.score(q,row['truth'])['joint_nll']

def objective(theta,rows,kind,prod,penalty):
    total=0.;count=0
    for r in rows:
        q=predict(r,theta,kind,prod)
        total+=fold_nll(r,q,prod);count+=1
    return total/count+penalty*float(np.mean(theta**2))

def fit(rows,kind,prod,penalty):
    dim={'global':1,'horizon':2,'history':4}[kind]
    x0=np.zeros(dim)
    result=minimize(objective,x0,args=(rows,kind,prod,penalty),method='L-BFGS-B',options={'maxiter':300,'ftol':1e-11})
    if not result.success: raise RuntimeError('Optimization failed: '+str(result.message))
    return result.x,{'success':bool(result.success),'iterations':int(result.nit)}

def evaluate(rows,model,theta,prod,split,shuffle_seed):
    out=[]
    kind={'M1_independent':'independent','M5_fixed':'fixed','M5_global':'global','M5_horizon':'horizon','M5_history':'history','M5_history_shuffled':'history'}[model]
    for r in rows:
        # Matched intervention: frozen history model, same parameters, shuffle only pairing.
        shuffled=model=='M5_history_shuffled'
        seed=int(hashlib.sha256(f"{shuffle_seed}:{r['scenario']}:{r['seed']}:{r['fold_id']}".encode()).hexdigest()[:16],16)
        q=predict(r,theta,kind,prod,shuffled,seed)
        s=prod.score(q,r['truth'])
        out.append(dict(split=split,scenario=r['scenario'],seed=r['seed'],fold_id=r['fold_id'],cutoff=r['cutoff'],target_generation=r['target_generation'],model=model,n_pairs=len(r['pairs']),**s))
    return out

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train-seeds',nargs='+',type=int,default=[11,23,67,101])
    p.add_argument('--validation-seeds',nargs='+',type=int,default=[42])
    p.add_argument('--penalties',nargs='+',type=float,default=[0.,0.001,0.01,0.1,1.])
    p.add_argument('--scenarios',nargs='*',default=[])
    p.add_argument('--shuffle-seed',type=int,default=431)
    args=p.parse_args()
    if set(args.train_seeds)&set(args.validation_seeds):p.error('overlapping seeds')
    if any(x<0 for x in args.penalties):p.error('negative penalty')
    if args.output.resolve()==args.input.resolve():p.error('output must differ from input')
    prod=load_source(args.source)
    scenario_set=set(args.scenarios) if args.scenarios else None
    train=prod.load_folds(args.input,args.train_seeds,scenario_set,'adjacent',0)
    val=prod.load_folds(args.input,args.validation_seeds,scenario_set,'adjacent',0)
    selected={};all_metrics=[];fitted={}
    for kind in ('global','horizon','history'):
        candidates=[]
        for penalty in args.penalties:
            theta,info=fit(train,kind,prod,penalty)
            rows=evaluate(val,'M5_'+kind,theta,prod,'validation',args.shuffle_seed)
            candidates.append((np.mean([r['joint_nll'] for r in rows]),penalty,theta,info))
        nll,penalty,theta,info=min(candidates,key=lambda x:(x[0],x[1]))
        fitted[kind]=theta
        selected['M5_'+kind]=dict(penalty=float(penalty),coefficients=theta.tolist(),validation_joint_nll=float(nll),optimizer=info)
    for model in MODELS:
        kind={'M5_global':'global','M5_horizon':'horizon','M5_history':'history','M5_history_shuffled':'history'}.get(model)
        theta=fitted[kind] if kind else np.array([])
        rows=evaluate(val,model,theta,prod,'validation',args.shuffle_seed)
        all_metrics.extend(rows)
        print(model,'validation joint NLL',round(float(np.mean([r['joint_nll'] for r in rows])),6),flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    write_csv(args.output/'metrics.csv',all_metrics)
    (args.output/'selected_models.json').write_text(json.dumps(selected,indent=2))
    protocol={'status':'EXPLORATORY_VALIDATION_ONLY','train_seeds':args.train_seeds,'validation_seeds':args.validation_seeds,'n_train_folds':len(train),'n_validation_folds':len(val),'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),'primary_metric':'joint_nll','pair_mode':'adjacent_disjoint','shuffled_semantics':'post-fit matched coefficients; shuffle cutoff clone pairing only; preserve marginals','limitations':['No locked test accessed','No simulator truth accessed','Overlapping folds are correlated','neutral and linkage_only can duplicate trajectories','Binary two-site forecasting only','Hyperparameters chosen using validation labels']}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print('Saved',args.output)
if __name__=='__main__':main()
