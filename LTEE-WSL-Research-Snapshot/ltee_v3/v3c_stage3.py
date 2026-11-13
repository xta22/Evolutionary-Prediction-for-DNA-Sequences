#!/usr/bin/env python3
"""V3-C Stage 3: observed-history-only latent-regime forecasting.

Binary sampled-population allele forecasting. No simulator truth is loaded.
M4 is a two-expert mixture, not a certified biological mutator-state estimator.
"""
import argparse, csv, json, hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

EPS=1e-5
MODELS=('B1_latest','M1_shared','M4_static','M4_latent','M4_shuffled')

def read_csv(path):
    with open(path,newline='') as f: return list(csv.DictReader(f))

def valid_binary(seq, length=None):
    if not seq or any(c not in '01' for c in seq): raise ValueError('invalid binary sequence')
    if length is not None and len(seq)!=length: raise ValueError('inconsistent sequence lengths')
    return np.fromiter((int(c) for c in seq),dtype=np.float64)

def make_rows(root, seeds, scenarios=None):
    rows=[]
    for scenario_dir in sorted(root.iterdir()):
        if not scenario_dir.is_dir() or (scenarios and scenario_dir.name not in scenarios):continue
        for seed in seeds:
            folder=scenario_dir/f'seed_{seed}'
            if not folder.is_dir(): raise FileNotFoundError(folder)
            folds=read_csv(folder/'folds.csv')
            history=read_csv(folder/'historical_inputs.csv')
            targets=read_csv(folder/'evaluation_targets_RESTRICTED.csv')
            hh={};tt={}
            for r in history: hh.setdefault(r['fold_id'],[]).append(r)
            for r in targets: tt.setdefault(r['fold_id'],[]).append(r)
            for fold in folds:
                fid=fold['fold_id'];cutoff=int(fold['cutoff']);target=int(fold['target_generation'])
                if target<=cutoff:raise ValueError('nonfuture target')
                hist=hh.get(fid,[]); future=tt.get(fid,[])
                if not hist or not future:raise ValueError('missing fold data '+str(folder/fid))
                if any(int(r['generation'])>cutoff for r in hist):raise ValueError('future in history')
                if any(int(r['generation'])!=target for r in future):raise ValueError('target generation mismatch')
                L=len(hist[0]['sequence']); by_time={}
                for r in hist:by_time.setdefault(int(r['generation']),[]).append(valid_binary(r['sequence'],L))
                times=sorted(by_time)
                if times[-1]!=cutoff:raise ValueError('no cutoff sample')
                latest=np.mean(by_time[cutoff],axis=0)
                prev=times[-2] if len(times)>1 else None
                if prev is None:recent=np.zeros(L);volatility=0.
                else:
                    old=np.mean(by_time[prev],axis=0)
                    recent=latest-old
                    volatility=float(np.mean(np.abs(recent)))
                # Regime proxy uses only observed historical changes, NOT simulator event counts.
                # Across-site average is deliberately low-dimensional to reduce site overfitting.
                horizon=target-cutoff
                y=np.mean([valid_binary(r['sequence'],L) for r in future],axis=0)
                n=len(future)
                rows.append({'scenario':scenario_dir.name,'seed':seed,'fold_id':fid,
                    'cutoff':cutoff,'target_generation':target,'horizon':horizon,
                    'latest':latest,'recent':recent,'volatility':volatility,
                    'y':y,'n':n,'n_cutoff':len(by_time[cutoff]),'n_hist_times':len(times),'L':L})
    if not rows:raise ValueError('no folds')
    return rows

def design(rows, horizon_scale, volatility_scale):
    latest=np.concatenate([r['latest'] for r in rows]); recent=np.concatenate([r['recent'] for r in rows]);
    y=np.concatenate([r['y'] for r in rows]);n=np.concatenate([np.full(r['L'],r['n']) for r in rows]);
    h=np.concatenate([np.full(r['L'],r['horizon']/horizon_scale) for r in rows]);
    v=np.concatenate([np.full(r['L'],r['volatility']/volatility_scale) for r in rows]);
    cutoff_n=np.concatenate([np.full(r['L'],r['n_cutoff']) for r in rows])
    base=np.clip((latest*cutoff_n+0.5)/(cutoff_n+1),EPS,1-EPS) # smooth using historical sample count only
    # shared expert features, with time-varying observed changes
    X=np.column_stack([np.ones(len(y)),h,recent,h*recent]);G=np.column_stack([np.ones(len(y)),v,h*v]);
    return {'X':X,'G':G,'base':base,'y':y,'n':n,'h':h,'v':v}

def predictions(theta,data,model):
    X,G,base=data['X'],data['G'],data['base']
    if model=='B1_latest':return base
    z=logit(base)
    if model=='M1_shared':return expit(np.clip(z+X[:,:2]@theta,-30,30))
    if model=='M4_static':
        a,b=theta[:4],theta[4:8]
        return np.clip(.5*expit(np.clip(z+X@a,-30,30))+.5*expit(np.clip(z+X@b,-30,30)),EPS,1-EPS)
    if model in ('M4_latent','M4_shuffled'):
        a,b,g=theta[:4],theta[4:8],theta[8:11]
        gate=expit(np.clip(G@g,-30,30))
        return np.clip((1-gate)*expit(np.clip(z+X@a,-30,30))+gate*expit(np.clip(z+X@b,-30,30)),EPS,1-EPS)
    raise ValueError(model)

def loss(theta,data,model,penalty):
    p=np.clip(predictions(theta,data,model),EPS,1-EPS)
    n,y=data['n'],data['y']
    # weighted by number of target genomes, not number of overlapping folds
    ce=-np.sum(n*(y*np.log(p)+(1-y)*np.log1p(-p)))/np.sum(n)
    return float(ce+(penalty*np.mean(theta**2) if len(theta) else 0.0))

def fit(data,model,penalty,seed=123):
    if model=='B1_latest':return np.zeros(0),{'success':True,'iterations':0,'message':'baseline'}
    dim={'M1_shared':2,'M4_static':8,'M4_latent':11,'M4_shuffled':11}[model]
    init=np.zeros(dim)
    if dim>=8:
        init[:4]=np.array([0.02,0,0.01,0]);init[4:8]=np.array([-0.02,0,-0.01,0])
    if dim==11:init[9]=0.1
    opt=minimize(loss,init,args=(data,model,penalty),method='L-BFGS-B',
        options={'maxiter':250,'ftol':1e-11,'gtol':1e-7})
    return opt.x,{'success':bool(opt.success),'iterations':int(opt.nit),'message':str(opt.message)}

def score_rows(rows,theta,model,hscale,vscale,split):
    out=[]
    for r in rows:
        data=design([r],hscale,vscale)
        p=np.clip(predictions(theta,data,model),EPS,1-EPS)
        y=r['y'];nll=float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p)))
        brier=float(np.mean(y*(1-p)**2+(1-y)*p**2))
        out.append({'split':split,'scenario':r['scenario'],'seed':r['seed'],'fold_id':r['fold_id'],
            'cutoff':r['cutoff'],'target_generation':r['target_generation'],'model':model,
            'nll':nll,'brier':brier,'abs_derived_count_error':float(abs(p.sum()-y.sum())),
            'n_sites':r['L'],'n_target_clones':r['n']})
    return out

def shuffled(rows, seed):
    # Shuffle gate features across folds without altering observed outcomes, latest states, or horizons.
    # This is a negative control, not a valid alternative generative model.
    rng=np.random.default_rng(seed);order=rng.permutation(len(rows))
    out=[]
    for i,r in enumerate(rows):out.append({**r,'volatility':rows[order[i]]['volatility']})
    return out

def save_csv(path,rows):
    with open(path,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train-seeds',type=int,nargs='+',default=[11,23]);p.add_argument('--validation-seeds',type=int,nargs='+',default=[42]);
    p.add_argument('--test-seeds',type=int,nargs='*',default=[])
    p.add_argument('--scenarios',nargs='*',default=[])
    p.add_argument('--penalties',type=float,nargs='+',default=[0.001,0.01,0.1,1.0]);p.add_argument('--shuffle-seed',type=int,default=412)
    args=p.parse_args()
    allseeds=args.train_seeds+args.validation_seeds+args.test_seeds
    if len(allseeds)!=len(set(allseeds)):p.error('overlapping trajectory seeds')
    if any(x<0 for x in args.penalties):p.error('negative penalty')
    scenarios=set(args.scenarios) if args.scenarios else None
    train=make_rows(args.input,args.train_seeds,scenarios)
    val=make_rows(args.input,args.validation_seeds,scenarios)
    test=make_rows(args.input,args.test_seeds,scenarios) if args.test_seeds else []
    # Do not pool identical neutral/linkage-only trajectories as independent evidence.
    hscale=float(np.median([r['horizon'] for r in train]));vscale=max(0.01,float(np.std([r['volatility'] for r in train])))
    dt=design(train,hscale,vscale);dv=design(val,hscale,vscale)
    # Gate negative control uses train-derived permutation only, fixed before validation.
    train_shuf=shuffled(train,args.shuffle_seed);val_shuf=shuffled(val,args.shuffle_seed+1)
    dts=design(train_shuf,hscale,vscale);dvs=design(val_shuf,hscale,vscale)
    selected={};results=[]
    for model in MODELS:
        candidates=[]
        for penalty in ([0.] if model=='B1_latest' else args.penalties):
            tr=dts if model=='M4_shuffled' else dt; va=dvs if model=='M4_shuffled' else dv
            theta,info=fit(tr,model,penalty)
            vloss=loss(theta,va,model,0.)
            candidates.append((vloss,penalty,theta,info))
        best=min(candidates,key=lambda x:(x[0],x[1]))
        _,penalty,theta,info=best
        selected[model]={'penalty':penalty,'validation_nll':best[0],
            'coefficients':theta.tolist(),'optimizer':info,'n_train_folds':len(train),'n_validation_folds':len(val)}
        results+=score_rows(val_shuf if model=='M4_shuffled' else val,theta,model,hscale,vscale,'validation')
        if test:
            trows=shuffled(test,args.shuffle_seed+2) if model=='M4_shuffled' else test
            results+=score_rows(trows,theta,model,hscale,vscale,'test_exploratory')
        print(model,'validation_nll',round(best[0],7),'penalty',penalty,'success',info['success'],flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    save_csv(args.output/'metrics.csv',results)
    (args.output/'selected_models.json').write_text(json.dumps(selected,indent=2))
    protocol={'train_seeds':args.train_seeds,'validation_seeds':args.validation_seeds,'test_seeds':args.test_seeds,
        'scenarios':sorted({r['scenario'] for r in train}),'horizon_scale_train_only':hscale,
        'volatility_scale_train_only':vscale,'gate_semantics':'historical frequency volatility proxy; NOT identified biological mutator state',
        'target_semantics':'binary site-level allele probabilities for future sampled genomes',
        'limitations':['not four-state DNA','not joint haplotype model','simulation parameters may not match LTEE',
            'overlapping folds correlated','neutral and linkage_only may be identical','test seeds previously inspected are exploratory',
            'unobserved future regime shift cannot be predicted without precursor signal'],
        'restricted_files':'Only evaluation_targets_RESTRICTED.csv is read for scoring; no simulation_truth, mutation_events or population_statistics read'}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print('Saved',args.output,'validation folds',len(val),'test folds',len(test))
if __name__=='__main__':main()
