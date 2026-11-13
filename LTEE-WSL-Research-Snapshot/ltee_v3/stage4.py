#!/usr/bin/env python3
"""V3-C Stage 4: binary pairwise haplotype forecasting, observed-history only.

Independent trajectory split; no simulator truth is read. This is a two-site
joint distribution benchmark, NOT a mechanistic epistasis estimator.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np

EPS=1e-9
STATES=np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.int8)
MODELS=('M1_independent','M5_empirical','M5_shrunk','M5_dynamics','M5_shuffled')

def read_csv(p):
    with open(p,newline='') as f:return list(csv.DictReader(f))

def seq_matrix(records,L=None):
    if not records:raise ValueError('empty sample')
    seqs=[r['sequence'] for r in records]
    if L is None:L=len(seqs[0])
    if any(len(s)!=L or set(s)-set('01') for s in seqs):raise ValueError('invalid binary sequences')
    return np.array([[int(c) for c in s] for s in seqs],dtype=np.int8)

def pairs_for_length(L,mode='adjacent',max_pairs=0):
    if L<2:raise ValueError('need at least two sites')
    if mode=='adjacent':pairs=[(i,i+1) for i in range(0,L-1,2)]
    elif mode=='all_adjacent':pairs=[(i,i+1) for i in range(L-1)]
    else:raise ValueError(mode)
    return pairs[:max_pairs] if max_pairs else pairs

def frequencies(mat,pairs):
    out=[]
    for i,j in pairs:
        idx=2*mat[:,i]+mat[:,j]
        out.append(np.bincount(idx,minlength=4)/len(idx))
    return np.asarray(out,dtype=float)

def independent(q):
    a=q[:,2]+q[:,3];b=q[:,1]+q[:,3]
    return np.column_stack(((1-a)*(1-b),(1-a)*b,a*(1-b),a*b))

def normalize(q):
    q=np.maximum(q,EPS)
    return q/q.sum(axis=1,keepdims=True)

def predict(latest,previous,model,alpha=0.,trend=0.,horizon_ratio=1.,shuffle_seed=0,latest_matrix=None,pairs=None):
    base=independent(latest)
    if model=='M1_independent':return normalize(base)
    if model=='M5_empirical':return normalize(latest)
    if model=='M5_shrunk':return normalize((1-alpha)*latest+alpha*base)
    if model=='M5_dynamics':
        # Historical pair-frequency extrapolation, shrunk towards latest independence.
        delta=np.zeros_like(latest) if previous is None else latest-previous
        return normalize((1-alpha)*(latest+trend*horizon_ratio*delta)+alpha*base)
    if model=='M5_shuffled':
        # Destroy pairing within observed cutoff samples; retain each site's marginal.
        if latest_matrix is None or pairs is None:raise ValueError('shuffle requires historical cutoff sequences')
        rng=np.random.default_rng(shuffle_seed)
        q=[]
        for i,j in pairs:
            x=latest_matrix[:,i]
            y=latest_matrix[rng.permutation(len(latest_matrix)),j]
            q.append(np.bincount(2*x+y,minlength=4)/len(x))
        return normalize((1-alpha)*np.asarray(q)+alpha*base)
    raise ValueError(model)

def score(q,truth):
    q=normalize(q)
    joint_nll=float(-np.mean(np.sum(truth*np.log(q),axis=1)))
    m=independent(q);mt=independent(truth)
    # Marginal cross-entropy over both sites, averaged per site.
    p1=q[:,2]+q[:,3];p2=q[:,1]+q[:,3]
    t1=truth[:,2]+truth[:,3];t2=truth[:,1]+truth[:,3]
    def ce(t,p):
        p=np.clip(p,EPS,1-EPS)
        return -(t*np.log(p)+(1-t)*np.log1p(-p))
    marginal_nll=float(np.mean((ce(t1,p1)+ce(t2,p2))/2))
    pred_D=q[:,3]-(q[:,2]+q[:,3])*(q[:,1]+q[:,3])
    true_D=truth[:,3]-(truth[:,2]+truth[:,3])*(truth[:,1]+truth[:,3])
    return {'joint_nll':joint_nll,'marginal_nll':marginal_nll,
            'double_mutant_mae':float(np.mean(abs(q[:,3]-truth[:,3]))),
            'ld_D_mae':float(np.mean(abs(pred_D-true_D))),
            'joint_brier':float(np.mean(np.sum((q-truth)**2,axis=1)))}

def load_folds(root,seeds,scenarios,pair_mode,max_pairs):
    rows=[]
    for sd in sorted(root.iterdir()):
        if not sd.is_dir() or (scenarios and sd.name not in scenarios):continue
        for seed in seeds:
            d=sd/f'seed_{seed}'
            if not d.is_dir():raise FileNotFoundError(d)
            folds=read_csv(d/'folds.csv');history=read_csv(d/'historical_inputs.csv');targets=read_csv(d/'evaluation_targets_RESTRICTED.csv')
            hh={};tt={}
            for r in history:hh.setdefault(r['fold_id'],[]).append(r)
            for r in targets:tt.setdefault(r['fold_id'],[]).append(r)
            for fold in folds:
                fid=fold['fold_id'];cutoff=int(fold['cutoff']);target=int(fold['target_generation'])
                if target<=cutoff:raise ValueError('target not future')
                hist=hh.get(fid,[]);future=tt.get(fid,[])
                if not hist or not future:raise ValueError('missing fold rows')
                if any(int(r['generation'])>cutoff for r in hist):raise ValueError('future in history')
                if any(int(r['generation'])!=target for r in future):raise ValueError('incorrect future generation')
                by_time={}
                for r in hist:by_time.setdefault(int(r['generation']),[]).append(r)
                times=sorted(by_time)
                if times[-1]!=cutoff:raise ValueError('no cutoff observations')
                L=len(hist[0]['sequence']);pairs=pairs_for_length(L,pair_mode,max_pairs)
                last=seq_matrix(by_time[cutoff],L);future_mat=seq_matrix(future,L)
                prev=seq_matrix(by_time[times[-2]],L) if len(times)>1 else None
                rows.append(dict(scenario=sd.name,seed=seed,fold_id=fid,cutoff=cutoff,target_generation=target,
                    horizon=target-cutoff,history_interval=cutoff-times[-2] if prev is not None else None,
                    pairs=pairs,latest_matrix=last,latest=frequencies(last,pairs),
                    previous=frequencies(prev,pairs) if prev is not None else None,
                    truth=frequencies(future_mat,pairs),n_target_clones=len(future)))
    if not rows:raise ValueError('no folds loaded')
    return rows

def evaluate(rows,model,alpha,trend,split,shuffle_seed):
    out=[]
    for r in rows:
        ratio=r['horizon']/r['history_interval'] if r['history_interval'] else 0.
        # Stable deterministic shuffle across evaluations of a fold.
        s=shuffle_seed+sum(ord(c) for c in f"{r['scenario']}:{r['seed']}:{r['fold_id']}")
        q=predict(r['latest'],r['previous'],model,alpha,trend,ratio,s,r['latest_matrix'],r['pairs'])
        out.append(dict(split=split,scenario=r['scenario'],seed=r['seed'],fold_id=r['fold_id'],
            cutoff=r['cutoff'],target_generation=r['target_generation'],model=model,
            alpha=alpha,trend=trend,n_pairs=len(r['pairs']),n_target_clones=r['n_target_clones'],**score(q,r['truth'])))
    return out

def mean_nll(rows):
    # One fold = one unit; folds are correlated, so no naive SE claims.
    return float(np.mean([r['joint_nll'] for r in rows]))

def save_csv(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--train-seeds',type=int,nargs='+',default=[11,23]);ap.add_argument('--validation-seeds',type=int,nargs='+',default=[42]);ap.add_argument('--test-seeds',type=int,nargs='*',default=[])
    ap.add_argument('--scenarios',nargs='*',default=[]);ap.add_argument('--pair-mode',choices=['adjacent','all_adjacent'],default='adjacent');ap.add_argument('--max-pairs',type=int,default=0)
    ap.add_argument('--alphas',type=float,nargs='+',default=[0.,0.25,0.5,0.75,1.]);ap.add_argument('--trends',type=float,nargs='+',default=[0.,0.25,0.5])
    ap.add_argument('--shuffle-seed',type=int,default=413);ap.add_argument('--allow-exploratory-test',action='store_true')
    args=ap.parse_args()
    all_seeds=args.train_seeds+args.validation_seeds+args.test_seeds
    if len(all_seeds)!=len(set(all_seeds)):ap.error('trajectory seeds overlap')
    if args.test_seeds and not args.allow_exploratory_test:ap.error('test evaluation requires explicit --allow-exploratory-test')
    if args.max_pairs<0 or any(a<0 or a>1 for a in args.alphas) or any(t<0 for t in args.trends):ap.error('invalid hyperparameters')
    scenarios=set(args.scenarios) if args.scenarios else None
    train=load_folds(args.input,args.train_seeds,scenarios,args.pair_mode,args.max_pairs)
    val=load_folds(args.input,args.validation_seeds,scenarios,args.pair_mode,args.max_pairs)
    test=load_folds(args.input,args.test_seeds,scenarios,args.pair_mode,args.max_pairs) if args.test_seeds else []
    selected={};all_metrics=[]
    for model in MODELS:
        if model in ('M1_independent','M5_empirical'):grid=[(0.,0.)]
        elif model=='M5_dynamics':grid=[(a,t) for a in args.alphas for t in args.trends]
        else:grid=[(a,0.) for a in args.alphas]
        # Choose hyperparameters on validation; train folds are retained for future fitted models.
        candidates=[]
        for alpha,trend in grid:
            vr=evaluate(val,model,alpha,trend,'validation',args.shuffle_seed)
            candidates.append((mean_nll(vr),alpha,trend,vr))
        best=min(candidates,key=lambda x:(x[0],x[1],x[2]))
        nll,alpha,trend,vr=best
        selected[model]={'alpha':alpha,'trend':trend,'validation_joint_nll':nll,'n_train_folds_reserved':len(train),'n_validation_folds':len(val)}
        all_metrics.extend(vr)
        if test:all_metrics.extend(evaluate(test,model,alpha,trend,'test_exploratory',args.shuffle_seed+1000))
        print(model,'validation_joint_nll',round(nll,6),'alpha',alpha,'trend',trend,flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    save_csv(args.output/'metrics.csv',all_metrics)
    (args.output/'selected_models.json').write_text(json.dumps(selected,indent=2))
    protocol={'status':'EXPLORATORY','train_seeds':args.train_seeds,'validation_seeds':args.validation_seeds,'test_seeds':args.test_seeds,
        'n_train_folds':len(train),'n_validation_folds':len(val),'n_test_folds':len(test),'pair_mode':args.pair_mode,'max_pairs':args.max_pairs,
        'limitations':['binary two-site joint forecasts only','no learned epistasis coefficient','train trajectories reserved, not used to fit parameters in this version',
          'historical and target rows read separately for scoring','overlapping folds are correlated','neutral and linkage_only may be identical','pair selection fixed before scoring',
          'shuffled model destroys cutoff within-sample pairings; it is not a retrained gate permutation','test seeds previously inspected are exploratory']}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print('Saved',args.output,'validation folds',len(val),'test folds',len(test))
if __name__=='__main__':main()
