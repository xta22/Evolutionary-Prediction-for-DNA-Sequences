#!/usr/bin/env python3
"""V3-C stage 1: leakage-controlled sampled-population forecasting baselines.

Uses V3-B independent forward trajectories, not paired ancestor/descendant clones.
Predicts allele probabilities for future sampled haplotypes. NOT a CTMC, M4, or M5.
Each fold is fitted using only that trajectory's observations <= cutoff.
Validation/test seeds are disjoint. No hidden simulator truth is read.
"""
import argparse, csv, json, math
from pathlib import Path
import numpy as np

SCENARIOS=('neutral','heterogeneous','context','regime_shift','linkage_only','recombination','selection_only','epistasis')
MODELS=('B0_global','B1_latest','B2_shrunk_trend','B3_site_trend')

def read_csv(path):
    with open(path,newline='') as f: return list(csv.DictReader(f))

def save_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: raise ValueError('empty results')
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def parse_history(rows, cutoff):
    selected=[r for r in rows if int(r['generation'])<=cutoff]
    if not selected: raise ValueError('no historical observations')
    times=sorted({int(r['generation']) for r in selected})
    arrays={}
    for t in times:
        ss=[r['sequence'] for r in selected if int(r['generation'])==t]
        if len(set(map(len,ss)))!=1: raise ValueError('inconsistent sequence lengths')
        if any(set(s)-set('01') for s in ss): raise ValueError('nonbinary sequence')
        arrays[t]=np.array([[int(c) for c in s] for s in ss],dtype=float)
    return arrays

def predict(arrays,cutoff,target,model,alpha,trend_strength):
    times=sorted(arrays)
    if times[-1]>cutoff: raise ValueError('future information in history')
    if target<=cutoff: raise ValueError('nonfuture target')
    k=arrays[times[-1]].shape[1]
    pooled=np.concatenate([arrays[t] for t in times],axis=0)
    global_rate=(pooled.sum()+alpha)/(pooled.size+2*alpha)
    if model=='B0_global': return np.full(k,global_rate)
    last=arrays[times[-1]]
    p=(last.sum(axis=0)+alpha*global_rate)/(len(last)+alpha)
    if model=='B1_latest': return np.clip(p,1e-5,1-1e-5)
    if len(times)<2: return np.clip(p,1e-5,1-1e-5)
    prev=arrays[times[-2]]
    q=(prev.sum(axis=0)+alpha*global_rate)/(len(prev)+alpha)
    dt=times[-1]-times[-2]
    horizon=(target-cutoff)/dt
    if model=='B2_shrunk_trend':
        delta=np.mean(p-q)
        out=p+trend_strength*horizon*delta
    elif model=='B3_site_trend':
        out=p+trend_strength*horizon*(p-q)
    else: raise ValueError(model)
    return np.clip(out,1e-5,1-1e-5)

def score(p,y):
    p=np.asarray(p);y=np.asarray(y)
    if y.ndim!=2 or y.shape[1]!=len(p):raise ValueError('target shape mismatch')
    nll=-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))
    brier=np.mean((y-p[None,:])**2)
    observed=float(y.sum()); predicted=float(len(y)*p.sum())
    # A marginal Bernoulli model cannot estimate haplotype joint likelihood.
    return dict(nll=float(nll),brier=float(brier),observed_allele1_count=observed,
                predicted_allele1_count=predicted,abs_count_error=abs(observed-predicted),
                n_target_genomes=len(y),n_sites=len(p))

def get_run(root,scenario,seed):
    d=root/scenario/f'seed_{seed}'
    if not d.exists():raise FileNotFoundError(d)
    return d

def evaluate_run(d,model,alpha,strength,min_cutoff):
    # IMPORTANT: Do not read observed_genomes_RESTRICTED.csv, simulation truth,
    # or full-trajectory statistics. Only fold-specific historical/target files.
    folds=read_csv(d/'folds.csv')
    history=read_csv(d/'historical_inputs.csv')
    targets=read_csv(d/'evaluation_targets_RESTRICTED.csv')
    results=[]
    for f in folds:
        cutoff=int(f['cutoff']);target=int(f['target_generation']);fid=f['fold_id']
        if cutoff<min_cutoff:continue
        h=[r for r in history if r['fold_id']==fid]
        y=[r for r in targets if r['fold_id']==fid]
        if not h or not y:raise ValueError('empty fold')
        if any(int(r['generation'])>cutoff for r in h):raise ValueError('HISTORY LEAKAGE')
        if any(int(r['generation'])!=target for r in y):raise ValueError('TARGET TIME LEAKAGE')
        arr=parse_history(h,cutoff)
        probs=predict(arr,cutoff,target,model,alpha,strength)
        yy=np.array([[int(c) for c in r['sequence']] for r in y],dtype=float)
        results.append(dict(fold_id=fid,cutoff=cutoff,target_generation=target,
                            **score(probs,yy)))
    return results

def select_hyperparams(root,scenarios,seeds,alpha_grid,strength_grid,min_cutoff):
    # Validation trajectories only; test seeds never touched here.
    scores={}
    for model in MODELS:
        candidates=[(a,s) for a in alpha_grid for s in (strength_grid if model.startswith('B2') or model.startswith('B3') else [0.0])]
        for a,s in candidates:
            vals=[]
            for scenario in scenarios:
                for seed in seeds:
                    vals.extend(r['nll'] for r in evaluate_run(get_run(root,scenario,seed),model,a,s,min_cutoff))
            if vals:scores[(model,a,s)]=float(np.mean(vals))
    selected={}
    for model in MODELS:
        opts=[(loss,a,s) for (m,a,s),loss in scores.items() if m==model]
        if not opts:raise ValueError('no validation folds')
        loss,a,s=min(opts)
        selected[model]={'alpha':a,'trend_strength':s,'validation_nll':loss}
    return selected

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default='../LTEE_prospective/v3b_simulation')
    ap.add_argument('--output',default='../LTEE_prospective/v3c_stage1')
    ap.add_argument('--train-seeds',nargs='+',type=int,default=[11,23])
    ap.add_argument('--validation-seeds',nargs='+',type=int,default=[42])
    ap.add_argument('--test-seeds',nargs='+',type=int,default=[67,101])
    ap.add_argument('--scenarios',nargs='+',choices=SCENARIOS,default=list(SCENARIOS))
    ap.add_argument('--min-cutoff',type=int,default=10)
    args=ap.parse_args()
    groups=[set(args.train_seeds),set(args.validation_seeds),set(args.test_seeds)]
    if any(groups[i]&groups[j] for i in range(3) for j in range(i+1,3)):ap.error('seed splits must be disjoint')
    if not all(groups):ap.error('each split needs at least one seed')
    root=Path(args.input);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    # No cross-trajectory learning in this first-stage baseline; train seeds reserved
    # for later learned models. Hyperparameters selected ONLY on validation seeds.
    selected=select_hyperparams(root,args.scenarios,args.validation_seeds,[1.0,5.0,20.0],[0.0,0.25,0.5,1.0],args.min_cutoff)
    (out/'selected_hyperparameters.json').write_text(json.dumps(selected,indent=2))
    rows=[]
    for split,seeds in [('validation',args.validation_seeds),('test',args.test_seeds)]:
        for scenario in args.scenarios:
            for seed in seeds:
                for model in MODELS:
                    conf=selected[model]
                    for r in evaluate_run(get_run(root,scenario,seed),model,conf['alpha'],conf['trend_strength'],args.min_cutoff):
                        rows.append(dict(split=split,scenario=scenario,seed=seed,model=model,**r))
    save_csv(out/'metrics.csv',rows)
    report={'status':'EXPLORATORY_BASELINES_ONLY','input':str(root),'splits':{k:getattr(args,k.replace('-','_')) for k in ['train_seeds','validation_seeds','test_seeds']},
            'scenarios':args.scenarios,'rows':len(rows),'models':list(MODELS),
            'limitations':['Binary allele-frequency prediction, not four-state CTMC','No paired individual ancestry assumed',
                           'No cross-trajectory model training yet; train seeds reserved','No M4/M5 implemented',
                           'Joint haplotype likelihood and linkage/epistasis forecasting NOT assessed',
                           'Same simulator mechanism parameters may be shared across independent seeds']}
    (out/'audit_report.json').write_text(json.dumps(report,indent=2))
    print('V3C STAGE1',report['status'],'rows',len(rows))
    for model in MODELS:
        a=[r['nll'] for r in rows if r['split']=='test' and r['model']==model]
        print(model,'test NLL',round(float(np.mean(a)),6) if a else 'NA')
    print('Saved',out)

if __name__=='__main__':main()
