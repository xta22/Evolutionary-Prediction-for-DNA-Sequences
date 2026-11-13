#!/usr/bin/env python3
"""Stage 3.2: frozen-expert gate observability audit, validation only.

Uses v3c_stage3.py's real make_rows/design and pretrained expert coefficients.
Only gate parameters are fitted on training trajectories. No simulator truth is read.
This is a diagnostic conditional on pretrained experts, NOT a fair end-to-end M4 comparison.
"""
import argparse, importlib.util, json, csv, hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

EPS=1e-6

def load_module(path):
    spec=importlib.util.spec_from_file_location('production_stage3',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def feature_rows(rows, hscale, vscale):
    # One gate per fold; all inputs are observed at or before cutoff.
    feats=[]
    for r in rows:
        latest=np.asarray(r['latest']); recent=np.asarray(r['recent'])
        h=r['horizon']/hscale; v=r['volatility']/vscale
        feats.append([1.,v,h*v, float(np.mean(latest)),float(np.mean(np.abs(recent))),
                      float(np.mean(latest*(1-latest))),float(np.mean(recent)),
                      float(r['n_hist_times'])])
    return np.asarray(feats,float)

def normalize(train,val):
    # Intercept is untouched; all other columns scaled with train only.
    mu=train[:,1:].mean(axis=0); sd=train[:,1:].std(axis=0)
    sd=np.where(sd<1e-8,1.,sd)
    def f(x):return np.column_stack([np.ones(len(x)),(x[:,1:]-mu)/sd])
    return f(train),f(val),{'mean':mu.tolist(),'std':sd.tolist()}

def expert_arrays(prod,rows,theta,hscale,vscale):
    d=prod.design(rows,hscale,vscale)
    a,b=theta[:4],theta[4:8]
    z=logit(np.clip(d['base'],prod.EPS,1-prod.EPS)); X=d['X']
    pa=expit(np.clip(z+X@a,-30,30));pb=expit(np.clip(z+X@b,-30,30))
    return pa,pb,d['y'],d['n'],[r['L'] for r in rows]

def repeat_gate(g, lengths):return np.repeat(g,lengths)

def nll_from_gate(g,arrays):
    pa,pb,y,n,lengths=arrays
    p=np.clip((1-repeat_gate(g,lengths))*pa+repeat_gate(g,lengths)*pb,EPS,1-EPS)
    return float(-np.sum(n*(y*np.log(p)+(1-y)*np.log1p(-p)))/np.sum(n)),p

def fit_gate(F,arrays,columns,penalty):
    def objective(w):
        g=expit(np.clip(F[:,columns]@w,-30,30))
        nll,_=nll_from_gate(g,arrays)
        return nll+penalty*float(np.mean(w[1:]**2)) if len(w)>1 else nll
    init=np.zeros(len(columns));init[0]=-1.3
    opt=minimize(objective,init,method='L-BFGS-B',options={'maxiter':300,'ftol':1e-12})
    return opt.x,{'success':bool(opt.success),'iterations':int(opt.nit),'message':str(opt.message)}

def fold_scores(rows,arrays,gate):
    pa,pb,y,n,lengths=arrays
    out=[];pos=0
    for i,(r,L) in enumerate(zip(rows,lengths)):
        p=np.clip((1-gate[i])*pa[pos:pos+L]+gate[i]*pb[pos:pos+L],EPS,1-EPS)
        yy=y[pos:pos+L];pos+=L
        out.append({'scenario':r['scenario'],'seed':r['seed'],'fold_id':r['fold_id'],
                    'cutoff':r['cutoff'],'target_generation':r['target_generation'],
                    'gate':float(gate[i]),'nll':float(np.mean(-yy*np.log(p)-(1-yy)*np.log1p(-p)))})
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--original-results',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--train-seeds',type=int,nargs='+',default=[11,23])
    ap.add_argument('--validation-seeds',type=int,nargs='+',default=[42])
    ap.add_argument('--penalties',type=float,nargs='+',default=[0.001,0.01,0.1,1.0])
    ap.add_argument('--permutations',type=int,default=200)
    ap.add_argument('--seed',type=int,default=3201)
    args=ap.parse_args()
    if set(args.train_seeds)&set(args.validation_seeds):ap.error('overlapping train and validation seeds')
    if args.permutations<1:ap.error('permutations must be positive')
    prod=load_module(args.source)
    protocol=json.loads((args.original_results/'protocol.json').read_text())
    selected=json.loads((args.original_results/'selected_models.json').read_text())
    theta=np.asarray(selected['M4_latent']['coefficients'],float)
    if len(theta)!=11:raise ValueError('Expected 11 M4_latent coefficients')
    if list(args.train_seeds)!=protocol['train_seeds'] or list(args.validation_seeds)!=protocol['validation_seeds']:
        raise ValueError('Seed mismatch with original pretrained experts; abort')
    hscale=protocol['horizon_scale_train_only'];vscale=protocol['volatility_scale_train_only']
    scenarios=set(protocol['scenarios'])
    train=prod.make_rows(args.input,args.train_seeds,scenarios)
    val=prod.make_rows(args.input,args.validation_seeds,scenarios)
    if len(train)!=selected['M4_latent']['n_train_folds'] or len(val)!=selected['M4_latent']['n_validation_folds']:
        raise ValueError('Fold count mismatch; data may not match pretrained experts')
    ft=feature_rows(train,hscale,vscale);fv=feature_rows(val,hscale,vscale)
    ft,fv,norm=normalize(ft,fv)
    atr=expert_arrays(prod,train,theta,hscale,vscale)
    ava=expert_arrays(prod,val,theta,hscale,vscale)
    # Feature selection is predeclared, not chosen by validation results.
    configs={'constant':[0],'volatility':[0,1,2],'history_rich':list(range(8))}
    report={'status':'EXPLORATORY_VALIDATION_ONLY','protocol':'Frozen pretrained experts, refit gate on train only; no test labels or simulator truth',
            'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
            'train_folds':len(train),'validation_folds':len(val),'normalization_train_only':norm,'models':{}}
    results=[]; rng=np.random.default_rng(args.seed)
    for name,cols in configs.items():
        # Penalty selected by validation: exploratory, no confirmatory claims.
        candidates=[]
        for penalty in args.penalties:
            w,info=fit_gate(ft,atr,cols,penalty)
            g=expit(np.clip(fv[:,cols]@w,-30,30));vloss,_=nll_from_gate(g,ava)
            candidates.append((vloss,penalty,w,info))
        vloss,penalty,w,info=min(candidates,key=lambda t:(t[0],t[1]))
        g=expit(np.clip(fv[:,cols]@w,-30,30))
        perms=[]
        # Within scenario only, preserving outcomes and expert predictions.
        groups={s:np.asarray([i for i,r in enumerate(val) if r['scenario']==s]) for s in sorted({r['scenario'] for r in val})}
        for _ in range(args.permutations):
            gp=g.copy()
            for ix in groups.values():gp[ix]=g[rng.permutation(ix)]
            perm_nll,_=nll_from_gate(gp,ava)
            perms.append(perm_nll-vloss)
        record={'validation_nll':vloss,'penalty':penalty,'coefficients':w.tolist(),'optimizer':info,
                'gate_mean':float(g.mean()),'gate_std':float(g.std()),
                'permutation_delta_nll_mean':float(np.mean(perms)),
                'permutation_delta_nll_q025_q975':np.quantile(perms,[.025,.975]).tolist()}
        report['models'][name]=record
        for r in fold_scores(val,ava,g):results.append({'model':name,**r})
        print(name,'validation_nll',round(vloss,7),'gate_std',round(float(g.std()),7),
              'postfit_perm_delta',round(float(np.mean(perms)),8),'penalty',penalty,flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'stage32_report.json').write_text(json.dumps(report,indent=2))
    with (args.output/'validation_folds.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    print('Saved',args.output)

if __name__=='__main__':main()
