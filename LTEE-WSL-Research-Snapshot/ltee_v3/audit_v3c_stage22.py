#!/usr/bin/env python3
"""Read-only production V3-C Stage 2 instrumentation, validation only.
Loads the actual v3c_stage2.py, its fitted coefficients, and historical inputs.
Never reads test seeds or simulation_truth_RESTRICTED.
"""
import argparse, csv, importlib.util, json
from pathlib import Path
import numpy as np

BLOCKS = {'M1_shared': {'shared': [0,1]},
          'M2_history': {'shared':[0,1], 'history':[2,3,4]},
          'M3_local_context': {'shared':[0,1], 'history':[2,3,4], 'local_context':[5,6,7]}}

def load_source(path):
    spec=importlib.util.spec_from_file_location('v3c_production_stage2',str(path.resolve()))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required=('collect','prepare','predict','metrics','fit_model','logit','sigmoid')
    missing=[x for x in required if not hasattr(module,x)]
    if missing: raise RuntimeError('Production interface mismatch: '+str(missing))
    return module

def objective_and_gradient(prod, data, model, w, penalty):
    X,offset,s,n=prod.prepare(data,model)
    z=offset+X@w
    p=prod.sigmoid(z)
    loss=float(np.sum(n*np.logaddexp(0,z)-s*z)/n.sum()+0.5*penalty*np.dot(w,w))
    grad=(X.T@(n*p-s))/n.sum()+penalty*w
    return loss,grad

def score(prod,data,model,coef,perturb=None,seed=0):
    rng=np.random.default_rng(seed)
    scores=[]; changes=[]
    for d in data:
        p=prod.predict(d,model,{'coef':coef})
        if perturb is not None:
            X=d['features'][model].copy()
            # Permute columns jointly across sites, preserving their within-block relationships.
            # This is a conditional, fixed-model sensitivity diagnostic, NOT causal importance.
            idx=BLOCKS[model][perturb]
            order=rng.permutation(len(X))
            X[:,idx]=X[order][:,idx]
            pnew=np.clip(prod.sigmoid(prod.logit(d['p0'])+X@coef),prod.EPS,1-prod.EPS)
            changes.append(float(np.mean(np.abs(pnew-p))))
            p=pnew
        scores.append(float(prod.metrics(p,d['y'])['nll']))
    return float(np.mean(scores)),float(np.mean(changes)) if changes else 0.

def write_csv(path,rows):
    if not rows:return
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run(args):
    prod=load_source(Path(args.source))
    selected=json.loads(Path(args.models).read_text())
    protocol=json.loads(Path(args.protocol).read_text())
    val_seeds=protocol['validation_seeds'];train_seeds=protocol['train_seeds'];test_seeds=protocol['test_seeds']
    assert not(set(val_seeds)&set(test_seeds)) and not(set(train_seeds)&set(test_seeds))
    scenarios=protocol['scenarios']
    root=Path(args.input);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    # Validation only; collect internally reads evaluator labels for validation.
    data=prod.collect(root,scenarios,val_seeds,args.min_cutoff)
    rows=[];perms=[];opts=[];preds=[]
    for model in ('M1_shared','M2_history','M3_local_context'):
        info=selected[model];coef=np.asarray(info['coefficients'],dtype=float)
        penalty=float(info['penalty']);X,offset,s,n=prod.prepare(data,model)
        if len(coef)!=X.shape[1]:raise ValueError('Coefficient dimension mismatch')
        l0,g0=objective_and_gradient(prod,data,model,np.zeros_like(coef),penalty)
        lf,gf=objective_and_gradient(prod,data,model,coef,penalty)
        opts.append(dict(model=model,penalty=penalty,initial_objective=l0,final_objective=lf,
                         objective_improvement=l0-lf,initial_gradient_norm=float(np.linalg.norm(g0)),
                         final_gradient_norm=float(np.linalg.norm(gf)),gradient_max_abs=float(np.max(np.abs(gf))),
                         fitted_coefficients=coef.tolist(),n_training_samples_in_audit='validation_only_gradient'))
        for j in range(X.shape[1]):
            z=X[:,j]*coef[j]
            rows.append(dict(model=model,feature_index=j,block=next((b for b,ids in BLOCKS[model].items() if j in ids),'unknown'),
                             coefficient=coef[j],feature_mean=float(X[:,j].mean()),feature_std=float(X[:,j].std()),
                             feature_min=float(X[:,j].min()),feature_max=float(X[:,j].max()),
                             contribution_std=float(z.std()),contribution_mean_abs=float(np.mean(np.abs(z))),
                             feature_near_constant=bool(X[:,j].std()<1e-10)))
        base,_=score(prod,data,model,coef)
        for block in BLOCKS[model]:
            if block=='shared':continue
            for rep in range(args.permutations):
                changed,mean_dp=score(prod,data,model,coef,perturb=block,seed=202600+rep)
                perms.append(dict(model=model,block=block,replicate=rep,validation_nll_original=base,
                                  validation_nll_permuted=changed,delta_nll=changed-base,
                                  mean_abs_probability_change=mean_dp))
        for d in data:
            p=prod.predict(d,model,{'coef':coef});p0=d['p0']
            preds.append(dict(model=model,scenario=d['scenario'],seed=d['seed'],fold_id=d['fold_id'],
                              cutoff=d['cutoff'],target_generation=d['target_generation'],
                              mean_abs_delta_vs_B1=float(np.mean(np.abs(p-p0))),
                              max_abs_delta_vs_B1=float(np.max(np.abs(p-p0))),
                              fraction_sites_changed=float(np.mean(np.abs(p-p0)>1e-8)),
                              nll=float(prod.metrics(p,d['y'])['nll'])))
    # Pairwise M3 vs M2 predictions on identical folds.
    for d in data:
        ps={m:prod.predict(d,m,{'coef':np.asarray(selected[m]['coefficients'])}) for m in ('M1_shared','M2_history','M3_local_context')}
        for a,b in [('M2_history','M1_shared'),('M3_local_context','M2_history')]:
            diff=np.abs(ps[a]-ps[b])
            preds.append(dict(model=a+'_vs_'+b,scenario=d['scenario'],seed=d['seed'],fold_id=d['fold_id'],
                              cutoff=d['cutoff'],target_generation=d['target_generation'],
                              mean_abs_delta_vs_B1=float(diff.mean()),max_abs_delta_vs_B1=float(diff.max()),
                              fraction_sites_changed=float(np.mean(diff>1e-8)),nll=''))
    write_csv(out/'feature_diagnostics.csv',rows)
    write_csv(out/'prediction_diagnostics.csv',preds)
    write_csv(out/'permutation_validation.csv',perms)
    (out/'optimization_diagnostics.json').write_text(json.dumps({'status':'VALIDATION_ONLY_POST_HOC',
        'warning':'Gradients computed on validation data at training-fitted coefficients; NOT training gradients or optimizer trace.',
        'gradient_definition':'penalized validation binomial objective; loss normalized by validation target count',
        'permutation':'site-row permutation within each validation fold, fixed coefficients, no retraining',
        'limitation':'Neighboring allele-frequency features are not simulator hidden context; no mechanism recovery established.',
        'test_seeds_not_loaded':test_seeds,'models':opts},indent=2))
    print('SAVED',out)
    for r in opts:print(r['model'],'validation objective improvement',round(r['objective_improvement'],8),
                        'validation final grad norm',round(r['final_gradient_norm'],8))
    for model in ('M2_history','M3_local_context'):
        pp=[r['delta_nll'] for r in perms if r['model']==model]
        print(model,'mean permutation delta NLL',float(np.mean(pp)) if pp else 'N/A')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='v3c_stage2.py');p.add_argument('--input',required=True)
    p.add_argument('--models',required=True);p.add_argument('--protocol',required=True)
    p.add_argument('--output',required=True);p.add_argument('--min-cutoff',type=int,default=10)
    p.add_argument('--permutations',type=int,default=20)
    args=p.parse_args()
    if args.permutations<1:p.error('permutations must be positive')
    run(args)
if __name__=='__main__':main()
