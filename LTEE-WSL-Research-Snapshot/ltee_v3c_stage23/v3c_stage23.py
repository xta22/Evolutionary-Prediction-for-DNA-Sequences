#!/usr/bin/env python3
"""Stage 2.3: observable information vs capacity, validation-only.
Imports the actual Stage 2 production feature builder. Never reads simulation_truth.
Synthetic oracle is separately generated and NEVER mixed with V3-B data.
"""
import argparse, csv, importlib.util, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

EPS=1e-6

def production(path):
    spec=importlib.util.spec_from_file_location('v3c_stage2_production',str(Path(path).resolve()))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    for name in ('collect','make_features','logit','sigmoid'):
        if not hasattr(mod,name): raise RuntimeError('Production missing '+name)
    return mod

def expand(d, kind):
    X=np.asarray(d['features']['M3_local_context'],float)
    if kind=='shared': return X[:,:2]
    if kind=='history': return X[:,:5]
    if kind=='local': return X
    if kind=='nonlinear':
        # Fixed feature expansion, shared coefficients, not site identity lookup.
        z=X[:,2:]
        extra=np.column_stack([z*z, z[:,0]*z[:,3], z[:,2]*z[:,3], z[:,3]*z[:,4]])
        return np.column_stack([X,extra])
    raise ValueError(kind)

def matrices(data,kind):
    xs=[];offsets=[];ys=[];ns=[]
    for d in data:
        x=expand(d,kind)
        xs.append(x)
        offsets.append(np.asarray(d['p0'],float))
        ys.append(np.asarray(d['y'],float).sum(axis=0))
        ns.append(np.full(x.shape[0],len(d['y']),float))
    return np.concatenate(xs),np.concatenate(offsets),np.concatenate(ys),np.concatenate(ns)

def fit(data,kind,penalty,maxiter=300):
    X,p0,k,n=matrices(data,kind)
    base=np.log(np.clip(p0,EPS,1-EPS)/(1-np.clip(p0,EPS,1-EPS)))
    def obj(w):
        z=np.clip(base+X@w,-35,35)
        p=1/(1+np.exp(-z))
        loss=np.sum(n*np.logaddexp(0,z)-k*z)/n.sum()+penalty*np.dot(w,w)/2
        grad=X.T@(n*p-k)/n.sum()+penalty*w
        return float(loss),grad
    init=np.zeros(X.shape[1]);initial,g0=obj(init)
    res=minimize(obj,init,jac=True,method='L-BFGS-B',options={'maxiter':maxiter,'ftol':1e-12,'gtol':1e-8})
    end,g1=obj(res.x)
    return {'coef':res.x,'success':bool(res.success),'message':str(res.message),'iterations':int(res.nit),
            'train_initial_objective':initial,'train_final_objective':end,
            'train_initial_gradient_norm':float(np.linalg.norm(g0)),
            'train_final_gradient_norm':float(np.linalg.norm(g1))}

def predict(d,kind,coef):
    p=np.clip(np.asarray(d['p0']),EPS,1-EPS)
    z=np.log(p/(1-p))+expand(d,kind)@coef
    return np.clip(1/(1+np.exp(-np.clip(z,-35,35))),EPS,1-EPS)

def score(data,kind,coef):
    losses=[];weights=[]
    for d in data:
        p=predict(d,kind,coef)
        y=np.asarray(d['y'])
        losses.append(float(-np.sum(y*np.log(p)+(1-y)*np.log1p(-p))))
        weights.append(y.size)
    return sum(losses)/sum(weights)

def writecsv(path,rows):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def real_audit(args):
    prod=production(args.source)
    if set(args.train_seeds)&set(args.validation_seeds): raise ValueError('Train/validation seeds overlap')
    root=Path(args.input)
    train=prod.collect(root,args.scenarios,args.train_seeds,args.min_cutoff)
    val=prod.collect(root,args.scenarios,args.validation_seeds,args.min_cutoff)
    results=[];chosen={}
    for kind in ('shared','history','local','nonlinear'):
        candidates=[]
        for penalty in args.penalties:
            model=fit(train,kind,penalty,args.maxiter)
            v=score(val,kind,model['coef'])
            candidates.append((v,penalty,model))
        v,penalty,m=min(candidates,key=lambda a:(a[0],a[1]))
        chosen[kind]=(penalty,m)
        results.append({'model':kind,'validation_nll':v,'penalty':penalty,
                        'train_initial_objective':m['train_initial_objective'],
                        'train_final_objective':m['train_final_objective'],
                        'train_initial_gradient_norm':m['train_initial_gradient_norm'],
                        'train_final_gradient_norm':m['train_final_gradient_norm'],
                        'fit_success':m['success'],'iterations':m['iterations']})
    # Diagnostics: feature signal and marginal prediction differences; no test labels.
    feature=[]
    for kind in ('history','local','nonlinear'):
        X=np.concatenate([expand(d,kind) for d in train])
        for j in range(X.shape[1]):
            feature.append({'model':kind,'feature_index':j,'train_std':float(np.std(X[:,j])),
                            'train_min':float(np.min(X[:,j])),'train_max':float(np.max(X[:,j])),
                            'coefficient':float(chosen[kind][1]['coef'][j]),
                            'contribution_std':float(np.std(X[:,j]*chosen[kind][1]['coef'][j]))})
    changes=[]
    base=chosen['shared'][1]['coef']
    for kind in ('history','local','nonlinear'):
        coef=chosen[kind][1]['coef']
        for d in val:
            a=predict(d,'shared',base);b=predict(d,kind,coef)
            changes.append({'scenario':d['scenario'],'seed':d['seed'],'fold_id':d['fold_id'],
                            'model':kind,'mean_abs_probability_delta':float(np.mean(abs(a-b))),
                            'max_abs_probability_delta':float(np.max(abs(a-b)))})
    # Synthetic informative feature fixture, separate from actual trajectory data.
    fixture=synthetic_fixture(args.fixture_seed)
    return results,feature,changes,fixture

def synthetic_fixture(seed):
    """Oracle diagnostic: known informative feature, noisy observable proxy, irrelevant feature.
    Independent samples; true hidden z is never used in observable model.
    """
    rng=np.random.default_rng(seed)
    ntrain,nval=10000,4000
    z=rng.normal(size=ntrain+nval)
    proxy=z+rng.normal(scale=1.0,size=len(z))
    noise=rng.normal(size=len(z))
    p=1/(1+np.exp(-(-2.0+1.5*z)))
    y=rng.binomial(1,p)
    tr=slice(0,ntrain);va=slice(ntrain,None)
    def fit_probe(x):
        x=np.asarray(x,float).reshape(-1,1)
        def obj(w):
            zz=np.clip(w[0]+w[1]*x[tr,0],-30,30)
            loss=np.mean(np.logaddexp(0,zz)-y[tr]*zz)+0.0001*w[1]**2
            pp=1/(1+np.exp(-zz))
            g=np.array([np.mean(pp-y[tr]),np.mean((pp-y[tr])*x[tr,0])+0.0002*w[1]])
            return loss,g
        r=minimize(obj,[-2,0],jac=True,method='L-BFGS-B')
        q=1/(1+np.exp(-np.clip(r.x[0]+r.x[1]*x[va,0],-30,30)))
        return float(-np.mean(y[va]*np.log(q)+(1-y[va])*np.log1p(-q)))
    return {'status':'SYNTHETIC_ORACLE_DIAGNOSTIC_ONLY','n_train':ntrain,'n_validation':nval,
            'oracle_hidden_z_nll':fit_probe(z),'observable_noisy_proxy_nll':fit_probe(proxy),
            'uninformative_noise_nll':fit_probe(noise),
            'warning':'Oracle hidden z is a diagnostic upper bound, never a legitimate forecasting input.'}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',default='v3c_stage2.py')
    ap.add_argument('--input',default='../LTEE_prospective/v3b_simulation')
    ap.add_argument('--output',default='../LTEE_prospective/v3c_stage23_audit')
    ap.add_argument('--train-seeds',type=int,nargs='+',default=[11,23])
    ap.add_argument('--validation-seeds',type=int,nargs='+',default=[42])
    ap.add_argument('--scenarios',nargs='+',default=['neutral','heterogeneous','context','regime_shift','linkage_only','recombination','selection_only','epistasis'])
    ap.add_argument('--penalties',nargs='+',type=float,default=[0.0001,0.001,0.01,0.1])
    ap.add_argument('--min-cutoff',type=int,default=10)
    ap.add_argument('--maxiter',type=int,default=300)
    ap.add_argument('--fixture-seed',type=int,default=917)
    args=ap.parse_args()
    if any(p<0 for p in args.penalties): ap.error('Negative penalty')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    rows,features,changes,fixture=real_audit(args)
    writecsv(out/'capacity_validation.csv',rows)
    writecsv(out/'feature_contributions.csv',features)
    writecsv(out/'prediction_deltas_validation.csv',changes)
    (out/'oracle_fixture.json').write_text(json.dumps(fixture,indent=2))
    (out/'protocol.json').write_text(json.dumps({'status':'EXPLORATORY_VALIDATION_ONLY','train_seeds':args.train_seeds,
       'validation_seeds':args.validation_seeds,'scenarios':args.scenarios,
       'test_seeds_accessed':False,'restricted_truth_accessed':False,
       'caveats':['Fixed nonlinear expansion is not a universal capacity test',
                  'Feature informativeness is not established by permutation alone',
                  'Oracle fixture is separate synthetic data, not actual simulator truth']},indent=2))
    print('=== CAPACITY VALIDATION ===')
    for r in rows: print(r['model'], 'NLL',round(r['validation_nll'],7),'grad_init',round(r['train_initial_gradient_norm'],7),'grad_final',round(r['train_final_gradient_norm'],7))
    print('=== SYNTHETIC ORACLE FIXTURE ===');print(json.dumps(fixture,indent=2))
    print('SAVED',out)
if __name__=='__main__': main()
