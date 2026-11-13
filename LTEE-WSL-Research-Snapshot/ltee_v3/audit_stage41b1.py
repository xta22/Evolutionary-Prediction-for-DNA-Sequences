#!/usr/bin/env python3
"""Stage 4.1B.1: train-only identifiability and trajectory bootstrap audit.
No test seeds, no simulator truth, and no modifications to production scripts.
"""
import argparse, csv, hashlib, importlib.util, json
from pathlib import Path
import numpy as np
from scipy.special import expit


def import_file(path, name):
    spec=importlib.util.spec_from_file_location(name,str(path.resolve()))
    if spec is None or spec.loader is None: raise RuntimeError('Cannot import '+str(path))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def design_summary(rows, mod, kind):
    x=np.concatenate([mod.features(r,kind) for r in rows],axis=0)
    singular=np.linalg.svd(x,compute_uv=False)
    tol=np.finfo(float).eps*max(x.shape)*singular[0] if len(singular) else 0
    rank=int(np.sum(singular>tol))
    return {'shape':list(x.shape),'rank':rank,'columns':x.shape[1],
        'singular_values':singular.tolist(),'condition_number':float(singular[0]/singular[-1]) if singular[-1]>tol else None,
        'column_means':np.mean(x,axis=0).tolist(),'column_std':np.std(x,axis=0).tolist(),
        'column_min':np.min(x,axis=0).tolist(),'column_max':np.max(x,axis=0).tolist(),
        'unique_horizons':sorted(set(float(r['horizon']) for r in rows)),
        'full_column_rank':rank==x.shape[1]}


def mean_nll(rows, theta, kind, mod, prod):
    return float(np.mean([mod.fold_nll(r,mod.predict(r,theta,kind,prod),prod) for r in rows]))


def group_trajectories(rows):
    groups={}
    for r in rows:
        key=(str(r['scenario']),int(r['seed']))
        groups.setdefault(key,[]).append(r)
    return groups


def bootstrap(rows, val, mod, prod, kind, penalty, reps, seed):
    groups=group_trajectories(rows); keys=list(groups)
    rng=np.random.default_rng(seed)
    records=[]
    for b in range(reps):
        idx=rng.integers(0,len(keys),len(keys))
        sampled=[r for i in idx for r in groups[keys[int(i)]]]
        try:
            theta,info=mod.fit(sampled,kind,prod,penalty)
            record={'replicate':b,'success':True,'n_trajectories':len(keys),
                    'coefficients':theta.tolist(),'validation_nll':mean_nll(val,theta,kind,mod,prod),
                    'train_nll':mean_nll(sampled,theta,kind,mod,prod)}
        except Exception as exc:
            record={'replicate':b,'success':False,'error':str(exc)}
        records.append(record)
    return records


def prediction_sensitivity(rows, theta_global, theta_history, mod, prod):
    changes=[]; rho_changes=[]
    for r in rows:
        p0=mod.predict(r,theta_global,'global',prod)
        p1=mod.predict(r,theta_history,'history',prod)
        changes.extend(np.abs(p1-p0).ravel())
        rg=expit(mod.features(r,'global')@theta_global)
        rh=expit(mod.features(r,'history')@theta_history)
        rho_changes.extend(np.abs(rh-rg).ravel())
    return {'mean_abs_joint_probability_delta':float(np.mean(changes)),
            'max_abs_joint_probability_delta':float(np.max(changes)),
            'mean_abs_rho_delta':float(np.mean(rho_changes)),
            'max_abs_rho_delta':float(np.max(rho_changes))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True,help='Production stage4.py')
    p.add_argument('--model-source',type=Path,required=True,help='Production stage41b.py')
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--selected-models',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train-seeds',type=int,nargs='+',default=[11,23,67,101])
    p.add_argument('--validation-seeds',type=int,nargs='+',default=[42])
    p.add_argument('--scenarios',nargs='*',default=[])
    p.add_argument('--bootstrap-reps',type=int,default=50)
    p.add_argument('--bootstrap-seed',type=int,default=20261113)
    args=p.parse_args()
    if not args.train_seeds or not args.validation_seeds or set(args.train_seeds)&set(args.validation_seeds):p.error('invalid or overlapping seeds')
    if args.bootstrap_reps<1:p.error('bootstrap reps must be positive')
    if args.output.resolve()==args.input.resolve():p.error('output cannot equal input')
    prod=import_file(args.source,'stage4_production_audit')
    mod=import_file(args.model_source,'stage41b_production_audit')
    selection=json.loads(args.selected_models.read_text())
    for kind in ('global','horizon','history'):
        if 'M5_'+kind not in selection:raise ValueError('Missing selected model M5_'+kind)
    scenarios=set(args.scenarios) if args.scenarios else None
    train=prod.load_folds(args.input,args.train_seeds,scenarios,'adjacent',0)
    val=prod.load_folds(args.input,args.validation_seeds,scenarios,'adjacent',0)
    summaries={kind:design_summary(train,mod,kind) for kind in ('global','horizon','history')}
    coefficients={kind:np.asarray(selection['M5_'+kind]['coefficients'],float) for kind in summaries}
    penalties={kind:float(selection['M5_'+kind]['penalty']) for kind in summaries}
    for kind,theta in coefficients.items():
        if len(theta)!=summaries[kind]['columns']:raise ValueError('coefficient dimension mismatch: '+kind)
    original={kind:{'train_nll':mean_nll(train,coefficients[kind],kind,mod,prod),
                    'validation_nll':mean_nll(val,coefficients[kind],kind,mod,prod),
                    'penalty':penalties[kind], 'coefficients':coefficients[kind].tolist()} for kind in summaries}
    stability={kind:bootstrap(train,val,mod,prod,kind,penalties[kind],args.bootstrap_reps,args.bootstrap_seed+idx)
               for idx,kind in enumerate(('global','horizon','history'))}
    bootstrap_summary={}
    for kind,records in stability.items():
        good=[r for r in records if r['success']]
        bootstrap_summary[kind]={'successful':len(good),'attempted':len(records)}
        if good:
            a=np.asarray([r['coefficients'] for r in good])
            bootstrap_summary[kind].update({'coefficient_mean':a.mean(axis=0).tolist(),
                'coefficient_std':a.std(axis=0).tolist(),
                'validation_nll_mean':float(np.mean([r['validation_nll'] for r in good])),
                'validation_nll_std':float(np.std([r['validation_nll'] for r in good]))})
    sensitivity=prediction_sensitivity(val,coefficients['global'],coefficients['history'],mod,prod)
    report={'status':'EXPLORATORY_IDENTIFIABILITY_AUDIT','n_train_folds':len(train),'n_validation_folds':len(val),
            'n_train_trajectories':len(group_trajectories(train)),
            'n_validation_trajectories':len(group_trajectories(val)),
            'design':summaries,'original':original,'bootstrap':bootstrap_summary,'prediction_sensitivity':sensitivity,
            'source_hashes':{'stage4':hashlib.sha256(args.source.read_bytes()).hexdigest(),
                             'stage41b':hashlib.sha256(args.model_source.read_bytes()).hexdigest()},
            'limitations':['Bootstrap unit is scenario x seed; same seed across scenarios is not necessarily independent.',
                           'Overlapping folds remain correlated within trajectories.',
                           'Bootstrap resamples and evaluates on the same fixed development validation set.',
                           'No locked test or simulator truth accessed.',
                           'Rank deficiency makes coefficient-level interpretation unreliable.']}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'stage41b1_report.json').write_text(json.dumps(report,indent=2))
    with (args.output/'bootstrap_replicates.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['model','replicate','success','n_trajectories','coefficients','train_nll','validation_nll','error'])
        w.writeheader()
        for kind,records in stability.items():
            for r in records:
                w.writerow({'model':kind,'replicate':r['replicate'],'success':r['success'],
                    'n_trajectories':r.get('n_trajectories',''),
                    'coefficients':json.dumps(r.get('coefficients',[])),
                    'train_nll':r.get('train_nll',''),'validation_nll':r.get('validation_nll',''),'error':r.get('error','')})
    print('=== FEATURE IDENTIFIABILITY ===')
    for kind,s in summaries.items():
        print(kind,'rank',s['rank'],'/',s['columns'],'horizons',s['unique_horizons'],'std',np.round(s['column_std'],6))
    print('=== ORIGINAL VALIDATION ===')
    for kind,s in original.items():print(kind,'nll',round(s['validation_nll'],8),'penalty',s['penalty'])
    print('=== TRAJECTORY BOOTSTRAP ===')
    for kind,s in bootstrap_summary.items():print(kind,'success',s['successful'],'/',s['attempted'],'coefficient_std',np.round(s.get('coefficient_std',[]),5))
    print('=== PREDICTION SENSITIVITY ===')
    print(json.dumps(sensitivity,indent=2))
    print('Saved',args.output)

if __name__=='__main__':main()
