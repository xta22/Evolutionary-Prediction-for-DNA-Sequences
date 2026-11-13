#!/usr/bin/env python3
"""Read-only Stage 2.1 audit. Never refits models or reads restricted truth."""
import argparse, ast, json, re
from pathlib import Path
import numpy as np
import pandas as pd

KEYS = ['scenario','seed','fold_id','cutoff','target_generation']

def scalar(x):
    if isinstance(x, dict): return {str(k):scalar(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [scalar(v) for v in x]
    if isinstance(x,np.generic): return x.item()
    return x

def metric_comparison(df):
    cols=[x for x in KEYS if x in df.columns]
    if not cols or 'model' not in df or 'nll' not in df: raise ValueError('metrics needs model, nll and fold identity columns')
    if df.duplicated(cols+['model']).any(): raise ValueError('duplicate fold/model rows')
    base=df[df.model=='M1_shared'][cols+['nll']].rename(columns={'nll':'m1_nll'})
    out=[]
    for model in ['B1_latest','M2_history','M3_local_context']:
        x=df[df.model==model][cols+['nll']].merge(base,on=cols,validate='one_to_one')
        if len(x)==0:continue
        d=x.nll-x.m1_nll
        out.append(dict(model=model,n_folds=len(x),mean_nll_delta=float(d.mean()),median_nll_delta=float(d.median()),fraction_better=float((d<0).mean()),max_abs_delta=float(abs(d).max())))
    return out

def inspect_source(path):
    tree=ast.parse(path.read_text())
    funcs=[]; imports=[]; strings=[]
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            funcs.append({'name':node.name,'line':node.lineno,'args':[a.arg for a in node.args.args]})
        if isinstance(node,ast.Import): imports += [a.name for a in node.names]
        if isinstance(node,ast.ImportFrom): imports.append(node.module or '')
        if isinstance(node,ast.Constant) and isinstance(node.value,str) and re.search('future|truth|restricted|validation|test',node.value,re.I):strings.append({'line':node.lineno,'literal':node.value[:100]})
    return {'functions':funcs,'imports':sorted(set(imports)),'review_literals':strings[:60], 'warning':'Static source scan is not a proof of temporal isolation.'}

def inspect_json(obj):
    output={}
    if not isinstance(obj,dict):return {'type':str(type(obj))}
    for model,v in obj.items():
        if isinstance(v,dict):
            d={}
            for k,val in v.items():
                if isinstance(val,(int,float,str,bool)) or val is None:d[k]=val
                elif isinstance(val,list):
                    try:
                        a=np.asarray(val,dtype=float)
                        d[k]={'shape':list(a.shape),'l2_norm':float(np.linalg.norm(a)),'max_abs':float(np.max(abs(a))) if a.size else 0.0,'near_zero_fraction':float(np.mean(abs(a)<1e-8)) if a.size else None}
                    except (ValueError,TypeError):d[k]={'type':'list','length':len(val)}
                else:d[k]={'type':type(val).__name__}
            output[model]=d
    return output

def optional_table(path, kind):
    if not path or not Path(path).exists():return {'status':'NOT_AVAILABLE','reason':f'{kind} CSV not supplied'}
    df=pd.read_csv(path)
    numeric=df.select_dtypes(include='number')
    if kind=='features':
        exclude=set(KEYS+['label','y','future_base','target','is_mutated','split'])
        fields=[c for c in numeric if c not in exclude]
        return {'status':'DESCRIPTIVE_ONLY','rows':len(df),'features':{c:{'variance':float(numeric[c].var()),'missing_fraction':float(numeric[c].isna().mean()),'unique':int(numeric[c].nunique())} for c in fields[:150]},'warning':'Feature variance alone does not establish informativeness; label-based checks require a verified training-only feature export.'}
    cols=set(df.columns)
    if not {'model','prediction'}.issubset(cols):return {'status':'SCHEMA_UNSUPPORTED','columns':list(df.columns),'required':['model','prediction']}
    keys=[c for c in KEYS+['site','site_index','position','sample_id','split'] if c in df]
    if not keys:return {'status':'SCHEMA_UNSUPPORTED','reason':'No sample/site identity columns'}
    b=df[df.model=='M1_shared'][keys+['prediction']].rename(columns={'prediction':'baseline'})
    result={}
    for m in ['M2_history','M3_local_context']:
        x=df[df.model==m][keys+['prediction']].merge(b,on=keys,validate='one_to_one')
        delta=np.abs(x.prediction-x.baseline)
        result[m]={'n':len(x),'mean_abs_probability_delta':float(delta.mean()) if len(x) else None,'max_abs_probability_delta':float(delta.max()) if len(x) else None,'fraction_changed_1e-8':float((delta>1e-8).mean()) if len(x) else None}
    return {'status':'DIRECT_PREDICTION_COMPARISON','results':result}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--metrics',required=True);p.add_argument('--models',required=True);p.add_argument('--source',required=True)
    p.add_argument('--features');p.add_argument('--predictions');p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(a.metrics)
    if 'split' not in df:raise ValueError('Missing split column')
    test=df[df.split=='test']; val=df[df.split=='validation']
    result={'status':'READ_ONLY_PARTIAL_AUDIT','test_fold_metric_comparison':metric_comparison(test),'validation_fold_metric_comparison':metric_comparison(val) if len(val) else [],'model_metadata':inspect_json(json.loads(Path(a.models).read_text())),'source':inspect_source(Path(a.source)),'features':optional_table(a.features,'features'),'predictions':optional_table(a.predictions,'predictions'),'not_verified':['gradient norms / training loss curves (need optimizer trace or production instrumentation)','permutation importance (needs feature matrices and production prediction interface)','mechanism alignment (needs documented feature semantics and training-only observations)','true independent trajectory uncertainty (requires additional independent seeds)'],'policy':'Does not fit, select hyperparameters, or read restricted simulator truth. Test metrics are descriptive only; test was previously inspected.'}
    (out/'audit_report.json').write_text(json.dumps(scalar(result),indent=2,allow_nan=False))
    pd.DataFrame(result['test_fold_metric_comparison']).to_csv(out/'paired_metric_deltas.csv',index=False)
    print('STATUS:',result['status']);print('TEST PAIRED NLL DELTAS:');print(pd.DataFrame(result['test_fold_metric_comparison']).to_string(index=False));print('FEATURES:',result['features']['status']);print('PREDICTIONS:',result['predictions']['status']);print('NOT VERIFIED:',*result['not_verified'],sep='\n - ');print('Saved:',out)
if __name__=='__main__':main()
