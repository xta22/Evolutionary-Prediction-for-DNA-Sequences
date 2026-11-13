#!/usr/bin/env python3
"""Read-only V3-C B1/B2/B3 trend audit. No future-label fitting."""
import argparse, ast, csv, json, re
from pathlib import Path
from collections import defaultdict
import numpy as np

MODELS=('B1_latest','B2_shrunk_trend','B3_site_trend')

def read_csv(path):
    with path.open(newline='') as f: return list(csv.DictReader(f))

def number(x):
    try: return float(x)
    except (ValueError,TypeError): return float('nan')

def audit_metrics(rows):
    if not rows: return {'status':'NO_METRICS'}
    fields=list(rows[0]); keys=[x for x in ('split','scenario','seed','fold_id','population','cutoff','target_generation','horizon') if x in fields]
    metrics=[x for x in ('nll','brier','abs_count_error') if x in fields]
    grouped=defaultdict(dict)
    for r in rows:
        if r.get('model') in MODELS:
            k=tuple(r.get(x,'') for x in keys)
            if r['model'] in grouped[k]: raise ValueError('Duplicate model row for key '+str(k))
            grouped[k][r['model']]=r
    counts=defaultdict(lambda: {'compared':0,'exact_equal':0,'max_abs_difference':0.0})
    for k,g in grouped.items():
        if not all(m in g for m in MODELS): continue
        for other in MODELS[1:]:
            label=f'{other}_vs_B1'
            counts[label]['compared']+=1
            diffs=[abs(number(g[other][x])-number(g[MODELS[0]][x])) for x in metrics]
            if all(np.isfinite(v) and v==0 for v in diffs): counts[label]['exact_equal']+=1
            valid=[v for v in diffs if np.isfinite(v)]
            if valid: counts[label]['max_abs_difference']=max(counts[label]['max_abs_difference'],max(valid))
    return {'group_keys':keys,'metric_columns':metrics,'comparisons':dict(counts),'warning':'Aggregate equality does not prove per-site prediction identity.'}

def inspect_source(path):
    src=path.read_text(); tree=ast.parse(src)
    symbols=[]
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            if re.search('trend|predict|valid|alpha|forecast|fit|select',node.name,re.I): symbols.append({'name':node.name,'line':node.lineno})
    hits=[]
    for i,line in enumerate(src.splitlines(),1):
        if re.search(r'B[123]_|alpha|trend|argmin|validation',line,re.I):
            hits.append({'line':i,'text':line.strip()[:180]})
    return {'path':str(path),'relevant_symbols':symbols,'relevant_lines':hits[:200], 'note':'Static inspection only; no claim of runtime equivalence.'}

def find_selection_files(root):
    out=[]
    for p in sorted(root.rglob('*.json')):
        if p.stat().st_size>3_000_000: continue
        try: obj=json.loads(p.read_text())
        except (UnicodeError,ValueError): continue
        matches=[]
        def walk(x,where='',depth=0):
            if depth>7 or len(matches)>100: return
            if isinstance(x,dict):
                for k,v in x.items():
                    q=f'{where}.{k}' if where else str(k)
                    if re.search(r'alpha|trend|hyperparam|selected|best',str(k),re.I): matches.append({'path':q,'value':str(v)[:250]})
                    walk(v,q,depth+1)
            elif isinstance(x,list):
                for j,v in enumerate(x[:40]): walk(v,f'{where}[{j}]',depth+1)
        walk(obj)
        if matches: out.append({'file':str(p),'matches':matches[:100]})
    return out

def synthetic_fixture():
    # Independent mathematical reference only, NOT an integration test of v3c_forecast.py.
    t=np.array([0.,10.,20.]); y=np.array([[.20,.80],[.30,.70],[.40,.60]])
    latest=y[-1]; slopes=(y[-1]-y[-2])/(t[-1]-t[-2]); horizon=10.
    trend=np.clip(latest+horizon*slopes,1e-6,1-1e-6)
    assert not np.array_equal(latest,trend)
    assert np.allclose(trend,[.50,.50])
    return {'latest':latest.tolist(),'nonzero_trend_prediction':trend.tolist(),'warning':'Only validates reference arithmetic, not production B2/B3 implementation.'}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--metrics',type=Path,required=True); p.add_argument('--source',type=Path,required=True); p.add_argument('--results-dir',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    for path in (a.metrics,a.source,a.results_dir):
        if not path.exists(): p.error('Missing input: '+str(path))
    report={'metrics':audit_metrics(read_csv(a.metrics)),'source':inspect_source(a.source),'selection_metadata':find_selection_files(a.results_dir),'reference_fixture':synthetic_fixture(),'conclusion':'INSUFFICIENT_FOR_PREDICTION_IDENTITY; inspect selected hyperparameters and actual per-site predictions.','read_only':True}
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'trend_audit.json').write_text(json.dumps(report,indent=2))
    print('=== METRIC IDENTITY ===');print(json.dumps(report['metrics'],indent=2))
    print('\n=== RELEVANT SOURCE SYMBOLS ===');print(json.dumps(report['source']['relevant_symbols'],indent=2))
    print('\n=== SELECTION METADATA ===');print(json.dumps(report['selection_metadata'],indent=2)[:3500])
    print('\nSaved:',a.output/'trend_audit.json')
    print('NOTE: To prove why B2/B3 equal B1, inspect production predictions and selected alphas; this audit does not invent those values.')
if __name__=='__main__': main()
