#!/usr/bin/env python3
"""LTEE branch-count heterogeneity investigation (training split only).
Requires pandas, numpy, scipy, statsmodels, patsy.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix
from scipy.special import gammaln

L = 4629812

def fit_poisson(name, y, X, offset):
    fit = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset).fit(maxiter=200)
    mu = np.asarray(fit.predict(X, offset=offset))
    pearson = float(np.sum((y - mu)**2 / np.maximum(mu, 1e-12)))
    return dict(name=name, fit=fit, mu=mu, pearson=pearson,
                dispersion=pearson / max(1, len(y)-X.shape[1]),
                zero_expected=float(np.exp(-mu).sum()),
                aic=float(fit.aic), deviance=float(fit.deviance),
                df=int(len(y)-X.shape[1]), params=int(X.shape[1]))

def build_design(df, mode, spline_df=4):
    pop = pd.get_dummies(df['population'].astype(str), prefix='pop', drop_first=True, dtype=float)
    base = pd.DataFrame({'intercept': np.ones(len(df))}, index=df.index)
    if mode == 'common':
        return base
    base = pd.concat([base, pop], axis=1)
    if mode == 'population':
        return base
    if mode == 'linear':
        return pd.concat([base, pd.DataFrame({'midpoint_10k': df['midpoint'].to_numpy()/10000}, index=df.index)], axis=1)
    if mode == 'spline':
        spl = dmatrix(f'bs(x, df={spline_df}, degree=3, include_intercept=False) - 1',
                      {'x': df['midpoint'].to_numpy()/10000}, return_type='dataframe')
        spl.index = df.index
        spl.columns = [f'spline_{i}' for i in range(spl.shape[1])]
        return pd.concat([base, spl], axis=1)
    raise ValueError(mode)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest', default='LTEE_model_ready/model_manifest.csv')
    ap.add_argument('--output', default='LTEE_heterogeneity_investigation')
    ap.add_argument('--genome-length', type=int, default=L)
    ap.add_argument('--trim-top', type=int, default=5, help='Remove top N mutation-count branches for sensitivity')
    args=ap.parse_args()
    out=Path(args.output); out.mkdir(parents=True, exist_ok=True)
    df=pd.read_csv(args.manifest)
    required=['branch_id','split','population','actual_mutation_count','delta_time','parent_time','child_time']
    missing=[x for x in required if x not in df.columns]
    if missing:
        raise SystemExit('Missing required columns: '+str(missing)+'\nAvailable: '+str(df.columns.tolist()))
    df=df.loc[df['split'].astype(str).str.lower()=='train'].copy()
    if df.empty: raise SystemExit('No training branches found')
    for col in ['actual_mutation_count','delta_time','parent_time','child_time']:
        df[col]=pd.to_numeric(df[col], errors='coerce')
    if df[required].isna().any().any():
        raise SystemExit('Missing/non-numeric required values in training rows')
    if (df['delta_time']<=0).any() or (df['actual_mutation_count']<0).any():
        raise SystemExit('Non-positive delta_time or negative count')
    mismatch=np.abs((df.child_time-df.parent_time)-df.delta_time)
    if (mismatch>np.maximum(1e-3,df.delta_time.abs()*1e-5)).any():
        print('WARNING: parent/child times and delta_time differ on', int((mismatch>np.maximum(1e-3,df.delta_time.abs()*1e-5)).sum()), 'branches')
    df['midpoint']=(df.parent_time+df.child_time)/2
    df['count']=df.actual_mutation_count.astype(float)
    df=df.reset_index(drop=True)

    def run_models(data, tag):
        y=data['count'].to_numpy()
        offset=np.log(args.genome_length*data.delta_time.to_numpy())
        modes=['common','population','linear','spline']
        fits=[]; records=[]
        for mode in modes:
            X=build_design(data,mode)
            res=fit_poisson(mode,y,X,offset)
            fits.append(res)
            records.append({k:res[k] for k in ['name','aic','deviance','pearson','dispersion','zero_expected','df','params']})
            if tag=='full':
                data[f'expected_{mode}']=res['mu']
        tab=pd.DataFrame(records)
        tab.to_csv(out/f'{tag}_model_comparison.csv',index=False)
        print('\n===',tag.upper(),'MODEL COMPARISON ===')
        print(tab.to_string(index=False,float_format=lambda x:f'{x:.4g}'))
        if tag=='full':
            print('Observed zero-count branches:',int((y==0).sum()))
            for res in fits:
                if res['name']=='linear':
                    X=build_design(data,'linear')
                    coef=res['fit'].params['midpoint_10k']
                    print('Linear time effect: rate ratio per 10,000 generations =',float(np.exp(coef)))
            data.to_csv(out/'train_branch_diagnostics.csv',index=False)
        return tab

    print('Training branches:',len(df),'Populations:',df.population.nunique())
    print('Genome length:',args.genome_length)
    print('Midpoint range:',df.midpoint.min(),df.midpoint.max())
    full=run_models(df.copy(),'full')
    pop=(df.groupby('population').agg(branches=('branch_id','size'),changes=('count','sum'),total_generations=('delta_time','sum'),zero_branches=('count',lambda x:int((x==0).sum()))).reset_index())
    pop['pooled_rate']=pop.changes/(args.genome_length*pop.total_generations)
    pop.to_csv(out/'population_summary.csv',index=False)
    print('\n=== POPULATION SUMMARY ===')
    print(pop.to_string(index=False,float_format=lambda x:f'{x:.5g}'))
    outliers=df.nlargest(min(15,len(df)),'count')[['branch_id','population','parent_time','child_time','delta_time','count']]
    outliers.to_csv(out/'largest_branches.csv',index=False)
    print('\n=== LARGEST COUNT BRANCHES ===')
    print(outliers.to_string(index=False))
    if args.trim_top and len(df)-args.trim_top>25:
        trimmed=df.drop(index=df.nlargest(args.trim_top,'count').index).reset_index(drop=True)
        run_models(trimmed,'trimmed')
    print('\nIMPORTANT: AIC is in-sample; correlated phylogenetic branches violate naive independent-Poisson inference.')
    print('Midpoint time approximates stage; spline is not an integrated NHPP. Compare residual dispersion and trimmed results.')
    print('Saved:',out)

if __name__=='__main__': main()
