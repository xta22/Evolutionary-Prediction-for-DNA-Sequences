#!/usr/bin/env python3
"""Descriptive LTEE branch-level rate-scale and Poisson diagnostic (CPU only)."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='LTEE_model_ready/model_manifest.csv')
    ap.add_argument('--genome-length', type=int, default=4629812)
    ap.add_argument('--output-dir', default='LTEE_rate_diagnostic')
    args = ap.parse_args()
    df = pd.read_csv(args.manifest)
    required = {'branch_id','population','split','actual_mutation_count','delta_time'}
    missing = required - set(df.columns)
    if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
    if df.branch_id.duplicated().any(): raise ValueError('Duplicate branch_id')
    df['M'] = pd.to_numeric(df.actual_mutation_count, errors='raise')
    df['dt'] = pd.to_numeric(df.delta_time, errors='raise')
    if not np.isfinite(df[['M','dt']].to_numpy()).all(): raise ValueError('Nonfinite count/time')
    if (df.M < 0).any() or (df.dt <= 0).any() or (df.M % 1 != 0).any(): raise ValueError('Invalid count/time')
    L = args.genome_length
    if L <= 0: raise ValueError('Genome length must be positive')
    df['changed_fraction'] = df.M / L
    df['rate_proxy_per_bp_generation'] = df.M / (L * df.dt)
    # Train-only reference prevents validation/test-informed rate fitting.
    train = df[df.split == 'train']
    if train.empty: raise ValueError('No training branches')
    r_train = train.M.sum() / (L * train.dt.sum())
    r_all = df.M.sum() / (L * df.dt.sum())
    df['expected_count_train_rate'] = L * r_train * df.dt
    df['pearson_component'] = (df.M - df.expected_count_train_rate)**2 / df.expected_count_train_rate.clip(lower=1e-12)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out/'branch_rates.csv',index=False)
    pop = df.groupby(['split','population']).agg(branches=('branch_id','count'),changes=('M','sum'),total_generations=('dt','sum'),mean_changes=('M','mean'),median_changes=('M','median'),zero_fraction=('M',lambda s:float((s==0).mean()))).reset_index()
    pop['pooled_rate_per_bp_generation'] = pop.changes / (L * pop.total_generations)
    pop.to_csv(out/'population_rates.csv',index=False)
    split = df.groupby('split').agg(branches=('branch_id','count'),changes=('M','sum'),total_generations=('dt','sum'),median_changes=('M','median'),zero_fraction=('M',lambda s:float((s==0).mean())),pearson_sum=('pearson_component','sum')).reset_index()
    split['pooled_rate_per_bp_generation'] = split.changes/(L*split.total_generations)
    split.to_csv(out/'split_rates.csv',index=False)
    print('=== RATE SCALE ===')
    print('Genome length:',L,'Total branches:',len(df))
    print('TRAIN pooled rate per bp per generation:',f'{r_train:.6e}')
    print('ALL pooled rate (descriptive only):',f'{r_all:.6e}')
    print('\n=== BY SPLIT ===')
    print(split.to_string(index=False,float_format=lambda x:f'{x:.6g}'))
    print('\n=== TRAIN POPULATIONS ===')
    print(pop[pop.split=='train'].to_string(index=False,float_format=lambda x:f'{x:.6g}'))
    print('\n=== POISSON DISPERSION (train, common rate) ===')
    expected = df.loc[df.split=='train','expected_count_train_rate']
    obs = train.M
    pearson = ((obs-expected)**2/expected.clip(lower=1e-12)).sum()
    dfree = len(train)-1
    print('Pearson chi-square:',f'{pearson:.3f}')
    print('Degrees of freedom:',dfree)
    print('Dispersion ratio:',f'{pearson/dfree:.3f}' if dfree>0 else 'N/A')
    print('Observed zero branches:',int((obs==0).sum()))
    print('Expected zero branches (sum exp(-expected)):',f'{np.exp(-expected).sum():.2f}')
    print('\nNOTE: Count overdispersion can reflect heterogeneity, dependence, reconstruction and dating error; not a formal Poisson GOF test.')
    print('NOTE: These are observed branch differences, not biochemical mutation rates. Branches can be phylogenetically dependent.')
    print('Saved:',out)

if __name__=='__main__': main()
