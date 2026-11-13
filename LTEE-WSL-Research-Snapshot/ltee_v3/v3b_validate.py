#!/usr/bin/env python3
"""Descriptive mechanism checks; no hypothesis test or certification."""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',default='../LTEE_prospective/v3b_simulation/run_summary.csv');args=ap.parse_args()
    df=pd.read_csv(args.input)
    print('=== STRUCTURAL STATUS ===')
    print(df.status.value_counts().to_string())
    print('\n=== MECHANISM DIAGNOSTICS (means over independent seeds) ===')
    cols=['mean_pre_mutations','mean_post_mutations','expected_pre_mutations','expected_post_mutations','crossovers_total','allele_a_end','double_mutant_end','r2_end','haplotype_diversity_end']
    print(df.groupby('scenario')[cols].agg(['mean','std']).round(4).to_string())
    base=df[df.scenario=='neutral'].set_index('seed')
    print('\n=== PAIRED DIFFERENCES VS NEUTRAL (descriptive) ===')
    for scenario,metric in [('selection_only','allele_a_end'),('epistasis','double_mutant_end'),('recombination','r2_end'),('heterogeneous','mean_pre_mutations'),('regime_shift','mean_post_mutations')]:
        x=df[df.scenario==scenario].set_index('seed')
        common=x.index.intersection(base.index)
        d=(x.loc[common,metric]-base.loc[common,metric]).dropna()
        print(f'{scenario:18s} {metric:24s} n={len(d)} mean_delta={d.mean() if len(d) else float("nan"):.5f} sd={d.std() if len(d)>1 else float("nan"):.5f}')
    print('\nCAUTION: same seeds share initial populations, not necessarily common random numbers after divergence.')
    print('LD can be undefined at fixed sites; epistasis is not identified by LD alone.')
    print('Mechanism switches and mutation counters are validated by unit tests; empirical effects need larger replications.')
if __name__=='__main__':main()
