#!/usr/bin/env python3
"""Prepare V3 branch time table from LTEE_model_ready/model_manifest.csv (WSL)."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='LTEE_model_ready/model_manifest.csv')
    ap.add_argument('--output', default='LTEE_temporal_head_v3/branch_time_metadata.csv')
    ap.add_argument('--tolerance', type=float, default=1e-3)
    args = ap.parse_args()
    df = pd.read_csv(args.manifest)
    required = ['branch_id','split','population','parent_time','child_time','delta_time','actual_mutation_count']
    missing = set(required)-set(df.columns)
    if missing:
        raise ValueError(f'Missing manifest columns: {sorted(missing)}. Available: {df.columns.tolist()}')
    df = df.copy()
    for c in ['parent_time','child_time','delta_time','actual_mutation_count']:
        df[c] = pd.to_numeric(df[c],errors='raise')
    if df[required].isna().any().any():
        raise ValueError('Missing required metadata')
    if df.branch_id.duplicated().any():
        raise ValueError('Duplicate branch IDs')
    if not df.split.isin(['train','validation','test']).all():
        raise ValueError('Unknown split names')
    if not np.isfinite(df[['parent_time','child_time','delta_time']].to_numpy()).all():
        raise ValueError('Nonfinite times')
    if (df.parent_time < 0).any() or (df.delta_time <= 0).any():
        raise ValueError('Negative parent time or nonpositive interval')
    error = (df.child_time-df.parent_time-df.delta_time).abs()
    if (error > args.tolerance).any():
        raise ValueError(f'Time inconsistencies: {(error > args.tolerance).sum()}; max={error.max()}')
    if (df.actual_mutation_count < 0).any():
        raise ValueError('Negative mutation count')
    optional = [c for c in ['parent_time_source','child_time_source','temporal_quality','parent_confidence','child_confidence'] if c in df.columns]
    out = df[required+optional].copy().sort_values(['split','population','branch_id'])
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(path,index=False)
    print('Saved:',path)
    print('Rows:',len(out),'unique branches:',out.branch_id.nunique())
    print(out.groupby('split').agg(branches=('branch_id','size'),populations=('population','nunique'),min_parent=('parent_time','min'),max_child=('child_time','max')).to_string())
    print('Max interval discrepancy:',error.max())
    print('Note: inferred internal-node times are estimates, not directly observed.')

if __name__=='__main__': main()
