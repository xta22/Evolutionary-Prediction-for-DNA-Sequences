#!/usr/bin/env python3
"""Prepare a full-scale, TRAIN-only LTEE frozen-Evo2 mutation baseline."""
import argparse
import json
import random
from collections import Counter
from pathlib import Path
import pandas as pd

DNA = set('ACGT')

def read_fasta(path):
    with open(path) as handle:
        return ''.join(line.strip().upper() for line in handle if not line.startswith('>'))

def resolve_path(value, root):
    p = Path(str(value))
    candidates = [p, root / p]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'Cannot locate sequence: {value}')

def window(seq, index, size):
    start = index - size // 2
    end = start + size
    if start < 0 or end > len(seq):
        return None
    return seq[start:end]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='LTEE_model_ready/train.csv')
    ap.add_argument('--output', default='evo2_baseline_full_scale.jsonl')
    ap.add_argument('--window', type=int, default=2048)
    ap.add_argument('--max-mutations', type=int, default=10)
    ap.add_argument('--controls-per-mutation', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if args.window < 2 or args.window % 2:
        ap.error('--window must be an even integer >= 2')
    rng = random.Random(args.seed)
    root = Path.cwd()
    df = pd.read_csv(args.manifest)
    required = {'branch_id','population','delta_time','actual_mutation_count',
                'parent_sequence_file','child_sequence_file'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if 'split' in df.columns and not (df['split'] == 'train').all():
        raise ValueError('Refusing non-training rows in training baseline')
    counts = Counter()
    output = Path(args.output)
    with output.open('w') as out:
        for _, row in df.iterrows():
            branch = str(row['branch_id'])
            pop = str(row['population'])
            parent = read_fasta(resolve_path(row['parent_sequence_file'], root))
            child = read_fasta(resolve_path(row['child_sequence_file'], root))
            if len(parent) != len(child):
                raise ValueError(f'Genome lengths differ: {branch}')
            mutations = [i for i, (a,b) in enumerate(zip(parent, child)) if a != b]
            if len(mutations) != int(row['actual_mutation_count']):
                raise ValueError(f'Mutation count inconsistent for {branch}')
            mutation_set = set(mutations)
            valid = [i for i in mutations if parent[i] in DNA and child[i] in DNA
                     and window(parent,i,args.window) is not None]
            if not valid:
                counts[(pop,'branches_without_eligible_mutations')] += 1
                continue
            selected = rng.sample(valid, min(args.max_mutations,len(valid)))
            # Precompute eligible unchanged sites by base, once per branch.
            half = args.window // 2
            by_base = {b: [] for b in DNA}
            for i in range(half, len(parent) - half):
                if i not in mutation_set and parent[i] in DNA:
                    by_base[parent[i]].append(i)
            used_controls = set()
            for pos in selected:
                base = parent[pos]
                eligible = [i for i in by_base[base]
                            if i not in used_controls and abs(i-pos) > 100]
                if len(eligible) < args.controls_per_mutation:
                    counts[(pop,'mutations_skipped_controls')] += 1
                    continue
                controls = rng.sample(eligible,args.controls_per_mutation)
                used_controls.update(controls)
                for site,label in [(pos,1)] + [(i,0) for i in controls]:
                    seq = window(parent,site,args.window)
                    record = {
                        'branch_id':branch, 'population':pop,
                        'delta_time':float(row['delta_time']),
                        'position_1based':site+1, 'is_mutated':label,
                        'parent_base':parent[site], 'child_base':child[site],
                        'target_index':args.window//2, 'sequence':seq,
                        'window_size':args.window,
                        'matched_mutation_position_1based':pos+1,
                    }
                    out.write(json.dumps(record)+'\n')
                    counts[(pop,'mutations' if label else 'controls')] += 1
            counts[(pop,'branches_with_samples')] += 1
    print(f'Saved: {output}')
    for pop in sorted(df.population.unique()):
        print(f'{pop:7s} branches={counts[(pop,"branches_with_samples")]:3d} '
              f'mutations={counts[(pop,"mutations")]:4d} '
              f'controls={counts[(pop,"controls")]:4d}')
    print('TOTAL mutations:',sum(v for (p,k),v in counts.items() if k=='mutations'))
    print('TOTAL controls:',sum(v for (p,k),v in counts.items() if k=='controls'))

if __name__=='__main__':
    main()

