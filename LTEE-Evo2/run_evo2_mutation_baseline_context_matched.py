#!/usr/bin/env python3
"""Frozen Evo2 full-scale site ranking. Resumable; no model training."""
import argparse
import csv
import json
from pathlib import Path
import torch
from evo2 import Evo2

DNA = 'ACGT'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',default='evo2_baseline_context_matched.jsonl')
    ap.add_argument('--output',default='results/evo2_baseline_matched.csv')
    ap.add_argument('--checkpoint-every',type=int,default=50)
    args=ap.parse_args()
    input_path=Path(args.input)
    output_path=Path(args.output)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    samples=[json.loads(line) for line in input_path.open() if line.strip()]
    if not samples:
        raise RuntimeError('No samples in input')
    keys=[k for k in samples[0] if k!='sequence']
    fields=keys+['p_parent','mutation_score']+[f'p_{b}' for b in DNA]
    completed=0
    if output_path.exists() and output_path.stat().st_size:
        with output_path.open(newline='') as handle:
            completed=max(0,sum(1 for _ in csv.reader(handle))-1)
        if completed>len(samples):
            raise RuntimeError('Output has more rows than input; remove or rename it')
    print(f'Total={len(samples)}, completed={completed}',flush=True)
    if completed==len(samples):
        print('All rows already computed.',flush=True)
        return
    model=Evo2('evo2_7b_base')
    token_ids={b:int(model.tokenizer.tokenize(b)[0]) for b in DNA}
    mode='a' if completed else 'w'
    with output_path.open(mode,newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        if not completed:
            writer.writeheader()
        for n in range(completed,len(samples)):
            sample=samples[n]
            target=int(sample['target_index'])
            tokens=model.tokenizer.tokenize(sample['sequence'])
            if not 0<target<len(tokens):
                raise ValueError(f'Invalid target index at sample {n}')
            input_ids=torch.tensor(tokens,dtype=torch.long,device='cuda').unsqueeze(0)
            with torch.inference_mode():
                output=model(input_ids)
                logits=output[0][0][0,target-1]
                dna_logits=torch.stack([logits[token_ids[b]] for b in DNA]).float()
                probs=torch.softmax(dna_logits,dim=0).cpu().tolist()
            p={b:float(probs[i]) for i,b in enumerate(DNA)}
            parent=sample['parent_base']
            record={k:v for k,v in sample.items() if k!='sequence'}
            record.update({'p_parent':p[parent], 'mutation_score':1-p[parent]})
            record.update({f'p_{b}':p[b] for b in DNA})
            writer.writerow(record)
            if (n+1)%args.checkpoint_every==0:
                handle.flush()
                print(f'Processed {n+1}/{len(samples)}',flush=True)
            del input_ids,output,logits
    print(f'Saved: {output_path}',flush=True)
    # Evaluation is descriptive; samples within branches are not independent.
    import pandas as pd
    from sklearn.metrics import roc_auc_score,average_precision_score
    df=pd.read_csv(output_path)
    print('\n=== Context-matched BASELINE ===',flush=True)
    print('Rows:',len(df),'Branches:',df.branch_id.nunique(),
          'Populations:',df.population.nunique(),flush=True)
    print('Labels:',df.is_mutated.value_counts().to_dict(),flush=True)
    if df.is_mutated.nunique()==2:
        print('AUROC:',roc_auc_score(df.is_mutated,df.mutation_score),flush=True)
        print('AUPRC:',average_precision_score(df.is_mutated,df.mutation_score),flush=True)
    print(df.groupby('is_mutated').mutation_score.agg(['count','mean','median']).to_string(),flush=True)
    print('\nBy population:',flush=True)
    print(pd.crosstab(df.population,df.is_mutated).to_string(),flush=True)

if __name__=='__main__':
    main()

