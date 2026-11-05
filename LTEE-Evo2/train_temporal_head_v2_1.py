#!/usr/bin/env python3
"""V2.1: sampling-aware CTMC ablation on frozen Evo2 embeddings.

Uses train/validation only. Weighted objective approximates a genome-position
average under the stated branch-wise uniform-inclusion assumptions; it does
NOT establish calibrated biological mutation rates. Test data are untouched.
"""
import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score

# Reuse the EXACT architecture and loading logic from V2.
from train_temporal_head_v2 import CTMCHead, load_split


def attach_weights(metadata, path, split):
    df = pd.read_csv(path)
    needed = {'sample_id', 'branch_id', 'split', 'is_mutated', 'sampling_weight'}
    if not needed.issubset(df.columns):
        raise ValueError(f'Missing columns in {path}: {needed - set(df.columns)}')
    if df.sample_id.duplicated().any():
        raise ValueError(f'Duplicate sample IDs in {path}')
    if len(df) != len(metadata):
        raise ValueError(f'Weight/embedding row count mismatch: {split}')
    index = df.set_index('sample_id')
    weights = []
    for m in metadata:
        sid = m['sample_id']
        if sid not in index.index:
            raise ValueError(f'No weight for {sid}')
        r = index.loc[sid]
        if str(r['branch_id']) != str(m['branch_id']) or str(r['split']) != split or int(r['is_mutated']) != int(m['is_mutated']):
            raise ValueError(f'Metadata mismatch for {sid}')
        weights.append(float(r['sampling_weight']))
    w = torch.tensor(weights, dtype=torch.float32)
    if not torch.isfinite(w).all() or (w <= 0).any():
        raise ValueError('Nonpositive/nonfinite sampling weights')
    return w


def metrics(probs, parent, future, weights):
    idx = np.arange(len(parent))
    changed = (parent != future).astype(int)
    scores = 1 - probs[idx, parent]
    pred = probs.argmax(axis=1)
    nll = -np.log(np.clip(probs[idx, future], 1e-12, 1))
    w = weights / weights.sum()
    return {
        'n': len(parent),
        'sampled_change_fraction': float(changed.mean()),
        'weighted_change_fraction': float(np.dot(w, changed)),
        'sampled_nucleotide_log_loss': float(nll.mean()),
        'weighted_nucleotide_log_loss': float(np.dot(w, nll)),
        'sampled_nucleotide_accuracy': float((pred == future).mean()),
        'weighted_nucleotide_accuracy': float(np.dot(w, (pred == future).astype(float))),
        'sampled_mutation_auroc': float(roc_auc_score(changed, scores)),
        'sampled_mutation_auprc': float(average_precision_score(changed, scores)),
        'weighted_mutation_auroc': float(roc_auc_score(changed, scores, sample_weight=weights)),
        'weighted_mutation_auprc': float(average_precision_score(changed, scores, sample_weight=weights)),
        'mean_predicted_change_sampled': float(scores.mean()),
        'mean_predicted_change_weighted': float(np.dot(w, scores)),
        'weighted_mutation_brier': float(np.dot(w, (scores-changed)**2)),
        'mutated_site_accuracy_sampled': float((pred[changed == 1] == future[changed == 1]).mean()),
    }


def predict(model, ds, device, batch_size):
    model.eval()
    ps = []
    with torch.inference_mode():
        for x, p, y, t, w in DataLoader(ds, batch_size=batch_size, shuffle=False):
            probs, _ = model(x.to(device), p.to(device), t.to(device))
            ps.append(probs.cpu().numpy())
    return np.concatenate(ps, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--embedding-dir', default='results/temporal_head_v1_embeddings')
    ap.add_argument('--weights-dir', default='LTEE_temporal_head_v1')
    ap.add_argument('--output-dir', default='results/temporal_head_v2_1_training')
    ap.add_argument('--alphas', nargs='+', type=float, default=[1.0, 0.5, 0.0])
    ap.add_argument('--models', nargs='+', choices=['shared_ctmc','evo2_ctmc'], default=['shared_ctmc','evo2_ctmc'])
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=12)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=0.05)
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--rate-regularization', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if any(not 0 <= a <= 1 for a in args.alphas):
        raise ValueError('Alpha must be in [0,1]')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device('cpu')  # small head, no GPU allocation
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    tr = load_split(Path(args.embedding_dir), 'train')
    va = load_split(Path(args.embedding_dir), 'validation')
    if set(m['sample_id'] for m in tr[4]) & set(m['sample_id'] for m in va[4]):
        raise ValueError('Train/validation sample IDs overlap')
    tw = attach_weights(tr[4], Path(args.weights_dir)/'train_weights.csv', 'train')
    vw = attach_weights(va[4], Path(args.weights_dir)/'validation_weights.csv', 'validation')
    mean = tr[0].mean(0)
    std = tr[0].std(0).clamp_min(1e-4)
    train_x = ((tr[0]-mean)/std).clamp(-10,10)
    val_x = ((va[0]-mean)/std).clamp(-10,10)
    train_ds = TensorDataset(train_x,tr[1],tr[2],tr[3],tw)
    val_ds = TensorDataset(val_x,va[1],va[2],va[3],vw)
    # Full-dataset denominators ensure that the stochastic objective has the
    # intended alpha mixture even though mini-batches have different class mix.
    ntrain = len(tw)
    sumw_train = tw.double().sum().item()
    sumw_val = vw.double().sum().item()
    print(f'Train={ntrain}, validation={len(vw)}, device={device}',flush=True)
    print(f'Train weighted prevalence={float(tw[tr[1]!=tr[2]].sum()/tw.sum()):.9g}',flush=True)
    print('WARNING: weights use reference genome length as eligible-position approximation.',flush=True)
    print('Do not interpret fitted CTMC rates as biochemical mutation rates.',flush=True)
    results = {}
    for model_name in args.models:
        for alpha in args.alphas:
            tag = f'{model_name}_alpha_{alpha:.2f}'
            torch.manual_seed(args.seed)
            model = CTMCHead(hidden=args.hidden,dropout=args.dropout,use_context=(model_name=='evo2_ctmc')).to(device)
            optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
            loader = DataLoader(train_ds,batch_size=args.batch_size,shuffle=True)
            best_loss=float('inf'); best_epoch=0; best_state=None; history=[]
            print(f'\n=== {tag} ===',flush=True)
            for epoch in range(1,args.epochs+1):
                model.train()
                total_nll=0.; seen=0
                for x,p,y,t,w in loader:
                    optimizer.zero_grad(set_to_none=True)
                    probs,multiplier=model(x,p,t)
                    nll=-torch.log(probs[torch.arange(len(y)),y])
                    # Normalization across the FULL train split, not per minibatch.
                    # Multiplication by N makes this an unbiased stochastic gradient
                    # estimator of alpha*mean(nll)+(1-alpha)*weighted_mean(nll).
                    coeff=alpha+(1-alpha)*(ntrain*w.double()/sumw_train).float()
                    loss=(coeff*nll).mean()
                    if model_name=='evo2_ctmc':
                        loss=loss+args.rate_regularization*torch.log(multiplier).square().mean()
                    loss.backward()
                    optimizer.step()
                    total_nll+=nll.detach().sum().item();seen+=len(y)
                model.eval()
                v_unweighted=0.;v_weighted=0.;vn=0
                with torch.inference_mode():
                    for x,p,y,t,w in DataLoader(val_ds,batch_size=args.batch_size):
                        probs,_=model(x,p,t)
                        nll=-torch.log(probs[torch.arange(len(y)),y])
                        v_unweighted+=nll.sum().item()
                        v_weighted+=(w.double()*nll.double()).sum().item()
                        vn+=len(y)
                v_unweighted/=vn
                v_weighted/=sumw_val
                # Each alpha uses a matching validation selection objective.
                selection=alpha*v_unweighted+(1-alpha)*v_weighted
                history.append({'epoch':epoch,'train_unweighted_nll':total_nll/seen,
                                'validation_sampled_nll':v_unweighted,
                                'validation_weighted_nll':v_weighted,
                                'validation_mixed_objective':selection})
                print(f'Epoch {epoch:03d} val_sampled={v_unweighted:.6f} val_weighted={v_weighted:.6f} val_mixed={selection:.6f}',flush=True)
                if selection<best_loss-1e-6:
                    best_loss=selection;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
                elif epoch-best_epoch>=args.patience:
                    print('Early stopping',flush=True)
                    break
            model.load_state_dict(best_state)
            probs=predict(model,val_ds,device,args.batch_size)
            result=metrics(probs,va[1].numpy(),va[2].numpy(),vw.numpy().astype(np.float64))
            result.update({'alpha':alpha,'model':model_name,'best_epoch':best_epoch,'best_selection_loss':best_loss})
            results[tag]=result
            print('Validation:',json.dumps(result,indent=2),flush=True)
            torch.save({'model_state_dict':best_state,'model_name':model_name,'alpha':alpha,
                        'embedding_layer':'blocks.15','embedding_mean':mean,'embedding_std':std,
                        'config':vars(args),'validation_metrics':result},outdir/f'{tag}_best.pt')
            (outdir/f'{tag}_history.json').write_text(json.dumps(history,indent=2))
    (outdir/'validation_comparison.json').write_text(json.dumps(results,indent=2))
    print('\n=== V2.1 VALIDATION COMPARISON ===',flush=True)
    print(json.dumps(results,indent=2),flush=True)
    print('Saved to:',outdir,flush=True)

if __name__=='__main__':
    main()
