#!/usr/bin/env python3
"""V3 diagnostic: gradients, rate decomposition, grouped calibration and counterfactuals.
Run in the same directory as train_temporal_head_v3.py and train_temporal_head_v2.py.
Uses train and validation only. Does not update model checkpoints.
"""
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from train_temporal_head_v3 import StageCTMC, load_weights, load_times, BASE_RATE
from train_temporal_head_v2 import load_split


def group_table(meta, start, end, weights, probs, parent, future, grouping):
    p=probs.detach().numpy(); a=parent.numpy(); b=future.numpy(); w=weights.numpy()
    scores=1-p[np.arange(len(a)),a]; changed=(a!=b).astype(float)
    d=pd.DataFrame({'population':[m['population'] for m in meta],
                    'branch_id':[m['branch_id'] for m in meta],
                    'midpoint':((start+end)/2).numpy(), 'weight':w,
                    'observed':changed,'predicted':scores})
    if grouping=='time':
        d['group']=pd.cut(d.midpoint, bins=[-np.inf,2500,5000,7500,10000,np.inf],
                          labels=['<=2500','2500-5000','5000-7500','7500-10000','>10000']).astype(str)
    else: d['group']=d.population
    rows=[]
    for name,g in d.groupby('group',sort=True):
        ww=g.weight.to_numpy();ww=ww/ww.sum()
        rows.append({'group':name,'n_samples':len(g),'n_branches':g.branch_id.nunique(),
                     'observed_weighted':float(np.dot(ww,g.observed)),
                     'predicted_weighted':float(np.dot(ww,g.predicted)),
                     'observed_sampled':float(g.observed.mean()),
                     'predicted_sampled':float(g.predicted.mean()),
                     'total_global_weight':float(g.weight.sum())})
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--embedding-dir',default='results/temporal_head_v1_embeddings')
    ap.add_argument('--weights-dir',default='LTEE_temporal_head_v1')
    ap.add_argument('--metadata',default='LTEE_temporal_head_v3/branch_time_metadata.csv')
    ap.add_argument('--checkpoint-dir',default='results/temporal_head_v3_training')
    ap.add_argument('--output-dir',default='results/temporal_head_v3_diagnostics')
    ap.add_argument('--counterfactual-samples',type=int,default=256)
    args=ap.parse_args()
    torch.set_num_threads(4)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    tab=pd.read_csv(args.metadata)
    assert not tab.branch_id.duplicated().any()
    tab=tab.set_index('branch_id')
    tr=load_split(Path(args.embedding_dir),'train'); va=load_split(Path(args.embedding_dir),'validation')
    results={}
    for variant in 'ABC':
        ckpt_path=Path(args.checkpoint_dir)/f'v3_{variant}_best.pt'
        ck=torch.load(ckpt_path,map_location='cpu',weights_only=False)
        pops=ck['populations']; cfg=ck['config']
        model=StageCTMC(variant,len(pops),hidden=cfg['hidden'],dropout=cfg['dropout'],
                        use_context=cfg['context']=='on').cpu()
        model.load_state_dict(ck['state_dict']);model.eval()
        def prep(ds,split):
            x=((ds[0]-ck['embedding_mean'])/ck['embedding_std']).clamp(-10,10)
            s,e,p=load_times(ds[4],tab,split,pops)
            w=load_weights(ds[4],Path(args.weights_dir)/f'{split}_weights.csv',split)
            return x,s,e,p,w
        tx,ts,te,tp,tw=prep(tr,'train')
        vx,vs,ve,vp,vw=prep(va,'validation')
        with torch.no_grad(): pred,_=model(vx,va[1],vs,ve,vp)
        popdf=group_table(va[4],vs,ve,vw,pred,va[1],va[2],'population')
        timedf=group_table(va[4],vs,ve,vw,pred,va[1],va[2],'time')
        popdf.to_csv(out/f'{variant}_population_calibration.csv',index=False)
        timedf.to_csv(out/f'{variant}_time_calibration.csv',index=False)
        # Recompute a single full-batch gradient at the saved checkpoint.
        # This measures the CURRENT weighted objective's gradient, not historical gradients.
        model.train();model.zero_grad(set_to_none=True)
        torch.manual_seed(42)
        pt,local=model(tx,tr[1],ts,te,tp)
        nll=-torch.log(pt[torch.arange(len(tx)),tr[2]])
        loss=(tw*nll).sum()
        reg=cfg['regularization']*(model.pop.square().mean()+model.log_global.square())
        if model.linear is not None: reg=reg+cfg['regularization']*model.linear.square()
        if model.nonlinear is not None: reg=reg+cfg['regularization']*model.nonlinear.square().mean()
        if model.net is not None: reg=reg+cfg['regularization']*local.square().mean()
        (loss+reg).backward()
        gradients={name: {'grad_norm':float(param.grad.norm()) if param.grad is not None else None,
                          'param_norm':float(param.detach().norm())}
                   for name,param in model.named_parameters()}
        model.eval()
        # Counterfactual: same sequence and population, shift the interval on the time axis.
        # Keep interval duration fixed, and restrict to training stage range.
        n=min(args.counterfactual_samples,len(vx)); ix=torch.arange(n)
        dur=(ve[ix]-vs[ix]).clamp(max=1000)
        cf={}
        with torch.no_grad():
            for stage in [1000.,4000.,8000.]:
                start=torch.full((n,),stage,dtype=torch.float64)
                pp,_=model(vx[ix],va[1][ix],start,start+dur,vp[ix])
                scores=1-pp[torch.arange(n),va[1][ix]]
                cf[str(int(stage))]=float(scores.mean())
        with torch.no_grad():
            rate_mult=torch.exp(model.pop.detach())
        summary={'variant':variant,'checkpoint_best_epoch':ck['metrics'].get('best_epoch'),
                 'train_weighted_nll_at_checkpoint':float(loss.detach()),
                 'train_regularization_at_checkpoint':float(reg.detach()),
                 'gradients':gradients,
                 'population_rate_multipliers':dict(zip(pops,[float(v) for v in rate_mult])),
                 'linear_stage_coefficient':None if model.linear is None else float(model.linear.detach()),
                 'nonlinear_stage_coefficients':None if model.nonlinear is None else model.nonlinear.detach().tolist(),
                 'time_counterfactual_mean_change_score':cf,
                 'mean_weighted_observed_validation':float(np.dot(vw.numpy(),(va[1]!=va[2]).numpy())),
                 'mean_weighted_predicted_validation':float(np.dot(vw.numpy(),(1-pred.detach().numpy()[np.arange(len(vx)),va[1].numpy()]))) }
        results[variant]=summary
        print(f'\n=== V3-{variant} ===',flush=True)
        print('Best epoch:',summary['checkpoint_best_epoch'])
        print('Weighted observed/predicted:',summary['mean_weighted_observed_validation'],summary['mean_weighted_predicted_validation'])
        print('Gradient norms:',{k:round(v['grad_norm'],9) if v['grad_norm'] is not None else None for k,v in gradients.items()})
        print('Population multipliers:',summary['population_rate_multipliers'])
        print('Counterfactual start times:',cf)
        print('Population calibration:\n',popdf.to_string(index=False))
        print('Time calibration:\n',timedf.to_string(index=False))
    (out/'diagnostic_summary.json').write_text(json.dumps(results,indent=2))
    print('\nSaved to:',out,flush=True)

if __name__=='__main__':main()
