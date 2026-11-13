#!/usr/bin/env python3
"""V3: population-aware time-varying CTMC on frozen Evo2 embeddings.

A: population-specific constant rates; B: + shared linear stage;
C: + shared low-dimensional nonlinear stage. Uses train/validation only.
Rates are effective sequence-state change proxies, not biochemical rates.
"""
import argparse, copy, json, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from train_temporal_head_v2 import load_split

BASE_RATE = 2.118290e-9  # train-only branch-level pooled rate, per bp/generation


def load_weights(metadata, path, split):
    d = pd.read_csv(path)
    required={'sample_id','branch_id','split','is_mutated','sampling_weight'}
    if not required.issubset(d.columns) or d.sample_id.duplicated().any():
        raise ValueError(f'Invalid weights: {path}')
    d=d.set_index('sample_id')
    vals=[]
    for m in metadata:
        r=d.loc[m['sample_id']]
        if str(r.branch_id)!=str(m['branch_id']) or str(r.split)!=split or int(r.is_mutated)!=int(m['is_mutated']):
            raise ValueError('Weight metadata mismatch: '+m['sample_id'])
        vals.append(float(r.sampling_weight))
    if len(d)!=len(vals): raise ValueError('Weight count mismatch')
    w=torch.tensor(vals,dtype=torch.float64)
    if not torch.isfinite(w).all() or (w<=0).any(): raise ValueError('Invalid weights')
    return w/w.sum()


def load_times(metadata, table, split, populations):
    start=[];end=[];pop=[]
    for m in metadata:
        b=str(m['branch_id'])
        if b not in table.index: raise ValueError('Unknown branch: '+b)
        r=table.loc[b]
        if r['split']!=split or r['population']!=m['population']:
            raise ValueError('Branch split/population mismatch: '+b)
        a=float(r['parent_time']);z=float(r['child_time']);dt=float(m['delta_time'])
        if not np.isclose(z-a,dt,atol=1e-3,rtol=1e-6):
            raise ValueError('Delta-time mismatch: '+b)
        if a<0 or z<=a: raise ValueError('Invalid times: '+b)
        start.append(a);end.append(z);pop.append(populations.index(m['population']))
    return torch.tensor(start,dtype=torch.float64),torch.tensor(end,dtype=torch.float64),torch.tensor(pop,dtype=torch.long)


class StageCTMC(nn.Module):
    def __init__(self, variant, n_pop, hidden=32, dropout=0.5, use_context=True, max_stage=10000.0):
        super().__init__()
        self.variant=variant; self.use_context=use_context;self.max_stage=max_stage
        self.pop=nn.Parameter(torch.zeros(n_pop,dtype=torch.float64))
        self.linear=nn.Parameter(torch.zeros((),dtype=torch.float64)) if variant in ('B','C') else None
        self.nonlinear=nn.Parameter(torch.zeros(2,dtype=torch.float64)) if variant=='C' else None
        self.log_global=nn.Parameter(torch.tensor(0.,dtype=torch.float64))
        # Relative substitution spectrum; normalize mean outgoing rate to 1.
        self.raw_q=nn.Parameter(torch.zeros((4,4),dtype=torch.float64))
        self.register_buffer('mask',1-torch.eye(4,dtype=torch.float64))
        self.net=(nn.Sequential(nn.Linear(4096+4,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,1)) if use_context else None)
        if self.net is not None:
            nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        # 8-point Gauss-Legendre integration for exp(stage function).
        from numpy.polynomial.legendre import leggauss
        x,w=leggauss(8)
        self.register_buffer('nodes',torch.tensor(x,dtype=torch.float64))
        self.register_buffer('quad_weights',torch.tensor(w,dtype=torch.float64))

    def stage_log(self,t):
        z=(t/self.max_stage).clamp(0,1)
        v=torch.zeros_like(z)
        if self.linear is not None: v=v+self.linear*z
        if self.nonlinear is not None:
            v=v+self.nonlinear[0]*F.relu(z-1/3).square()+self.nonlinear[1]*F.relu(z-2/3).square()
        return v

    def forward(self,x,parent,start,end,pop):
        # x float32; all rates/transition matrices float64 to preserve rare probabilities.
        if self.net is None: log_local=torch.zeros(len(parent),device=x.device,dtype=torch.float64)
        else:
            feat=torch.cat((x,F.one_hot(parent,4).float()),dim=1)
            log_local=self.net(feat).squeeze(-1).double().clamp(-3,3)
        midpoint=(start[:,None]+end[:,None])/2
        half=(end-start)[:,None]/2
        t=midpoint+half*self.nodes[None,:]
        stage_integral=half.squeeze(1)*(self.quad_weights[None,:]*torch.exp(self.stage_log(t))).sum(1)
        log_rate=(self.log_global.clamp(-12,12)+self.pop[pop].clamp(-8,8)+log_local).clamp(-15,15)
        exposure=BASE_RATE*torch.exp(log_rate)*stage_integral
        off=F.softplus(self.raw_q)*self.mask
        off=off/(off.sum(1).mean().clamp_min(1e-12))
        Q=off-torch.diag(off.sum(1))
        transition=torch.matrix_exp(exposure[:,None,None]*Q)
        probs=transition[torch.arange(len(parent),device=x.device),parent]
        # No renormalization: matrix exponential is stochastic to numerical precision.
        return probs.clamp_min(1e-300),log_local


def summarize(probs,parent,future,weights):
    p=probs.detach().cpu().numpy(); a=parent.numpy();b=future.numpy();w=weights.numpy()
    ix=np.arange(len(a)); changed=(a!=b).astype(int)
    scores=1-p[ix,a];nll=-np.log(np.clip(p[ix,b],1e-300,1))
    return dict(n=len(a),sampled_change_fraction=float(changed.mean()),weighted_change_fraction=float(np.dot(w,changed)),
        sampled_log_loss=float(nll.mean()),weighted_log_loss=float(np.dot(w,nll)),
        sampled_mutation_auroc=float(roc_auc_score(changed,scores)),
        sampled_mutation_auprc=float(average_precision_score(changed,scores)),
        weighted_mutation_auroc=float(roc_auc_score(changed,scores,sample_weight=w)),
        weighted_mutation_auprc=float(average_precision_score(changed,scores,sample_weight=w)),
        predicted_change_weighted=float(np.dot(w,scores)),predicted_change_sampled=float(scores.mean()),
        weighted_brier=float(np.dot(w,(scores-changed)**2)),
        nucleotide_accuracy_sampled=float((p.argmax(1)==b).mean()))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--embedding-dir',default='results/temporal_head_v1_embeddings')
    ap.add_argument('--weights-dir',default='LTEE_temporal_head_v1')
    ap.add_argument('--metadata',default='LTEE_temporal_head_v3/branch_time_metadata.csv')
    ap.add_argument('--output-dir',default='results/temporal_head_v3_training')
    ap.add_argument('--variants',nargs='+',choices=['A','B','C'],default=['A','B','C'])
    ap.add_argument('--epochs',type=int,default=120)
    ap.add_argument('--patience',type=int,default=20)
    ap.add_argument('--lr',type=float,default=0.005)
    ap.add_argument('--hidden',type=int,default=32)
    ap.add_argument('--dropout',type=float,default=0.5)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--context',choices=['on','off'],default='on')
    ap.add_argument('--regularization',type=float,default=0.001)
    args=ap.parse_args()
    torch.set_num_threads(4);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    tab=pd.read_csv(args.metadata)
    required={'branch_id','split','population','parent_time','child_time','delta_time'}
    if not required.issubset(tab.columns) or tab.branch_id.duplicated().any(): raise ValueError('Invalid branch time metadata')
    tab=tab.set_index('branch_id')
    pops=sorted(tab.loc[tab['split']=='train','population'].unique())
    tr=load_split(Path(args.embedding_dir),'train');va=load_split(Path(args.embedding_dir),'validation')
    if set(m['sample_id'] for m in tr[4]) & set(m['sample_id'] for m in va[4]): raise ValueError('Train/validation overlap')
    if not set(m['population'] for m in va[4]).issubset(pops): raise ValueError('Unseen validation population; needs new-population policy')
    tw=load_weights(tr[4],Path(args.weights_dir)/'train_weights.csv','train')
    vw=load_weights(va[4],Path(args.weights_dir)/'validation_weights.csv','validation')
    ts,te,tp=load_times(tr[4],tab,'train',pops);vs,ve,vp=load_times(va[4],tab,'validation',pops)
    mu=tr[0].mean(0);sd=tr[0].std(0).clamp_min(1e-4)
    tx=((tr[0]-mu)/sd).clamp(-10,10);vx=((va[0]-mu)/sd).clamp(-10,10)
    print('Train:',len(tx),'Validation:',len(vx),'Populations:',len(pops),flush=True)
    print('Train time range:',float(ts.min()),float(te.max()),'Validation:',float(vs.min()),float(ve.max()),flush=True)
    print('Using weighted genome-position proxy objective; test data untouched.',flush=True)
    print('WARNING: time stage is clipped above 10000 generations (plateau extrapolation).',flush=True)
    print('WARNING: branch counts/weights approximate eligible genome positions; not validated biological rates.',flush=True)
    results={}
    for variant in args.variants:
        torch.manual_seed(args.seed)
        model=StageCTMC(variant,len(pops),args.hidden,args.dropout,args.context=='on').to('cpu')
        optim=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.0)
        best=float('inf');best_epoch=0;best_state=None;history=[]
        print('\n=== V3-'+variant+' ===',flush=True)
        for epoch in range(1,args.epochs+1):
            model.train();optim.zero_grad(set_to_none=True)
            prob,local=model(tx,tr[1],ts,te,tp)
            nll=-torch.log(prob[torch.arange(len(tx)),tr[2]])
            loss=(tw*nll).sum()
            reg=args.regularization*(model.pop.square().mean()+model.log_global.square())
            if model.linear is not None:reg=reg+args.regularization*model.linear.square()
            if model.nonlinear is not None:reg=reg+args.regularization*model.nonlinear.square().mean()
            if model.net is not None:reg=reg+args.regularization*local.square().mean()
            (loss+reg).backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);optim.step()
            model.eval()
            with torch.inference_mode():
                pv,_=model(vx,va[1],vs,ve,vp)
                v_nll=-torch.log(pv[torch.arange(len(vx)),va[2]])
                weighted=float((vw*v_nll).sum());sampled=float(v_nll.mean())
            history.append({'epoch':epoch,'train_weighted_nll':float(loss.detach()),'val_weighted_nll':weighted,'val_sampled_nll':sampled})
            if epoch==1 or epoch%10==0:print(f'Epoch {epoch:03d} train_weighted={float(loss):.7g} val_weighted={weighted:.7g} val_sampled={sampled:.7g}',flush=True)
            if weighted<best-1e-9:best=weighted;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
            elif epoch-best_epoch>=args.patience:print('Early stopping',flush=True);break
        model.load_state_dict(best_state);model.eval()
        with torch.inference_mode():pv,_=model(vx,va[1],vs,ve,vp)
        result=summarize(pv,va[1],va[2],vw)
        result.update(variant=variant,best_epoch=best_epoch,best_validation_weighted_loss=best,
            population_log_effects={p:float(v) for p,v in zip(pops,model.pop.detach().tolist())},
            linear_stage_coefficient=None if model.linear is None else float(model.linear.detach()),
            nonlinear_coefficients=None if model.nonlinear is None else model.nonlinear.detach().tolist())
        results[variant]=result
        print('Validation:',json.dumps(result,indent=2),flush=True)
        torch.save({'state_dict':best_state,'variant':variant,'populations':pops,'embedding_mean':mu,'embedding_std':sd,
                    'config':vars(args),'metrics':result,'base_rate':BASE_RATE},out/f'v3_{variant}_best.pt')
        (out/f'v3_{variant}_history.json').write_text(json.dumps(history,indent=2))
    (out/'validation_comparison.json').write_text(json.dumps(results,indent=2))
    print('\n=== V3 VALIDATION COMPARISON ===',flush=True)
    print(json.dumps(results,indent=2),flush=True)
    print('Saved to:',out,flush=True)

if __name__=='__main__':main()
