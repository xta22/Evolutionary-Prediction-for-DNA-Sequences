#!/usr/bin/env python3
"""V3-C Stage 2.4: V3-B exact-kernel replay and simulator-specific oracle.
Evaluator-only. Fails closed if original V3-B observed sequences cannot be replayed.
No oracle information is supplied to forecasting-model training.
"""
import argparse, csv, json, hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np

SCENARIOS=('neutral','heterogeneous','context','regime_shift','linkage_only','recombination','selection_only','epistasis')

def rows(path):
    with open(path,newline='') as f:return list(csv.DictReader(f))

def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not data:return
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)

def initialize(seed,args):
    rng=np.random.default_rng(seed)
    n,L=args.individuals,args.sites
    pop=(rng.random((n,L))<args.initial_frequency).astype(np.uint8)
    pop[:,2*L//3]=pop[:,L//3]
    context=rng.lognormal(mean=0,sigma=args.context_sigma,size=L)
    context/=context.mean()
    return rng,pop,context

def step(pop,rng,context,scenario,t,args):
    n,L=pop.shape
    selection=scenario in ('selection_only','epistasis')
    if selection:
        a,b=L//3,2*L//3
        eps=args.epistasis_strength if scenario=='epistasis' else 0.
        logfitness=args.selection_strength*pop[:,a]+eps*pop[:,a]*pop[:,b]
        weights=np.exp(logfitness-logfitness.max());weights/=weights.sum()
    else:weights=None
    parents=rng.choice(n,size=n,replace=True,p=weights)
    child=pop[parents].copy()
    recomb=args.recombination_probability if scenario=='recombination' else 0.
    if recomb:
        do=rng.random(n)<recomb
        for j in np.flatnonzero(do):
            mate=rng.integers(n); breakpoint=int(rng.integers(1,L))
            child[j,breakpoint:]=pop[mate,breakpoint:]
    mult=args.heterogeneity_multiplier if scenario=='heterogeneous' else 1.
    rate=args.rate*mult
    if scenario=='regime_shift' and t>=args.shift_generation:rate*=args.shift_multiplier
    probs=rate*context if scenario=='context' else np.full(L,rate)
    if np.max(probs)>1:raise ValueError('Mutation probability > 1')
    flips=rng.random((n,L))<probs
    child ^= flips.astype(np.uint8)
    return child

def replay(seed,scenario,args,cutoffs,observed):
    """Replay original RNG call order, verify every observed sequence at every generation."""
    rng,pop,context=initialize(seed,args)
    checkpoints={}
    for t in range(args.generations+1):
        if t in cutoffs:checkpoints[t]=pop.copy()
        if t%args.observe_step==0 or t==args.generations:
            expected=observed.get(t,[])
            if len(expected)!=min(args.sample_size,args.individuals):
                raise RuntimeError(f'PARITY FAIL: sample count t={t}: {len(expected)}')
            for i,s in enumerate(expected):
                if s!=''.join(map(str,pop[i].tolist())):
                    raise RuntimeError(f'PARITY FAIL: {scenario} seed={seed} generation={t} sample={i}')
        if t<args.generations:pop=step(pop,rng,context,scenario,t,args)
    return checkpoints,context

def oracle_predict(pop,context,scenario,cutoff,target,args,seed,replicates):
    """Monte Carlo approximation of P(future allele=1 | full S_cutoff, true mechanism)."""
    L=pop.shape[1]
    total=np.zeros(L,dtype=float)
    for rep in range(replicates):
        rng=np.random.default_rng(np.random.SeedSequence([seed,cutoff,target,rep,8821]))
        current=pop.copy()
        for t in range(cutoff,target):
            current=step(current,rng,context,scenario,t,args)
        total+=current.mean(axis=0)
    return total/replicates

def latest_predict(hist,cutoff,alpha=5.):
    history=[r for r in hist if int(r['generation'])<=cutoff]
    times=sorted({int(r['generation']) for r in history})
    if not times:raise RuntimeError('No historical rows')
    latest=[r['sequence'] for r in history if int(r['generation'])==times[-1]]
    pooled=[r['sequence'] for r in history]
    arr=np.asarray([[int(c) for c in s] for s in latest],dtype=float)
    full=np.asarray([[int(c) for c in s] for s in pooled],dtype=float)
    gp=(full.sum()+alpha)/(full.size+2*alpha)
    return (arr.sum(axis=0)+alpha*gp)/(len(arr)+alpha)

def evaluate(p,y):
    p=np.clip(p,1e-8,1-1e-8)
    return {'nll':float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))),
            'brier':float(np.mean((y-p)**2)),
            'predicted_derived_count':float(p.sum()),
            'observed_derived_count':float(y.mean(axis=0).sum())}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--seeds',nargs='+',type=int,default=[42])
    ap.add_argument('--scenarios',nargs='+',default=list(SCENARIOS),choices=SCENARIOS)
    ap.add_argument('--replicates',type=int,default=200)
    ap.add_argument('--max-folds-per-run',type=int,default=3)
    a=ap.parse_args()
    if a.replicates<2:ap.error('--replicates must be >=2')
    root=Path(a.input);out=Path(a.output)
    config=json.loads((root/'config.json').read_text());sim=SimpleNamespace(**config)
    if sim.sites<8:raise RuntimeError('Unsupported V3-B configuration')
    results=[]; parity=[]
    for scenario in a.scenarios:
        for seed in a.seeds:
            run=root/scenario/f'seed_{seed}'
            folds=rows(run/'folds.csv')[:a.max_folds_per_run]
            obs_rows=rows(run/'observed_genomes_RESTRICTED.csv')
            observed={}
            for r in obs_rows:observed.setdefault(int(r['generation']),[]).append(r['sequence'])
            cutoffs={int(f['cutoff']) for f in folds}
            checkpoints,context=replay(seed,scenario,sim,cutoffs,observed)
            parity.append({'scenario':scenario,'seed':seed,'status':'EXACT_REPLAY_PASS','observations_verified':len(obs_rows)})
            hist=rows(run/'historical_inputs.csv')
            target_rows=rows(run/'evaluation_targets_RESTRICTED.csv')
            for fold in folds:
                fid=fold['fold_id'];cutoff=int(fold['cutoff']);target=int(fold['target_generation'])
                if target<=cutoff:raise RuntimeError('Invalid fold chronology')
                future=[r['sequence'] for r in target_rows if r['fold_id']==fid]
                y=np.asarray([[int(c) for c in s] for s in future],dtype=float)
                history=[r for r in hist if r['fold_id']==fid]
                if any(int(r['generation'])>cutoff for r in history):raise RuntimeError('History leakage')
                base=latest_predict(history,cutoff)
                oracle=oracle_predict(checkpoints[cutoff],context,scenario,cutoff,target,sim,seed,a.replicates)
                for model,p in [('B1_latest',base),('full_state_true_mechanism_oracle',oracle)]:
                    results.append({'scenario':scenario,'seed':seed,'fold_id':fid,'cutoff':cutoff,
                                    'target_generation':target,'model':model,'replicates':a.replicates,
                                    **evaluate(p,y)})
            print('PASS exact replay',scenario,seed,'folds',len(folds),flush=True)
    save(out/'oracle_metrics.csv',results);save(out/'parity_report.csv',parity)
    (out/'audit_protocol.json').write_text(json.dumps({
        'status':'EXPLORATORY_ORACLE_DIAGNOSTIC',
        'oracle_conditioning':'full population state at cutoff plus true V3-B mechanism',
        'oracle_type':'Monte Carlo marginal future allele probability',
        'limitations':['Not joint haplotype likelihood','Finite Monte Carlo approximation',
                       'Requires exact replay of original V3-B implementation and configuration',
                       'Seeds previously inspected are exploratory, not locked confirmatory test',
                       'No hidden parameters enter B1 model'],
        'input':str(root),'seeds':a.seeds,'scenarios':a.scenarios,'replicates':a.replicates},indent=2))
    print('Saved',out,'rows',len(results))

if __name__=='__main__':main()
