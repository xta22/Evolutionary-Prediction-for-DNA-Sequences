#!/usr/bin/env python3
"""V3-A causal forward-time haplotype simulation. NumPy only; Python >=3.9.
All mutation/selection happens forward in time. No phylogenetic reconstruction.
"""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

SCENARIOS = ('neutral','heterogeneous','context','regime_shift','linkage_only','selection_only','epistasis')
DNA = np.array(list('ACGT'))

def write_csv(path, rows, columns):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns); w.writeheader(); w.writerows(rows)

def context_rates(founder):
    # Observable 3-mer rule, normalized across sites; no future labels.
    left=np.roll(founder,1); right=np.roll(founder,-1)
    raw=np.exp(0.7*(left==2)+0.5*(right==1)-0.35*(founder==0))
    return raw/raw.mean()

def fitness(haps,scenario,selection_site_a,selection_site_b):
    if scenario not in ('selection_only','epistasis'):
        return np.ones(len(haps))
    a=(haps[:,selection_site_a]==1).astype(float)
    b=(haps[:,selection_site_b]==2).astype(float)
    logw=0.35*a
    if scenario=='epistasis':
        logw=logw+0.15*b+0.9*a*b
    return np.exp(logw-np.max(logw))

def simulate_population(rng, founder, scenario, pop, n_individuals, generations, rate, observe_at):
    L=len(founder)
    haps=np.tile(founder,(n_individuals,1)).copy()
    context=context_rates(founder)
    rate_multiplier=float(np.exp(rng.normal(0,0.65))) if scenario=='heterogeneous' else 1.0
    a,b=L//3,2*L//3
    records=[]; counts=[]; truths=[]
    def observe(t):
        # Draw from population, not inferred ancestor-descendant pairs.
        indices=rng.choice(n_individuals,size=min(8,n_individuals),replace=False)
        for j,idx in enumerate(indices):
            records.append({'population':pop,'generation':t,'clone_id':f'{pop}_g{t}_c{j}',
                            'sequence':''.join(DNA[haps[idx]]),'scenario':scenario})
        counts.append({'population':pop,'generation':t,'mutant_carrier_fraction':float(np.mean(np.any(haps!=founder,axis=1))),
                       'mean_differences_from_founder':float(np.mean(np.sum(haps!=founder,axis=1)))})
    observe(0)
    for t in range(1,generations+1):
        # Wright-Fisher reproduction with whole-haplotype inheritance: linkage, no recombination.
        w=fitness(haps,scenario,a,b)
        parents=rng.choice(n_individuals,size=n_individuals,replace=True,p=w/w.sum())
        haps=haps[parents].copy()
        mut_rate=rate*rate_multiplier
        if scenario=='regime_shift' and t>generations//2:
            mut_rate*=5.0
        site_weights=context if scenario=='context' else np.ones(L)
        events=rng.random((n_individuals,L))<np.minimum(mut_rate*site_weights,0.95)
        rows,cols=np.nonzero(events)
        if len(rows):
            shifts=rng.integers(1,4,size=len(rows))
            haps[rows,cols]=(haps[rows,cols]+shifts)%4
        if t in observe_at:
            observe(t)
    truths.append({'population':pop,'rate_multiplier':rate_multiplier,'regime_shift_generation':generations//2 if scenario=='regime_shift' else None,
                   'selection_site_a':a if scenario in ('selection_only','epistasis') else None,
                   'selection_site_b':b if scenario=='epistasis' else None,
                   'epistasis_log_fitness':0.9 if scenario=='epistasis' else 0.0,
                   'founder_sequence':''.join(DNA[founder])})
    return records,counts,truths

def build_folds(observations,cutoffs,horizon):
    by_pop={}
    for r in observations:
        by_pop.setdefault(r['population'],[]).append(r)
    folds=[];hist=[];targets=[]
    for pop,rs in sorted(by_pop.items()):
        for cutoff in cutoffs:
            target_t=cutoff+horizon
            past=[r for r in rs if r['generation']<=cutoff]
            future=[r for r in rs if r['generation']==target_t]
            if not past or not future:continue
            fold_id=f'{pop}_T{cutoff}_H{horizon}'
            folds.append({'fold_id':fold_id,'population':pop,'cutoff':cutoff,'target_generation':target_t,
                          'horizon':horizon,'historical_clone_count':len(past),'target_clone_count':len(future)})
            for r in past:hist.append({'fold_id':fold_id,'clone_id':r['clone_id'],'generation':r['generation']})
            for r in future:targets.append({'fold_id':fold_id,'clone_id':r['clone_id'],'generation':r['generation']})
    return folds,hist,targets

def audit(observations,folds,hist,targets):
    lookup={r['clone_id']:r for r in observations}
    assert len(lookup)==len(observations),'Duplicate clone IDs'
    assert len({len(r['sequence']) for r in observations})==1,'Inconsistent sequence lengths'
    assert all(set(r['sequence'])<=set('ACGT') for r in observations)
    fs={r['fold_id']:r for r in folds}
    for r in hist:
        assert r['generation']<=fs[r['fold_id']]['cutoff']
        assert lookup[r['clone_id']]['generation']==r['generation']
    for r in targets:
        assert r['generation']==fs[r['fold_id']]['target_generation']
        assert r['generation']>fs[r['fold_id']]['cutoff']
    h={(r['fold_id'],r['clone_id']) for r in hist}
    assert not any((r['fold_id'],r['clone_id']) in h for r in targets)
    return {'status':'PASS_STRUCTURAL_CHECKS_ONLY','fold_count':len(folds),'historical_links':len(hist),
            'target_links':len(targets),'warning':'Synthetic scenarios validate code paths, not biological realism.'}

def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    assert args.generations>=4 and args.sites>=6 and args.individuals>=8
    assert 0<args.rate<0.1 and args.horizon>0
    observe_at=sorted(set([0]+list(range(args.observe_step,args.generations+1,args.observe_step))+[args.generations]))
    cutoffs=[t for t in observe_at if t+args.horizon in observe_at and t>=args.observe_step]
    if not cutoffs:raise ValueError('No compatible forecast cutoffs; choose horizon multiple of observe-step')
    summary=[]
    for scenario in args.scenarios:
        for seed in args.seeds:
            rng=np.random.default_rng(seed)
            founder=rng.integers(0,4,size=args.sites,dtype=np.uint8)
            obs=[];stats=[];truth=[]
            for p in range(args.populations):
                pop=f'P{p:02d}'
                o,s,t=simulate_population(rng,founder,scenario,pop,args.individuals,args.generations,args.rate,observe_at)
                obs.extend(o);stats.extend(s);truth.extend(t)
            folds,hist,targets=build_folds(obs,cutoffs,args.horizon)
            result=audit(obs,folds,hist,targets)
            folder=out/scenario/f'seed_{seed}'
            folder.mkdir(parents=True,exist_ok=True)
            write_csv(folder/'observed_genomes.csv',obs,['population','generation','clone_id','sequence','scenario'])
            write_csv(folder/'population_statistics.csv',stats,['population','generation','mutant_carrier_fraction','mean_differences_from_founder'])
            write_csv(folder/'folds.csv',folds,['fold_id','population','cutoff','target_generation','horizon','historical_clone_count','target_clone_count'])
            write_csv(folder/'historical_inputs.csv',hist,['fold_id','clone_id','generation'])
            # Separate evaluator-only targets; no future sequences embedded in historical_inputs.
            write_csv(folder/'evaluation_targets_RESTRICTED.csv',targets,['fold_id','clone_id','generation'])
            (folder/'simulation_truth_RESTRICTED.json').write_text(json.dumps(truth,indent=2))
            (folder/'integrity_report.json').write_text(json.dumps(result,indent=2))
            summary.append({'scenario':scenario,'seed':seed,'observed_clones':len(obs),'folds':len(folds),'status':result['status']})
            print(scenario,seed,'clones',len(obs),'folds',len(folds),flush=True)
    write_csv(out/'run_summary.csv',summary,['scenario','seed','observed_clones','folds','status'])
    (out/'config.json').write_text(json.dumps(vars(args),indent=2))
    print('Saved',out)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='LTEE_prospective/v3a_simulation')
    p.add_argument('--seeds',type=int,nargs='+',default=[11,23,42])
    p.add_argument('--scenarios',nargs='+',choices=SCENARIOS,default=list(SCENARIOS))
    p.add_argument('--populations',type=int,default=6)
    p.add_argument('--individuals',type=int,default=120)
    p.add_argument('--sites',type=int,default=96)
    p.add_argument('--generations',type=int,default=80)
    p.add_argument('--observe-step',type=int,default=10)
    p.add_argument('--horizon',type=int,default=10)
    p.add_argument('--rate',type=float,default=0.002)
    run(p.parse_args())
if __name__=='__main__':main()
