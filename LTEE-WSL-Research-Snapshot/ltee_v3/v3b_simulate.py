#!/usr/bin/env python3
"""V3-B forward Wright-Fisher mechanism validation (NumPy only).
Independent trajectories, explicit recombination, event counters and truth audits.
No inference from future samples; this is a simulator and evaluator, not a trained predictor.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np

SCENARIOS = ('neutral','heterogeneous','context','regime_shift','linkage_only','recombination','selection_only','epistasis')

def save_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def ld_r2(pop, a, b):
    x=pop[:,a].astype(float); y=pop[:,b].astype(float)
    vx=x.var(); vy=y.var()
    if vx < 1e-10 or vy < 1e-10: return None
    return float(((x*y).mean()-x.mean()*y.mean())**2/(vx*vy))

def haplotype_diversity(pop):
    _, cnt=np.unique(pop,axis=0,return_counts=True)
    freq=cnt/cnt.sum()
    return float(1-np.sum(freq**2))

def simulate(scenario, seed, args):
    rng=np.random.default_rng(seed)
    n=args.individuals; sites=args.sites; generations=args.generations
    if sites<8 or n<10 or generations<4: raise ValueError('sites>=8, individuals>=10, generations>=4 required')
    # Initial segregating variants permit measurable LD, selection and epistasis.
    # Generate paired backgrounds from same seed across scenarios.
    pop=(rng.random((n,sites))<args.initial_frequency).astype(np.uint8)
    a,b=sites//3, 2*sites//3
    pop[:,b]=pop[:,a]  # controlled initial coupling, not evidence of epistasis
    # Neutral comparison starts from identical initial population.
    context=rng.lognormal(mean=0,sigma=args.context_sigma,size=sites)
    context=context/context.mean()
    rate_mult=1.0
    if scenario=='heterogeneous': rate_mult=args.heterogeneity_multiplier
    recomb_prob=args.recombination_probability if scenario=='recombination' else 0.0
    selection=(scenario in ('selection_only','epistasis'))
    epsilon=args.epistasis_strength if scenario=='epistasis' else 0.0
    observations=[]; stats=[]; events=[]
    prev_mut=0
    for t in range(generations+1):
        if t%args.observe_step==0 or t==generations:
            for idx in range(min(args.sample_size,n)):
                observations.append({'generation':t,'sample_index':idx,'sequence':''.join(map(str,pop[idx].tolist()))})
        stats.append({'generation':t,'allele_a':float(pop[:,a].mean()),'allele_b':float(pop[:,b].mean()),
                      'double_mutant':float(np.mean((pop[:,a]==1)&(pop[:,b]==1))),
                      'haplotype_diversity':haplotype_diversity(pop),
                      'r2_ab':ld_r2(pop,a,b), 'cumulative_mutations':prev_mut})
        if t==generations: break
        if selection:
            logfitness=args.selection_strength*pop[:,a]+epsilon*pop[:,a]*pop[:,b]
            weights=np.exp(logfitness-logfitness.max()); weights/=weights.sum()
        else: weights=None
        parents=rng.choice(n,size=n,replace=True,p=weights)
        child=pop[parents].copy()
        crossovers=0
        if recomb_prob:
            do=rng.random(n)<recomb_prob
            for j in np.flatnonzero(do):
                mate=rng.integers(n); breakpoint=int(rng.integers(1,sites))
                child[j,breakpoint:]=pop[mate,breakpoint:]
                crossovers+=1
        rate=args.rate*rate_mult
        if scenario=='regime_shift' and t>=args.shift_generation: rate*=args.shift_multiplier
        probs=np.full(sites,rate)
        if scenario=='context': probs=rate*context
        if probs.max()>1: raise ValueError('mutation probability exceeds 1')
        flips=rng.random((n,sites))<probs
        mutation_count=int(flips.sum()); child ^= flips.astype(np.uint8)
        prev_mut+=mutation_count
        events.append({'generation':t+1,'mutation_events':mutation_count,'crossovers':crossovers,
                       'expected_mutations':float(n*probs.sum()),'rate_multiplier':float(rate/rate_mult/args.rate) if args.rate>0 else 0.0})
        pop=child
    return {'observations':observations,'stats':stats,'events':events,
            'truth':{'seed':seed,'scenario':scenario,'selected_site':a,'partner_site':b,
                     'selection_strength':args.selection_strength if selection else 0.0,
                     'epistasis_strength':epsilon,'recombination_probability':recomb_prob,
                     'context_multiplier':context.tolist() if scenario=='context' else None,
                     'shift_generation':args.shift_generation if scenario=='regime_shift' else None}}

def mechanism_summary(data,args):
    e=data['events']; s=data['stats']; mid=args.shift_generation
    pre=[r for r in e if r['generation']<=mid]
    post=[r for r in e if r['generation']>mid]
    mean=lambda v:float(np.mean(v)) if v else None
    return {'mean_pre_mutations':mean([x['mutation_events'] for x in pre]),
            'mean_post_mutations':mean([x['mutation_events'] for x in post]),
            'expected_pre_mutations':mean([x['expected_mutations'] for x in pre]),
            'expected_post_mutations':mean([x['expected_mutations'] for x in post]),
            'crossovers_total':sum(x['crossovers'] for x in e),
            'allele_a_start':s[0]['allele_a'],'allele_a_end':s[-1]['allele_a'],
            'double_mutant_start':s[0]['double_mutant'],'double_mutant_end':s[-1]['double_mutant'],
            'r2_start':s[0]['r2_ab'],'r2_end':s[-1]['r2_ab'],
            'haplotype_diversity_end':s[-1]['haplotype_diversity']}

def make_folds(observations,args):
    times=sorted({r['generation'] for r in observations})
    folds=[]; historical=[]; targets=[]
    for origin in times:
        target=origin+args.horizon
        if target not in times: continue
        fold=f'{origin}_to_{target}'
        hist=[r for r in observations if r['generation']<=origin]
        future=[r for r in observations if r['generation']==target]
        folds.append({'fold_id':fold,'cutoff':origin,'target_generation':target,
                      'historical_count':len(hist),'target_count':len(future)})
        for r in hist: historical.append({'fold_id':fold,**r})
        for r in future: targets.append({'fold_id':fold,**r})
    return folds,historical,targets

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',type=int,nargs='+',default=[11,23,42,67,101])
    p.add_argument('--individuals',type=int,default=240)
    p.add_argument('--sites',type=int,default=96)
    p.add_argument('--generations',type=int,default=100)
    p.add_argument('--observe-step',type=int,default=10)
    p.add_argument('--horizon',type=int,default=10)
    p.add_argument('--sample-size',type=int,default=8)
    p.add_argument('--rate',type=float,default=0.001)
    p.add_argument('--initial-frequency',type=float,default=0.25)
    p.add_argument('--context-sigma',type=float,default=1.0)
    p.add_argument('--heterogeneity-multiplier',type=float,default=3.0)
    p.add_argument('--shift-generation',type=int,default=50)
    p.add_argument('--shift-multiplier',type=float,default=4.0)
    p.add_argument('--recombination-probability',type=float,default=0.5)
    p.add_argument('--selection-strength',type=float,default=0.12)
    p.add_argument('--epistasis-strength',type=float,default=0.25)
    p.add_argument('--scenarios',nargs='+',choices=SCENARIOS,default=list(SCENARIOS))
    p.add_argument('--output',default='../LTEE_prospective/v3b_simulation')
    args=p.parse_args()
    if args.observe_step<1 or args.horizon<1 or args.sample_size<1: p.error('invalid sampling parameters')
    if args.shift_generation<0 or args.shift_generation>=args.generations: p.error('shift generation outside range')
    if not 0<=args.recombination_probability<=1: p.error('recombination probability outside [0,1]')
    if not 0<=args.initial_frequency<=1: p.error('initial frequency outside [0,1]')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(vars(args),indent=2))
    rows=[]
    for seed in args.seeds:
        for scenario in args.scenarios:
            data=simulate(scenario,seed,args)
            fold,hist,target=make_folds(data['observations'],args)
            run=out/scenario/f'seed_{seed}';run.mkdir(parents=True,exist_ok=True)
            save_csv(run/'observed_genomes_RESTRICTED.csv',data['observations'],['generation','sample_index','sequence'])
            save_csv(run/'population_statistics_RESTRICTED.csv',data['stats'],list(data['stats'][0]))
            save_csv(run/'mutation_events_RESTRICTED.csv',data['events'],list(data['events'][0]))
            save_csv(run/'folds.csv',fold,['fold_id','cutoff','target_generation','historical_count','target_count'])
            save_csv(run/'historical_inputs.csv',hist,['fold_id','generation','sample_index','sequence'])
            save_csv(run/'evaluation_targets_RESTRICTED.csv',target,['fold_id','generation','sample_index','sequence'])
            (run/'simulation_truth_RESTRICTED.json').write_text(json.dumps(data['truth'],indent=2))
            valid=all(r['generation']<=int(r['fold_id'].split('_')[0]) for r in hist) and all(r['generation']==int(r['fold_id'].split('_')[-1]) for r in target)
            summary=mechanism_summary(data,args)
            (run/'mechanism_diagnostics.json').write_text(json.dumps({'structural_pass':valid,'summary':summary},indent=2))
            rows.append({'scenario':scenario,'seed':seed,'folds':len(fold),'status':'PASS_STRUCTURAL_CHECKS_ONLY' if valid else 'FAIL',**summary})
            print(scenario,seed,'folds',len(fold),'structural',valid,flush=True)
    save_csv(out/'run_summary.csv',rows,list(rows[0]))
    print('Saved',out,'runs',len(rows))
    print('NOTE: Mechanism effects require replicated statistical comparisons; no test certification.')

if __name__=='__main__':main()
