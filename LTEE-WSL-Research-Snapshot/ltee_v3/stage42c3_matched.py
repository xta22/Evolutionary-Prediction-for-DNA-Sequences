#!/usr/bin/env python3
"""V3-C 4.2C.3: development-only matched M5 / full-state oracle pilot.

Train M5 on independent simulator trajectories, freeze coefficients, and compare
predictions on identical held-out trajectories/cutoffs/horizons/target clones.
Full-state oracle has privileged information; gaps are not model-only gaps.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

SCENARIOS = ('neutral', 'recombination', 'selection_only', 'epistasis')

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path.resolve()))
    if spec is None or spec.loader is None:
        raise ValueError('Cannot import '+str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def config(a):
    return SimpleNamespace(**{k: getattr(a,k) for k in (
        'individuals','sites','generations','observe_step','sample_size','rate',
        'initial_frequency','context_sigma','heterogeneity_multiplier',
        'shift_generation','shift_multiplier','recombination_probability',
        'selection_strength','epistasis_strength')}, horizon=10)

def matrix(records):
    return np.array([[int(c) for c in r['sequence']] for r in records], dtype=np.uint8)

def make_rows(reference, prod, a, scenarios, seeds, cutoffs, horizons):
    rows=[]
    for scenario in scenarios:
        for seed in seeds:
            sim=reference.simulate(scenario,seed,config(a))
            by_time={}
            for r in sim['observations']:
                by_time.setdefault(int(r['generation']),[]).append(r)
            for cutoff in cutoffs:
                times=sorted(t for t in by_time if t<=cutoff)
                if not times or times[-1]!=cutoff:
                    raise ValueError(f'No cutoff observation {scenario}/{seed}/{cutoff}')
                latest=matrix(by_time[cutoff])
                previous=matrix(by_time[times[-2]]) if len(times)>1 else None
                pairs=prod.pairs_for_length(a.sites,'adjacent',0)
                for horizon in horizons:
                    target=cutoff+horizon
                    if target not in by_time:
                        raise ValueError(f'Missing target {scenario}/{seed}/{target}')
                    future=matrix(by_time[target])
                    rows.append(dict(scenario=scenario,seed=seed,fold_id=f'{cutoff}_to_{target}',
                        cutoff=cutoff,target_generation=target,horizon=horizon,
                        history_interval=(cutoff-times[-2]) if previous is not None else None,
                        pairs=pairs,latest_matrix=latest,
                        latest=prod.frequencies(latest,pairs),
                        previous=prod.frequencies(previous,pairs) if previous is not None else None,
                        truth=prod.frequencies(future,pairs),n_target_clones=len(future)))
    return rows

def checkpoint(oracle, scenario, seed, cutoff, a):
    pop, context, rng=oracle.initialize(seed,config(a))
    for t in range(cutoff):
        pop=oracle.step(pop,context,rng,t,scenario,config(a))
    return pop,context

def oracle_prob(oracle, pop, context, scenario, a, cutoff, horizon, pairs, seeds):
    q=np.zeros((len(pairs),4),float)
    for seed in seeds:
        future=oracle.independent_future(pop,context,scenario,config(a),cutoff,horizon,int(seed))
        q+=oracle.pair_freq(future,pairs)
    return q/len(seeds)

def write_csv(path, rows):
    if not rows: raise ValueError('empty rows')
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run(a):
    if a.output.exists() and any(a.output.iterdir()):
        raise ValueError('Output must be absent or empty')
    if set(a.train_seeds)&set(a.eval_seeds): raise ValueError('Overlapping train/eval seeds')
    if not a.train_seeds or not a.eval_seeds: raise ValueError('Need train and eval seeds')
    if not a.horizons or not a.cutoffs or min(a.horizons)<=0 or min(a.cutoffs)<0:
        raise ValueError('Invalid horizons/cutoffs')
    if any(c%a.observe_step or c+h>a.generations for c in a.cutoffs for h in a.horizons):
        raise ValueError('Cutoffs must be observed and targets within simulation')
    if a.rollouts<2: raise ValueError('At least 2 rollouts')
    ref=load(a.reference,'reference_42c3')
    prod=load(a.source,'stage4_42c3')
    model=load(a.model_source,'stage41b_42c3')
    oracle=load(a.oracle_source,'oracle_42c3')
    train=make_rows(ref,prod,a,a.scenarios,a.train_seeds,a.cutoffs,a.horizons)
    eval_rows=make_rows(ref,prod,a,a.scenarios,a.eval_seeds,a.cutoffs,a.horizons)
    theta={}
    for kind in ('global','horizon','history'):
        theta[kind],info=model.fit(train,kind,prod,a.penalty)
        print('TRAIN',kind,'theta',theta[kind].tolist(),'iterations',info['iterations'],flush=True)
    output=[];convergence=[]
    # Never use the original future to construct oracle predictions.
    for scenario in a.scenarios:
      for seed in a.eval_seeds:
       for cutoff in a.cutoffs:
        pop,context=checkpoint(oracle,scenario,seed,cutoff,a)
        selected=[r for r in eval_rows if r['scenario']==scenario and r['seed']==seed and r['cutoff']==cutoff]
        if not selected: raise ValueError('No eval rows')
        # Check checkpoint exactly reproduces cutoff sampled genomes.
        if not np.array_equal(pop[:a.sample_size],selected[0]['latest_matrix']):
            raise RuntimeError('Cutoff checkpoint does not match reference observed genomes')
        for row in selected:
            horizon=row['horizon'];pairs=row['pairs'];truth=row['truth']
            # Independent future RNG streams disjoint from the reference trajectory.
            ss=np.random.SeedSequence([seed,cutoff,horizon,SCENARIOS.index(scenario),920231])
            seeds=[int(x.generate_state(1,dtype=np.uint64)[0]) for x in ss.spawn(a.rollouts)]
            # Compute predictions first; target truth is used only by the scorer below.
            predictions={
                'sampled_independent':prod.normalize(prod.independent(row['latest'])),
                'M5_fixed':model.predict(row,np.array([]),'fixed',prod),
                'M5_global':model.predict(row,theta['global'],'global',prod),
                'M5_horizon':model.predict(row,theta['horizon'],'horizon',prod),
                'M5_history':model.predict(row,theta['history'],'history',prod),
            }
            full=oracle.pair_freq(pop,pairs)
            predictions['full_state_independent']=prod.normalize(prod.independent(full))
            accumulator=np.zeros((len(pairs),4),float)
            oracle_estimates={}
            for k,roll_seed in enumerate(seeds,1):
                future=oracle.independent_future(pop,context,scenario,config(a),cutoff,horizon,roll_seed)
                accumulator+=oracle.pair_freq(future,pairs)
                if k in (a.rollouts//2,a.rollouts):
                    oracle_estimates[k]=accumulator/k
            # Jeffreys pseudocount prevents zero-probability artifacts at finite R.
            oracle_counts=oracle_estimates[a.rollouts]*(a.rollouts*a.individuals)
            predictions['full_state_oracle']=(oracle_counts+0.5)/(a.rollouts*a.individuals+2.0)
            convergence.append(dict(scenario=scenario,seed=seed,cutoff=cutoff,horizon=horizon,
                mean_abs_mc_change=float(np.abs(oracle_estimates[a.rollouts]-oracle_estimates[a.rollouts//2]).mean()),
                max_abs_mc_change=float(np.abs(oracle_estimates[a.rollouts]-oracle_estimates[a.rollouts//2]).max())))
            for name,q in predictions.items():
                scores=prod.score(q,truth)
                output.append(dict(scenario=scenario,seed=seed,cutoff=cutoff,horizon=horizon,
                    fold_id=row['fold_id'],model=name,n_pairs=len(pairs),n_target_clones=row['n_target_clones'],**scores))
            print('MATCHED',scenario,seed,cutoff,horizon,flush=True)
    a.output.mkdir(parents=True,exist_ok=True)
    write_csv(a.output/'matched_metrics.csv',output)
    write_csv(a.output/'mc_convergence.csv',convergence)
    hashes={k:hashlib.sha256(getattr(a,k).read_bytes()).hexdigest() for k in ('reference','source','model_source','oracle_source')}
    report=dict(status='DEVELOPMENT_MATCHED_NOT_LOCKED',scenarios=a.scenarios,
        train_seeds=a.train_seeds,eval_seeds=a.eval_seeds,cutoffs=a.cutoffs,horizons=a.horizons,
        rollouts=a.rollouts,penalty=a.penalty,n_train_folds=len(train),n_eval_folds=len(eval_rows),
        n_rows=len(output),source_sha256=hashes,
        limitations=['Full-state oracle has privileged cutoff population and mechanism information.',
            'Same future targets for every model within a fold; no future used in predictions.',
            'Evaluation seeds are development-only, not locked test.',
            'Training penalty fixed before evaluation; no tuning on evaluation trajectories.',
            'Oracle MC uses future independent rollouts, not original realized future.',
            'Original simulator sampled first individuals, not random individuals.',
            'Overlapping folds are correlated; no fold-level significance claims.',
            'All model NLLs use production stage4.score and same pair definition.'])
    (a.output/'report.json').write_text(json.dumps(report,indent=2))
    summary={}
    for name in sorted({r['model'] for r in output}):
        vals=[r['joint_nll'] for r in output if r['model']==name]
        summary[name]=float(np.mean(vals))
    print('=== MATCHED JOINT NLL ===')
    for k,v in summary.items():print(k,round(v,6))
    print('SAVED',a.output,'status',report['status'])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('reference','source','model-source','oracle-source','output'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--scenarios',nargs='+',choices=SCENARIOS,default=list(SCENARIOS))
    p.add_argument('--train-seeds',nargs='+',type=int,default=[311,313,317,331])
    p.add_argument('--eval-seeds',nargs='+',type=int,default=[211,223])
    p.add_argument('--cutoffs',nargs='+',type=int,default=[20,50])
    p.add_argument('--horizons',nargs='+',type=int,default=[10,20,30])
    p.add_argument('--rollouts',type=int,default=100)
    p.add_argument('--penalty',type=float,default=0.01)
    for name,typ,default in [('individuals',int,120),('sites',int,24),('generations',int,100),
        ('observe-step',int,10),('sample-size',int,8),('rate',float,.001),
        ('initial-frequency',float,.25),('context-sigma',float,1.),
        ('heterogeneity-multiplier',float,3.),('shift-generation',int,50),
        ('shift-multiplier',float,4.),('recombination-probability',float,.5),
        ('selection-strength',float,.12),('epistasis-strength',float,.25)]:
        p.add_argument('--'+name,type=typ,default=default)
    a=p.parse_args()
    if a.penalty<0: p.error('negative penalty')
    if a.sample_size>a.individuals:p.error('sample size exceeds population')
    run(a)

if __name__=='__main__': main()
