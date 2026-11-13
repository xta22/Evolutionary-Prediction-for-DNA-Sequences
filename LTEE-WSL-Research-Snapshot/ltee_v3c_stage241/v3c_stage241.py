#!/usr/bin/env python3
"""Exploratory V3-B 2x2 state/mechanism information intervention.

True mechanism is evaluator-only. Estimated mechanism is a deliberately simple
historical flip-rate proxy (neutral reproduction, no selection or recombination).
Do NOT interpret contrasts as a unique causal decomposition or a Bayes bound.
"""
import argparse, csv, json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

HERE=Path(__file__).resolve().parent

def load_kernel(path):
    import importlib.util
    spec=importlib.util.spec_from_file_location('v3c_stage24_kernel',str(path))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def csvrows(path):
    with open(path,newline='') as f:return list(csv.DictReader(f))

def writecsv(path,rs):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rs:return
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)

def matrix(seqs):
    if not seqs:raise ValueError('Empty sequence sample')
    arr=np.asarray([[int(c) for c in s] for s in seqs],dtype=np.uint8)
    if arr.ndim!=2 or not np.isin(arr,[0,1]).all():raise ValueError('Invalid binary haplotypes')
    return arr

def historical_rate_proxy(hist,cutoff,default_rate,sites):
    """Estimate an effective symmetric flip probability using successive sampled frequencies.
    Not a consistent estimate of mutation rate in presence of drift/selection.
    """
    groups={}
    for r in hist:
        t=int(r['generation'])
        if t>cutoff:raise RuntimeError('Future history row')
        groups.setdefault(t,[]).append(r['sequence'])
    ts=sorted(groups)
    if len(ts)<2:return float(default_rate),'FALLBACK_ONE_TIMEPOINT'
    rates=[]
    for t0,t1 in zip(ts[:-1],ts[1:]):
        a=matrix(groups[t0]).mean(0);b=matrix(groups[t1]).mean(0)
        # Temporal frequency movement is dominated by drift in many settings;
        # this is a predictive proxy, not a biological rate estimator.
        rates.append(float(np.mean(np.abs(b-a))/max(t1-t0,1)))
    return float(np.clip(np.median(rates),1e-7,0.2)),'HISTORICAL_FREQUENCY_CHANGE_PROXY'

def sampled_population(hist,cutoff,n,seed):
    latest=max(int(r['generation']) for r in hist if int(r['generation'])<=cutoff)
    seqs=[r['sequence'] for r in hist if int(r['generation'])==latest]
    sample=matrix(seqs)
    rng=np.random.default_rng(seed)
    return sample[rng.integers(len(sample),size=n)].copy()

def proxy_step(pop,rng,rate):
    n,L=pop.shape
    child=pop[rng.integers(n,size=n)].copy()
    child ^= (rng.random((n,L))<rate).astype(np.uint8)
    return child

def forecast(kernel,pop,context,scenario,cutoff,target,args,seed,reps,mechanism,proxy_rate):
    L=pop.shape[1];s=np.zeros(L)
    for rep in range(reps):
        rng=np.random.default_rng(np.random.SeedSequence([seed,cutoff,target,rep,19171,0 if mechanism=='true' else 1]))
        cur=pop.copy()
        for t in range(cutoff,target):
            cur=kernel.step(cur,rng,context,scenario,t,args) if mechanism=='true' else proxy_step(cur,rng,proxy_rate)
        s+=cur.mean(0)
    return s/reps

def classify(cutoff,target,shift):
    if target<=shift:return 'pre_shift'
    if cutoff>=shift:return 'post_shift'
    return 'crossing_shift'

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='v3c_stage24.py')
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[42])
    p.add_argument('--scenarios',nargs='+',default=['neutral','regime_shift','selection_only','epistasis'])
    p.add_argument('--replicates',type=int,default=100)
    p.add_argument('--max-folds-per-run',type=int,default=0,help='0 means all folds; needed to include shift boundaries')
    a=p.parse_args(argv)
    if a.replicates<2:p.error('Need >=2 replicates')
    root=Path(a.input);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    kernel=load_kernel(Path(a.source).resolve())
    sim=SimpleNamespace(**json.loads((root/'config.json').read_text()))
    results=[];parity=[];schedule=[]
    for scenario in a.scenarios:
        for seed in a.seeds:
            run=root/scenario/f'seed_{seed}'
            folds=csvrows(run/'folds.csv')
            if a.max_folds_per_run:folds=folds[:a.max_folds_per_run]
            observed={}
            obs=csvrows(run/'observed_genomes_RESTRICTED.csv')
            for r in obs:observed.setdefault(int(r['generation']),[]).append(r['sequence'])
            checkpoints,context=kernel.replay(seed,scenario,sim,{int(f['cutoff']) for f in folds},observed)
            parity.append({'scenario':scenario,'seed':seed,'status':'EXACT_REPLAY_PASS','observed_rows':len(obs)})
            histories=csvrows(run/'historical_inputs.csv')
            targets=csvrows(run/'evaluation_targets_RESTRICTED.csv')
            for fold in folds:
                fid=fold['fold_id'];c=int(fold['cutoff']);t=int(fold['target_generation'])
                if t<=c:raise RuntimeError('Invalid chronology')
                hist=[r for r in histories if r['fold_id']==fid]
                if not hist or any(int(r['generation'])>c for r in hist):raise RuntimeError('History leakage')
                future=[r['sequence'] for r in targets if r['fold_id']==fid]
                y=matrix(future).astype(float)
                proxy_rate,method=historical_rate_proxy(hist,c,sim.rate,sim.sites)
                sampled=sampled_population(hist,c,sim.individuals,seed+c*13+t)
                full=checkpoints[c]
                period=classify(c,t,sim.shift_generation)
                # Time-resolved schedule audit, using actual step() convention t>=shift_generation.
                n_post=sum(1 for generation in range(c,t) if generation>=sim.shift_generation)
                schedule.append({'scenario':scenario,'seed':seed,'fold_id':fid,'cutoff':c,'target':t,
                                 'period':period,'n_pre_steps':t-c-n_post,'n_post_steps':n_post,
                                 'true_pre_rate':sim.rate,'true_post_rate':sim.rate*sim.shift_multiplier if scenario=='regime_shift' else sim.rate})
                b1=kernel.latest_predict(hist,c)
                cells=[('sampled','estimated',sampled,'estimated'),('sampled','true',sampled,'true'),
                       ('full','estimated',full,'estimated'),('full','true',full,'true')]
                for state_label,mechanism_label,pop,mode in cells:
                    pred=forecast(kernel,pop,context,scenario,c,t,sim,seed,a.replicates,mode,proxy_rate)
                    ev=kernel.evaluate(pred,y)
                    results.append({'scenario':scenario,'seed':seed,'fold_id':fid,'cutoff':c,'target_generation':t,
                                    'period':period,'state_information':state_label,'mechanism_information':mechanism_label,
                                    'model':f'{state_label}_{mechanism_label}','rate_proxy':proxy_rate,
                                    'rate_proxy_method':method,'replicates':a.replicates,**ev})
                results.append({'scenario':scenario,'seed':seed,'fold_id':fid,'cutoff':c,'target_generation':t,
                                'period':period,'state_information':'sampled','mechanism_information':'none',
                                'model':'B1_latest','rate_proxy':proxy_rate,'rate_proxy_method':method,
                                'replicates':0,**kernel.evaluate(b1,y)})
            print('PASS exact replay',scenario,seed,'folds',len(folds),flush=True)
    writecsv(out/'information_cells.csv',results)
    writecsv(out/'replay_parity.csv',parity)
    writecsv(out/'regime_boundary_audit.csv',schedule)
    # Contrasts are descriptive, non-causal; the estimated model is intentionally misspecified.
    idx={}
    for r in results:idx.setdefault((r['scenario'],r['seed'],r['fold_id']),{})[r['model']]=r['nll']
    contrasts=[]
    for (scenario,seed,fid),v in idx.items():
        contrasts.append({'scenario':scenario,'seed':seed,'fold_id':fid,
          'sampled_mechanism_gain':v['sampled_estimated']-v['sampled_true'],
          'full_mechanism_gain':v['full_estimated']-v['full_true'],
          'estimated_state_gain':v['sampled_estimated']-v['full_estimated'],
          'true_state_gain':v['sampled_true']-v['full_true'],
          'interaction':(v['sampled_estimated']-v['sampled_true'])-(v['full_estimated']-v['full_true']),
          'b1_to_full_true':v['B1_latest']-v['full_true']})
    writecsv(out/'information_contrasts.csv',contrasts)
    protocol={'status':'EXPLORATORY_2X2_INFORMATION_INTERVENTION',
      'estimated_mechanism':'historical allele-frequency movement proxy + neutral reproduction; NOT a fitted V3-B mechanism estimator',
      'sampled_state':'bootstrap latest sampled haplotypes to simulated population size; not posterior sampling',
      'true_mechanism':'original V3-B step() and original context weights; evaluator only',
      'state':'exact replay of full cutoff population for full cells',
      'warning':'Contrasts are not a unique causal or information-theoretic decomposition. Sampled-state/true-mechanism is privileged. No joint haplotype likelihood. Previously inspected seeds are exploratory.',
      'source':str(Path(a.source).resolve()),'replicates':a.replicates,'seeds':a.seeds,'scenarios':a.scenarios}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print('Saved',out,'rows',len(results))
if __name__=='__main__':main()
