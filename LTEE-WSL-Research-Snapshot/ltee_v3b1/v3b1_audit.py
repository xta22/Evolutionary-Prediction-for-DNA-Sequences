#!/usr/bin/env python3
"""Independent forward-time V3-B.1 mechanism/power audit (NumPy only).
Not a substitute for auditing v3b_simulate.py itself. No future labels used for fitting.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np


def fitness_table(s1, s2, eps):
    return np.exp(np.array([0., s2, s1, s1+s2+eps]))


def interaction_identity(s1, s2, eps):
    w=fitness_table(s1,s2,eps)
    return float(np.log(w[3])-np.log(w[2])-np.log(w[1])+np.log(w[0]))


def haplotype_run(seed, n, generations, s1, s2, eps, recomb, initial, mutation=0.):
    rng=np.random.default_rng(seed)
    # 0=00,1=01,2=10,3=11. Start from same frequencies in paired runs.
    states=np.asarray(initial,dtype=np.int8).copy()
    hist=[]
    w=fitness_table(s1,s2,eps)
    for t in range(generations+1):
        f=np.bincount(states,minlength=4)/n
        pA=f[2]+f[3]; pB=f[1]+f[3]
        D=f[3]-pA*pB
        denom=pA*(1-pA)*pB*(1-pB)
        hist.append((t,pA,pB,f[3],D*D/denom if denom>1e-12 else np.nan))
        if t==generations: break
        weights=w[states]; weights=weights/weights.sum()
        parents=rng.choice(n,size=n,replace=True,p=weights)
        offspring=states[parents].copy()
        if recomb>0:
            flags=rng.random(n)<recomb
            partners=rng.choice(n,size=int(flags.sum()),replace=True,p=weights)
            # locus 1 inherited from first parent, locus 2 from second
            offspring[flags]=(offspring[flags]&2)|(states[partners]&1)
        if mutation>0:
            flip1=rng.random(n)<mutation; flip2=rng.random(n)<mutation
            offspring ^= (flip1.astype(np.int8)*2 + flip2.astype(np.int8))
        states=offspring
    return np.asarray(hist,float)


def initial_population(seed,n):
    rng=np.random.default_rng(seed)
    return rng.choice(4,size=n,p=[.4,.2,.2,.2]).astype(np.int8)


def mutation_context(seed, sites, generations, individuals, strength):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=sites)
    susceptibility=np.exp(strength*x)
    susceptibility/=susceptibility.mean()
    # independent mutation opportunities: direct event counts, not endpoint Hamming
    counts=rng.poisson(individuals*generations*0.001*susceptibility)
    q=np.quantile(x,[.25,.75]); low=x<=q[0]; high=x>=q[1]
    return {'context_corr':float(np.corrcoef(x,counts)[0,1]),
            'context_high_low_ratio':float((counts[high].mean()+.01)/(counts[low].mean()+.01)),
            'context_expected_ratio':float(susceptibility[high].mean()/susceptibility[low].mean())}


def rate_heterogeneity(seed, n_pop, exposure, sigma):
    rng=np.random.default_rng(seed)
    logrates=rng.normal(0,sigma,size=n_pop)
    rates=np.exp(logrates)
    counts=rng.poisson(exposure*rates)
    return {'rate_true_cv':float(rates.std()/rates.mean()),
            'rate_observed_cv':float(counts.std()/max(counts.mean(),1e-12)),
            'rate_corr':float(np.corrcoef(rates,counts)[0,1]) if rates.std()>0 and counts.std()>0 else None,
            'total_events':int(counts.sum())}


def run(args):
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    rows=[]; trajectories=[]
    for seed in args.seeds:
        init=initial_population(seed,args.individuals)
        # Pairing shares initial population, not subsequent random numbers (mechanism-dependent RNG).
        settings=[('neutral',0.,0.,0.,0.),('selection_weak',args.selection,args.selection,0.,0.),
                  ('epistasis_positive',args.selection,args.selection,args.epistasis,0.),
                  ('epistasis_negative',args.selection,args.selection,-args.epistasis,0.),
                  ('recombination',0.,0.,0.,args.recombination)]
        outcomes={}
        for name,s1,s2,eps,rec in settings:
            traj=haplotype_run(seed+12345,args.individuals,args.generations,s1,s2,eps,rec,init,args.mutation)
            outcomes[name]=traj
            for t,pA,pB,p11,r2 in traj:
                trajectories.append({'seed':seed,'scenario':name,'generation':int(t),'allele_a':pA,'allele_b':pB,'double_mutant':p11,'r2':r2})
            rows.append({'seed':seed,'scenario':name,'endpoint_allele_a':traj[-1,1],
                         'endpoint_double_mutant':traj[-1,3],'endpoint_r2':traj[-1,4],
                         'midpoint_double_mutant':traj[args.generations//2,3],
                         'fitness_interaction':interaction_identity(s1,s2,eps)})
        for metric,col in [('double_mutant',3),('allele_a',1)]:
            for scenario,baseline in [('epistasis_positive','selection_weak'),('epistasis_negative','selection_weak'),('selection_weak','neutral')]:
                rows.append({'seed':seed,'scenario':'contrast_'+scenario+'_vs_'+baseline+'_'+metric,
                             'endpoint_allele_a':np.nan,'endpoint_double_mutant':np.nan,'endpoint_r2':np.nan,
                             'midpoint_double_mutant':np.nan,'fitness_interaction':np.nan,
                             'paired_delta_mid':outcomes[scenario][args.generations//2,col]-outcomes[baseline][args.generations//2,col],
                             'paired_delta_end':outcomes[scenario][-1,col]-outcomes[baseline][-1,col]})
        ctx=mutation_context(seed,args.sites,args.generations,args.individuals,args.context_strength)
        het=rate_heterogeneity(seed,args.populations,args.exposure,args.rate_sigma)
        rows.append({'seed':seed,'scenario':'context_diagnostic',**ctx})
        rows.append({'seed':seed,'scenario':'population_rate_diagnostic',**het})
    import pandas as pd
    df=pd.DataFrame(rows); df.to_csv(out/'mechanism_metrics.csv',index=False)
    pd.DataFrame(trajectories).to_csv(out/'allele_trajectories.csv',index=False)
    contrasts=df[df.scenario.str.startswith('contrast_')].groupby('scenario')[['paired_delta_mid','paired_delta_end']].agg(['mean','std','count'])
    print('=== PAIRED MECHANISM CONTRASTS ===');print(contrasts.to_string())
    print('\n=== ENDPOINT SUMMARY ===')
    print(df[df.scenario.isin(['neutral','selection_weak','epistasis_positive','epistasis_negative','recombination'])].groupby('scenario')[['endpoint_allele_a','endpoint_double_mutant','endpoint_r2']].mean().to_string())
    print('\n=== CONTEXT AND RATE DIAGNOSTICS ===')
    print(df[df.scenario.isin(['context_diagnostic','population_rate_diagnostic'])].groupby('scenario').mean(numeric_only=True).to_string())
    report={'status':'DESCRIPTIVE_MECHANISM_AUDIT_NOT_CERTIFIED', 'seeds':args.seeds,
            'fitness_identity_positive':interaction_identity(args.selection,args.selection,args.epistasis),
            'fitness_identity_additive':interaction_identity(args.selection,args.selection,0.),
            'limitations':['Standalone audit simulator; does not validate implementation of v3b_simulate.py',
                           'Paired runs share initial populations but not matched subsequent random draws',
                           'Two-locus haplotypes; no four-state genome prediction',
                           'LD undefined at fixed loci; means omit NaNs',
                           'Power requires more seeds and effect-size uncertainty intervals']}
    (out/'audit_report.json').write_text(json.dumps(report,indent=2))
    print('\nSaved',out)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',type=int,nargs='+',default=[11,23,42,67,101,137,173,211,257,307,401,503,601,701,809,907,1009,1103,1201,1301])
    ap.add_argument('--individuals',type=int,default=240)
    ap.add_argument('--generations',type=int,default=70)
    ap.add_argument('--selection',type=float,default=.025)
    ap.add_argument('--epistasis',type=float,default=.08)
    ap.add_argument('--recombination',type=float,default=.5)
    ap.add_argument('--mutation',type=float,default=.0001)
    ap.add_argument('--sites',type=int,default=500)
    ap.add_argument('--context-strength',type=float,default=1.)
    ap.add_argument('--populations',type=int,default=12)
    ap.add_argument('--exposure',type=float,default=30.)
    ap.add_argument('--rate-sigma',type=float,default=.8)
    ap.add_argument('--output',default='../LTEE_prospective/v3b1_audit')
    a=ap.parse_args()
    if a.individuals<4 or a.generations<2 or a.sites<8 or a.populations<2 or not 0<=a.recombination<=1 or not 0<=a.mutation<=1: ap.error('invalid parameters')
    run(a)
