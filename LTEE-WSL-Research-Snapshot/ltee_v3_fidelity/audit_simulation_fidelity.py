#!/usr/bin/env python3
"""Read-only LTEE vs V3-B simulation fidelity audit. Never certifies comparability."""
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd

DNA=set('ACGT')
def fasta(path):
    with open(path) as f:
        return ''.join(x.strip().upper() for x in f if x.strip() and not x.startswith('>'))
def parse_gd(path):
    meta={}; snps=[]; types={}; malformed=[]
    with open(path) as f:
        for lineno,line in enumerate(f,1):
            s=line.strip()
            if s.startswith('#='):
                a=s[2:].split(None,1)
                if len(a)==2:meta[a[0]]=a[1]
            elif s and not s.startswith('#'):
                a=s.split(); types[a[0]]=types.get(a[0],0)+1
                if a[0]=='SNP':
                    if len(a)<6 or not a[4].isdigit() or a[5].upper() not in DNA:
                        malformed.append(lineno)
                    else:snps.append((int(a[4]),a[5].upper()))
    return meta,snps,types,malformed

def real_table(gd_dir,ref):
    records=[]; issues=[]; refseq=fasta(ref) if ref.suffix.lower() not in ('.gbk','.gb') else None
    if refseq is None:
        from Bio import SeqIO
        refseq=str(SeqIO.read(ref,'genbank').seq).upper()
    for path in sorted(gd_dir.glob('*.gd')):
        meta,snps,types,bad=parse_gd(path)
        m=re.match(r'^(Ara[+-]\d+)_(\d+)gen_',path.stem)
        if not m:
            issues.append({'file':path.name,'issue':'unrecognized clone name; excluded'});continue
        pop,t=m.group(1),int(m.group(2))
        if bad:issues.append({'file':path.name,'issue':f'malformed SNP records {bad[:10]}'})
        uniq={};conflicts=0;invalid=0
        for pos,b in snps:
            if pos<1 or pos>len(refseq):invalid+=1;continue
            if pos in uniq and uniq[pos]!=b:conflicts+=1
            uniq[pos]=b
        if invalid or conflicts:issues.append({'file':path.name,'issue':f'invalid positions={invalid}; conflicting SNPs={conflicts}'})
        records.append({'source':'LTEE_GD','population':pop,'generation':t,'clone_id':path.stem,
          'snp_calls':len(uniq),'snp_density':len(uniq)/len(refseq),
          'non_snp_calls':sum(v for k,v in types.items() if k!='SNP'),
          'reference_length':len(refseq),'malformed_snp':len(bad),
          'non_snp_affected_callability':'UNKNOWN'})
    return pd.DataFrame(records),issues

def simulation_table(root):
    rows=[];issues=[]
    for p in sorted(root.glob('*/seed_*/observed_genomes.csv')):
        scenario=p.parent.parent.name;seed=p.parent.name
        try:df=pd.read_csv(p,dtype={"sequence":"string"})
        except Exception as e:issues.append({'file':str(p),'issue':str(e)});continue
        if 'generation' not in df or 'sequence' not in df:
            issues.append({'file':str(p),'issue':'missing generation or sequence'});continue
        for t,g in df.groupby('generation'):
            seqs=g.sequence.astype(str).tolist()
            if not seqs:continue
            lengths={len(s) for s in seqs}
            if len(lengths)!=1:issues.append({'file':str(p),'issue':'mixed lengths'});continue
            arr=np.array([list(s) for s in seqs]);
            if not np.isin(arr,['0','1']).all():issues.append({'file':str(p),'issue':'nonbinary sequence'});continue
            freqs=(arr=='1').mean(axis=0)
            rows.append({'scenario':scenario,'seed':seed,'generation':float(t),
              'sample_size':len(seqs),'sites':arr.shape[1],
              'derived_allele_fraction':float(freqs.mean()),
              'segregating_site_fraction':float(((freqs>0)&(freqs<1)).mean()),
              'mean_site_heterozygosity':float((2*freqs*(1-freqs)).mean())})
    return pd.DataFrame(rows),issues

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--gd-dir',type=Path,required=True)
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--simulation',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if not args.gd_dir.is_dir() or not args.reference.is_file() or not args.simulation.is_dir():
        ap.error('missing required input path; nothing inferred or fabricated')
    real,ri=real_table(args.gd_dir,args.reference)
    sim,si=simulation_table(args.simulation)
    if real.empty or sim.empty:raise SystemExit('BLOCKED: missing real clone records or simulation sampled genomes')
    real.to_csv(args.output/'real_clone_snp_summary.csv',index=False)
    sim.to_csv(args.output/'simulation_sample_summary.csv',index=False)
    real_summary=real.groupby(['population','generation'],as_index=False).agg(clones=('clone_id','nunique'),mean_snp_calls=('snp_calls','mean'),mean_snp_density=('snp_density','mean'),mean_non_snp_calls=('non_snp_calls','mean'))
    real_summary.to_csv(args.output/'real_temporal_summary.csv',index=False)
    sim_summary=sim.groupby(['scenario','generation'],as_index=False).agg(trajectories=('seed','nunique'),mean_derived_fraction=('derived_allele_fraction','mean'),mean_heterozygosity=('mean_site_heterozygosity','mean'),mean_segregating_fraction=('segregating_site_fraction','mean'))
    sim_summary.to_csv(args.output/'simulation_temporal_summary.csv',index=False)
    # Sampling interval diagnostics use unique generations, not pairwise pseudo-lineages.
    intervals=[]
    for pop,g in real.groupby('population'):
        ts=sorted(g.generation.unique());intervals.extend([{'source':'LTEE_GD','group':pop,'interval':int(b-a)} for a,b in zip(ts,ts[1:])])
    for (scenario,seed),g in sim.groupby(['scenario','seed']):
        ts=sorted(g.generation.unique());intervals.extend([{'source':'V3B','group':f'{scenario}/{seed}','interval':float(b-a)} for a,b in zip(ts,ts[1:])])
    pd.DataFrame(intervals,columns=['source','group','interval']).to_csv(args.output/'sampling_intervals.csv',index=False)
    report={'status':'DESCRIPTIVE_NOT_COMPARABLE','real_clones':len(real),'sim_sample_timepoints':len(sim),
      'real_populations':int(real.population.nunique()),'sim_scenarios':sorted(sim.scenario.unique().tolist()),
      'issues':ri+si,
      'critical_limitations':[
       'V3-B binary sites are not 4-state LTEE nucleotide positions: no direct SNP spectrum or per-site mutation-rate comparison.',
       'GD SNP counts are accumulated differences from REL606, not branch mutation events or population allele frequencies.',
       'GD non-SNP events and callable regions are unresolved; no confident unchanged-site labels.',
       'Observed clones are not parent-child pairs; no branch-level mutation count inferred.',
       'Sampling intervals in generations do not imply equivalent mutation scales or effective population sizes.',
       'No similarity score or calibration certificate is issued; compare only within well-defined common observables.'
      ],'next_steps':['Implement reference-coordinate-aware callable masks and event parsing',
       'Build a real four-state observed-clone benchmark with certified time-causal provenance',
       'Calibrate simulation observation model and event-rate scale before sim-to-real claims']}
    (args.output/'fidelity_report.json').write_text(json.dumps(report,indent=2))
    print('STATUS',report['status'],'real clones',len(real),'sim timepoints',len(sim),'issues',len(ri)+len(si))
    print('Saved',args.output)
if __name__=='__main__':main()
