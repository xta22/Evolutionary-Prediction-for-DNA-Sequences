#!/usr/bin/env python3
"""Read-only, conservative GenomeDiff provenance / coordinate audit.
No training authorization, no verified callable mask. Python 3.9+.
"""
import argparse, csv, hashlib, json, re
from pathlib import Path
from collections import Counter, defaultdict

EVENTS={'SNP','DEL','INS','MOB','AMP','CON','SUB','INV'}
INTERVAL={'DEL','AMP','SUB','CON','INV'}
ANCHOR={'INS','MOB'}
FOUNDERS={'Anc-_0gen_REL606','Anc+_0gen_REL607'}

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def write_csv(path,rows,columns):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)

def read_csv(path):
    if not path.exists():return []
    with path.open(newline='') as f:return list(csv.DictReader(f))

def split_circular_interval(start, size, length):
    """Candidate 1-based intervals for a circular reference; NOT callability certification."""
    if not (1 <= start <= length and 1 <= size <= length):
        raise ValueError('Invalid circular interval')
    end = start + size - 1
    if end <= length:
        return [(start, end)]
    return [(start, length), (1, end - length)]


def extract_relation_annotations(fields):
    return ';'.join(f for f in fields[6:] if f.startswith(('within=', 'before=', 'after=', 'adjacent=', 'mediated=', 'mob_region=')))


def parse_gd(path,length,reference_ids):
    metadata=defaultdict(list);events=[];issues=[];counts=Counter()
    with path.open(errors='replace') as f:
        for line_no,line in enumerate(f,1):
            s=line.strip()
            if s.startswith('#='):
                x=s[2:].split(None,1)
                if x:metadata[x[0].upper()].append(x[1].strip() if len(x)>1 else '')
                continue
            if not s or s.startswith('#'):continue
            fields=s.split(); typ=fields[0]
            if typ not in EVENTS:continue
            counts[typ]+=1
            e={'clone_id':path.stem,'event_type':typ,'line':line_no,'event_id':fields[1] if len(fields)>1 else '',
               'reference':fields[3] if len(fields)>3 else '', 'position':fields[4] if len(fields)>4 else '',
               'raw':s,'classification':'UNRESOLVED','start_1based':'','end_1based':'','reason':'',
               'relation_annotations':extract_relation_annotations(fields),'circular_candidate':False,'candidate_segments':''}
            if len(fields)<6:
                e['reason']='TOO_FEW_FIELDS';issues.append((line_no,'TOO_FEW_FIELDS'))
            else:
                try:pos=int(fields[4])
                except ValueError:pos=None
                if fields[3] not in reference_ids:
                    e['reason']='REFERENCE_MISMATCH';issues.append((line_no,'REFERENCE_MISMATCH'))
                elif pos is None or not 1<=pos<=length:
                    e['reason']='INVALID_POSITION';issues.append((line_no,'INVALID_POSITION'))
                elif typ in INTERVAL:
                    # GenomeDiff common syntax: pos, size; uncertain types are candidate intervals only.
                    try:size=int(fields[5])
                    except ValueError:size=None
                    if size is None or size <= 0 or size > length:
                        e['reason']='INVALID_OR_UNKNOWN_SIZE';issues.append((line_no,'INVALID_OR_UNKNOWN_SIZE'))
                    elif pos + size - 1 > length:
                        # GenomeDiff circular semantics and nested event coordinates are NOT independently verified.
                        # Keep this as a candidate, never a certified interval or mask.
                        segments = split_circular_interval(pos, size, length)
                        e.update(classification='CIRCULAR_INTERVAL_CANDIDATE',
                                 start_1based=pos, end_1based=pos+size-1,
                                 circular_candidate=True,
                                 candidate_segments=json.dumps(segments),
                                 reason='CIRCULAR_WRAP_POSSIBLE_REQUIRES_GD_SEMANTICS_REVIEW')
                    else:
                        e.update(classification='CANDIDATE_REFERENCE_INTERVAL',start_1based=pos,end_1based=pos+size-1,
                                 candidate_segments=json.dumps([(pos,pos+size-1)]),
                                 reason='NEEDS_EVENT_SEMANTICS_REVIEW')
                elif typ in ANCHOR:
                    e.update(classification='ANCHOR_ONLY',start_1based=pos,end_1based=pos,reason='INSERTION_EXTENT_UNRESOLVED')
                else:
                    e.update(classification='SNP_RECORD',start_1based=pos,end_1based=pos,reason='')
            events.append(e)
    return metadata,events,issues,counts

def canonical(name):
    aliases={'Ara-2_30000gen_30K_11832':'Ara-2_30000gen_11832',
             'Ara-2_30000gen_30K_11833':'Ara-2_30000gen_11833'}
    return aliases.get(name,name)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='.')
    ap.add_argument('--phase1',default='LTEE_prospective')
    ap.add_argument('--gd-dir',default='LTEE-clone-curated')
    ap.add_argument('--reference',default='reference/REL606.gbk')
    ap.add_argument('--out',default=None)
    args=ap.parse_args()
    root=Path(args.root).resolve();phase=(root/args.phase1).resolve();out=Path(args.out).resolve() if args.out else phase/'phase1_3'
    gd_dir=root/args.gd_dir;ref=root/args.reference
    if not gd_dir.is_dir() or not ref.is_file():ap.error('GD directory or reference missing')
    from Bio import SeqIO
    record=SeqIO.read(ref,'genbank');L=len(record.seq)
    reference_ids={record.id,record.name,'REL606','REL606.6'}
    manifest_path=phase/'manifests'/'observed_clone_manifest.csv'
    folds_path=phase/'splits'/'rolling_origin_folds.csv'
    manifest=read_csv(manifest_path);folds=read_csv(folds_path)
    if not manifest or not folds:ap.error('Phase 1 manifest or folds missing; verify paths')
    cols=set(manifest[0]);idcol=next((x for x in ('clone_id','sample_id','id') if x in cols),None)
    if not idcol:ap.error('Could not locate clone ID column in Phase 1 manifest: '+str(sorted(cols)))
    manifest_ids={canonical(r[idcol]) for r in manifest}
    provenance=[];events_all=[];counts=Counter();hard=[];meta_conflicts=[]
    for p in sorted(gd_dir.glob('*.gd')):
        md,ev,issues,c=parse_gd(p,L,reference_ids);events_all.extend(ev);counts.update(c)
        def one(k):return md[k][0] if md[k] else ''
        title=one('TITLE');time=one('TIME');pop=one('POPULATION');readseq=one('READSEQ');refseq=one('REFSEQ')
        m=re.search(r'_(\d+)gen_',p.stem)
        expected_gen=m.group(1) if m else ('0' if p.stem.startswith('Anc') else '')
        problems=[str(a)+':'+b for a,b in issues]
        if title and canonical(title)!=canonical(p.stem):problems.append('TITLE_MISMATCH')
        if expected_gen and time and expected_gen!=time:problems.append('TIME_MISMATCH')
        if p.stem.startswith('Ara') and pop and not p.stem.startswith(pop+'_'):problems.append('POPULATION_MISMATCH')
        if not readseq and p.stem not in FOUNDERS:problems.append('MISSING_READSEQ')
        founder_exception = bool(not readseq and p.stem in FOUNDERS)
        if not refseq:problems.append('MISSING_REFSEQ')
        provenance.append(dict(clone_id=p.stem,canonical_id=canonical(p.stem),in_phase1=canonical(p.stem) in manifest_ids,
            title=title,generation=time,population=pop,readseq=readseq,refseq=refseq,gd_sha256=digest(p),
            event_count=len(ev),founder_readseq_exception=founder_exception,issues=';'.join(problems),status='METADATA_CONFLICT' if problems else 'SOURCE_TRACEABLE_NOT_CERTIFIED'))
        if problems:meta_conflicts.append({'clone':p.stem,'issues':problems[:12]})
    interval_rows=[{k:e[k] for k in ('clone_id','event_type','line','event_id','reference','start_1based','end_1based','classification','reason','relation_annotations','circular_candidate','candidate_segments')}
                   for e in events_all if e['classification'] in ('CANDIDATE_REFERENCE_INTERVAL','CIRCULAR_INTERVAL_CANDIDATE','ANCHOR_ONLY')]
    clone_summary=[];byclone=defaultdict(list)
    for e in events_all:byclone[e['clone_id']].append(e)
    for r in provenance:
        ee=byclone[r['clone_id']]; cc=Counter(x['classification'] for x in ee)
        clone_summary.append({'clone_id':r['clone_id'],'candidate_intervals':cc['CANDIDATE_REFERENCE_INTERVAL'],
            'circular_candidates':cc['CIRCULAR_INTERVAL_CANDIDATE'],
            'anchor_only':cc['ANCHOR_ONLY'],'unresolved':cc['UNRESOLVED'],
            'callability_status':'NOT_VERIFIED','source_status':r['status']})
    # Only audit fold metadata; future target IDs are never used as features.
    fold_summary=[];bad_fold=[]
    for i,r in enumerate(folds):
        try:
            cutoff=int(float(r['cutoff']));target=int(float(r['target_generation']));horizon=int(float(r['horizon']))
            hist=int(float(r['historical_max_generation']))
            ok=hist<=cutoff<target and target-cutoff==horizon
        except (KeyError,ValueError,TypeError):ok=False
        if not ok:bad_fold.append(i)
        fold_summary.append({'row':i,'population':r.get('population',''),'cutoff':r.get('cutoff',''),
            'target_generation':r.get('target_generation',''),'time_fields_consistent':ok,
            'eligibility':'BLOCKED_PROVENANCE_AND_CALLABILITY' if ok else 'INVALID_TEMPORAL_METADATA'})
    if bad_fold:hard.append('invalid temporal metadata rows: '+str(bad_fold[:20]))
    if meta_conflicts:hard.append('GD metadata/event conflicts: '+str(len(meta_conflicts)))
    # Reference intervals are NOT certified masks. Future clone exclusions must never be model inputs.
    write_csv(out/'manifests'/'gd_provenance_manifest.csv',provenance,
              ['clone_id','canonical_id','in_phase1','title','generation','population','readseq','refseq','gd_sha256','event_count','founder_readseq_exception','issues','status'])
    write_csv(out/'audits'/'non_snp_events.csv',events_all,
              ['clone_id','event_type','line','event_id','reference','position','classification','start_1based','end_1based','reason','relation_annotations','circular_candidate','candidate_segments','raw'])
    write_csv(out/'audits'/'reference_exclusion_candidates.csv',interval_rows,
              ['clone_id','event_type','line','event_id','reference','start_1based','end_1based','classification','reason','relation_annotations','circular_candidate','candidate_segments'])
    write_csv(out/'audits'/'clone_mask_summary.csv',clone_summary,
              ['clone_id','candidate_intervals','circular_candidates','anchor_only','unresolved','callability_status','source_status'])
    write_csv(out/'audits'/'fold_eligibility.csv',fold_summary,
              ['row','population','cutoff','target_generation','time_fields_consistent','eligibility'])
    report={'status':'BLOCKED' if hard else 'AUDITED_NOT_APPROVED','hard_issues':hard,'reference_id':record.id,
        'reference_length':L,'gd_files':len(provenance),'phase1_manifest_rows':len(manifest),
        'phase1_clone_id_column':idcol,'fold_rows':len(folds),'invalid_fold_rows':len(bad_fold),
        'event_types':dict(counts),'event_classifications':dict(Counter(e['classification'] for e in events_all)),
        'metadata_conflicts_examples':meta_conflicts[:20],
        'founder_readseq_exceptions':[r['clone_id'] for r in provenance if r['founder_readseq_exception']],
        'circular_candidates':[{'clone_id':e['clone_id'],'line':e['line'],'event_id':e['event_id'],
            'event_type':e['event_type'],'candidate_segments':e['candidate_segments'],
            'relation_annotations':e['relation_annotations']} for e in events_all if e['circular_candidate']],
        'provenance':'SOURCE_TRACEABILITY_ONLY; upstream calling and curation NOT certified time-causal',
        'callability':'NOT VERIFIED; candidate intervals are NOT a callable mask',
        'training_authorized':False,
        'limitations':['SNP-only reference representation','No verified coverage-based callable regions',
                       'Future target annotations must be used only by evaluator',
                       'No assessment of upstream joint variant calling or post-hoc curation',
                       'No biological interpretation of candidate interval semantics without GenomeDiff specification',
                        'Circular wraparound coordinates are hypotheses, not confirmed GenomeDiff interpretations']}
    (out/'audits').mkdir(parents=True,exist_ok=True)
    (out/'audits'/'benchmark_integrity_report.json').write_text(json.dumps(report,indent=2))
    print('PHASE 1.3',report['status'])
    print('GD files',len(provenance),'events',sum(counts.values()),'fold rows',len(folds))
    print('Event classes',report['event_classifications'])
    print('Hard issues',hard[:5]);print('Saved',out)
    print('DO NOT TRAIN: provenance and callable positions remain unverified')

if __name__=='__main__':main()