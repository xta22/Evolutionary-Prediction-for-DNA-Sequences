#!/usr/bin/env python3
"""Phase 1.1: read-only, fail-closed prospective benchmark finalization audit.
Does NOT approve training, infer ancestry, or use future target states as features.
"""
import argparse, csv, hashlib, json, re
from collections import Counter, defaultdict
from pathlib import Path
from Bio import SeqIO

FOUNDER = {'Anc-_0gen_REL606.gd':'Ara- founder REL606', 'Anc+_0gen_REL607.gd':'Ara+ founder REL607'}
TIMELINE = [(10000,15000,'development'),(10000,20000,'development'),(20000,30000,'development'),(30000,40000,'locked_candidate'),(40000,50000,'locked_candidate')]

def rows(path):
    with path.open(newline='') as f: return list(csv.DictReader(f))

def csvout(path, data, cols):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(data)

def jsout(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def parse_non_snp(path):
    events=[]; counts=Counter(); unknown=0
    with path.open() as f:
        for line_no,line in enumerate(f,1):
            s=line.strip()
            if not s or s.startswith('#'):continue
            fields=s.split();kind=fields[0]
            if kind=='SNP':continue
            counts[kind]+=1
            # GenomeDiff mutation records usually contain seq_id and position in fields 3 and 4.
            # Do NOT assume affected interval is known; this is an audit, not a mask generator.
            pos=None
            if len(fields)>4:
                try:pos=int(fields[4])
                except (ValueError,TypeError):pass
            if pos is None:unknown+=1
            events.append({'line':line_no,'type':kind,'position_hint':pos})
    return events,counts,unknown

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path('.'))
    ap.add_argument('--phase1',type=Path,default=Path('LTEE_prospective'))
    ap.add_argument('--gd-dir',type=Path,default=Path('LTEE-clone-curated'))
    ap.add_argument('--observed-dir',type=Path,default=Path('dating_inputs/observed_alignments'))
    ap.add_argument('--reference',type=Path,default=Path('reference/REL606.gbk'))
    ap.add_argument('--out',type=Path,default=None)
    args=ap.parse_args()
    root=args.root.resolve()
    def absolute(p):return p if p.is_absolute() else root/p
    phase=absolute(args.phase1);gd=absolute(args.gd_dir);observed=absolute(args.observed_dir);reference=absolute(args.reference)
    out=absolute(args.out) if args.out else phase/'phase1_1'
    manifest=phase/'manifests/observed_clone_manifest.csv'
    folds=phase/'splits/rolling_origin_folds.csv'
    for p in [manifest,folds,reference,gd]:
        if not p.exists():raise SystemExit(f'MISSING required input: {p}')
    clones=rows(manifest); candidates=rows(folds)
    valid={r['clone_id']:r for r in clones if r['status']=='PASS'}
    if len(valid)!=sum(r['status']=='PASS' for r in clones):raise SystemExit('duplicate clone IDs')
    ref=SeqIO.read(reference,'genbank'); refhash=digest(reference)
    issues=[]; warnings=[]
    for r in clones:
        if r['reference_sha256']!=refhash:issues.append(f"reference hash mismatch: {r['clone_id']}")
        if r['status']!='PASS':issues.append(f"invalid clone: {r['clone_id']}")
    # Founders are shared historical inputs, not new populations or ordinary sample targets.
    founders=[]
    for name,meaning in FOUNDER.items():
        p=gd/name
        founders.append({'file':name,'role':meaning,'exists':p.exists(),'sha256':digest(p) if p.exists() else '',
                         'generation':0,'status':'FOUND_UNVERIFIED' if p.exists() else 'MISSING'})
        if not p.exists():issues.append(f'missing founder {name}')
    csvout(out/'manifests/founder_manifest.csv',founders,['file','role','exists','sha256','generation','status'])
    # Compare original alignment IDs, but do not load entire 4.6 Mb sequences into RAM.
    old_ids=set(); old_duplicates=[]; old_counts=Counter()
    if observed.is_dir():
        for file in sorted(observed.glob('*.fasta')):
            for rec in SeqIO.parse(file,'fasta'):
                if rec.id in old_ids:old_duplicates.append(rec.id)
                old_ids.add(rec.id);old_counts[file.stem]+=1
    else: warnings.append('observed FASTA directory missing; reconciliation incomplete')
    new_ids=set(valid); old_only=sorted(old_ids-new_ids); new_only=sorted(new_ids-old_ids)
    reconciliation=[{'clone_id':c,'in_old_fasta':c in old_ids,'in_phase1_manifest':c in new_ids,
                     'classification':('BOTH' if c in old_ids and c in new_ids else 'OLD_ONLY' if c in old_ids else 'NEW_ONLY')}
                    for c in sorted(old_ids|new_ids)]
    csvout(out/'audits/clone_reconciliation.csv',reconciliation,['clone_id','in_old_fasta','in_phase1_manifest','classification'])
    if old_duplicates:issues.append(f'duplicate old FASTA IDs: {old_duplicates[:10]}')
    # Report differences; do not automatically treat old tree-derived cohort as complete ground truth.
    if old_only or new_only:warnings.append(f'cohort mismatch: old_only={len(old_only)} new_only={len(new_only)}')
    # Non-SNP variant records: explicit uncertainty, NOT falsely asserted callable.
    non_snp_rows=[]; count_types=Counter(); n_non_snp=0
    for cid,r in sorted(valid.items()):
        p=Path(r['gd_path'])
        if not p.is_file():issues.append(f'GD missing: {cid}');continue
        if digest(p)!=r['gd_sha256']:issues.append(f'GD hash changed: {cid}');continue
        events,counts,unknown=parse_non_snp(p)
        n_non_snp+=len(events);count_types.update(counts)
        non_snp_rows.append({'clone_id':cid,'non_snp_records':len(events),'types':';'.join(f'{k}:{v}' for k,v in sorted(counts.items())),
                             'callable_status':'UNKNOWN_NON_SNP_EFFECT' if events else 'SNP_ONLY_NO_CALLABILITY_PROOF',
                             'unresolved_positions':sum(e['position_hint'] is None for e in events)})
    csvout(out/'audits/non_snp_audit.csv',non_snp_rows,['clone_id','non_snp_records','types','callable_status','unresolved_positions'])
    # Origin policy: all latest-generation clones, uniform weights as a declared *sampling convention*.
    origin_rows=[];protocol=[]; fold_issues=[]; target_counts=Counter()
    for cutoff,target,usage in TIMELINE:
        for pop in sorted(set(r['population'] for r in valid.values())):
            hist=sorted((r for r in valid.values() if r['population']==pop and int(r['generation'])<=cutoff),key=lambda r:(int(r['generation']),r['clone_id']))
            futures=sorted((r for r in valid.values() if r['population']==pop and int(r['generation'])==target),key=lambda r:r['clone_id'])
            if not hist or not futures:
                fold_issues.append({'population':pop,'cutoff':cutoff,'target':target,'reason':'missing history or targets'});continue
            latest=max(int(r['generation']) for r in hist)
            origins=[r for r in hist if int(r['generation'])==latest]
            if not origins:raise RuntimeError('origin selection impossible')
            fold_id=f'{pop}_{cutoff}_{target}'
            for origin in origins:
                origin_rows.append({'fold_id':fold_id,'population':pop,'cutoff':cutoff,'target_generation':target,
                                    'origin_clone_id':origin['clone_id'],'origin_generation':latest,
                                    'origin_weight':1/len(origins),'origin_policy':'latest_generation_uniform_ensemble',
                                    'origin_gd_sha256':origin['gd_sha256'],'historical_clone_count':len(hist),
                                    'target_count_evaluation_only':len(futures),'usage':usage})
            protocol.append({'fold_id':fold_id,'population':pop,'cutoff':cutoff,'target_generation':target,'horizon':target-cutoff,
                             'usage':usage,'historical_clone_count':len(hist),'latest_origin_generation':latest,
                             'n_origins':len(origins),'n_future_targets':len(futures),
                             'status':'CANDIDATE_UNVERIFIED'})
            for future in futures:target_counts[future['clone_id']]+=1
    csvout(out/'manifests/forecast_origins.csv',origin_rows,['fold_id','population','cutoff','target_generation','origin_clone_id','origin_generation','origin_weight','origin_policy','origin_gd_sha256','historical_clone_count','target_count_evaluation_only','usage'])
    csvout(out/'splits/temporal_protocol.csv',protocol,['fold_id','population','cutoff','target_generation','horizon','usage','historical_clone_count','latest_origin_generation','n_origins','n_future_targets','status'])
    # All target IDs stay in a separate evaluation-only file.
    evaluation_targets=[]
    for r in candidates:
        cid=r['target_clone_id'];pop=r['population'];cutoff=int(r['cutoff']);target=int(r['target_generation'])
        if cid not in valid:issues.append(f'target not in valid manifest: {cid}');continue
        if valid[cid]['population']!=pop or int(valid[cid]['generation'])!=target or target<=cutoff:
            issues.append(f'target timing/pop mismatch: {cid}')
        evaluation_targets.append({'fold_id':f'{pop}_{cutoff}_{target}','target_clone_id':cid,'evaluation_only':True})
    csvout(out/'splits/evaluation_targets_RESTRICTED.csv',evaluation_targets,['fold_id','target_clone_id','evaluation_only'])
    if any(r['origin_generation']>r['cutoff'] for r in origin_rows):issues.append('future origin detected')
    if any(r['origin_clone_id'] in {x['target_clone_id'] for x in evaluation_targets if x['fold_id']==r['fold_id']} for r in origin_rows):issues.append('origin is same as future target')
    # A future target at an earlier fold can legitimately be history for a later cutoff;
    # this is rolling-origin evaluation, NOT a globally disjoint train/test split.
    jsout(out/'audits/phase1_1_report.json',{
        'status':'BLOCKED' if issues else 'CANDIDATE_ONLY_NOT_APPROVED',
        'hard_issues':issues,'warnings':warnings,'founders':founders,
        'clone_reconciliation':{'old_fasta_ids':len(old_ids),'phase1_valid_ids':len(new_ids),'old_only':old_only,'new_only':new_only,'duplicate_old_ids':old_duplicates},
        'non_snp':{'records':n_non_snp,'by_type':dict(count_types),'callable_mask_status':'NOT_IMPLEMENTED; do not score non-SNP affected sites as confidently unchanged'},
        'protocol':{'folds':len(protocol),'origin_rows':len(origin_rows),'evaluation_target_rows':len(evaluation_targets),
                    'missing_folds':fold_issues,'policy':'latest-generation observed clones, uniform mixture; not a true population frequency estimate',
                    'future_target_ids':'evaluation_targets_RESTRICTED.csv only; never provide as model inputs',
                    'usage':'development folds exploratory; locked_candidate folds NOT independently locked until approved'},
        'unverified':['upstream clone-local GD variant-calling provenance','GD naming and actual sample time metadata',
                      'non-SNP/callable masking','preprocessing dependency graph and independent evaluation','train/validation/test freeze protocol'],
        'source_hashes':{'reference':refhash,'phase1_manifest':digest(manifest),'phase1_folds':digest(folds)},
        'approval':'NO_TRAINING_AUTHORIZATION'
    })
    print('PHASE 1.1', 'BLOCKED' if issues else 'CANDIDATE_ONLY_NOT_APPROVED')
    print('founders',len(founders),'old FASTA',len(old_ids),'valid GD clones',len(valid),'old_only',len(old_only),'new_only',len(new_only))
    print('folds',len(protocol),'origin rows',len(origin_rows),'evaluation target rows',len(evaluation_targets))
    print('non-SNP records',n_non_snp,'types',dict(count_types))
    print('hard issues',len(issues),'warnings',len(warnings))
    for issue in issues[:10]:print('ISSUE',issue)
    print('Saved',out)
    print('DO NOT TRAIN: provenance and callability remain unverified.')
    if issues:raise SystemExit(2)

if __name__=='__main__':main()
