#!/usr/bin/env python3
"""Phase 1.2: conservative, read-only LTEE prospective benchmark audit.
No training, no future-target features, no automatic callability approval.
"""
import argparse, csv, gzip, hashlib, json, re
from collections import Counter, defaultdict
from pathlib import Path
from Bio import SeqIO

FOUNDERS = {'Anc+_0gen_REL607': 'Ara+', 'Anc-_0gen_REL606': 'Ara-'}
ALIASES = {'Ara-2_30000gen_30K_11832': 'Ara-2_30000gen_11832',
           'Ara-2_30000gen_30K_11833': 'Ara-2_30000gen_11833'}
# GenomeDiff mutation records only. Evidence and metadata lines are not mutation events.
EVENTS = {'SNP','SUB','DEL','INS','MOB','AMP','CON','INV'}
# These types have a documented start+size representation in common breseq GD files.
# For CON/INV/AMP the extent is recorded but interpretation remains uncertain.
SIZED = {'DEL','SUB','AMP','CON','INV'}
UNCERTAIN = {'INS','MOB','AMP','CON','INV'}

def sha_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def write_csv(p, data, fields):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(data)

def write_json(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')

def read_csv(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))

def gd_parse(path):
    """Parse GD mutation records and headers; preserve raw fields and flags."""
    events=[]; headers=[]; malformed=[]; evidence=Counter()
    with path.open() as f:
        for lineno,line in enumerate(f,1):
            s=line.strip()
            if not s:continue
            if s.startswith('#'):
                headers.append(s)
                continue
            t=s.split();kind=t[0]
            if kind not in EVENTS:
                evidence[kind]+=1
                continue
            if len(t)<5:
                malformed.append({'line':lineno,'reason':'too_few_fields','raw':s[:250]});continue
            try:pos=int(t[4])
            except ValueError:
                malformed.append({'line':lineno,'reason':'invalid_position','raw':s[:250]});continue
            events.append({'line':lineno,'kind':kind,'seq_id':t[3],'pos':pos,'args':t[5:], 'raw':s})
    return events,headers,malformed,evidence

def interval_for(event,length):
    """Return 0-based half-open reference span only when safely parsable."""
    kind=event['kind'];pos=event['pos'];a=event['args']
    if not 1<=pos<=length:return None,'INVALID_POSITION'
    if kind=='SNP':return (pos-1,pos),'SNP_EXCLUDED'
    if kind in SIZED:
        if not a:return None,'MISSING_SIZE'
        try:size=int(a[0])
        except ValueError:return None,'INVALID_SIZE'
        if size<=0 or pos-1+size>length:return None,'INVALID_INTERVAL'
        return (pos-1,pos-1+size),'REFERENCE_INTERVAL'
    if kind in {'INS','MOB'}:
        # Reference anchor is known, inserted bases have no 1:1 reference span.
        return (pos-1,pos),'ANCHOR_ONLY_UNRESOLVED_EXTENT'
    return None,'UNKNOWN_EVENT'

def merged(intervals):
    result=[]
    for start,end in sorted(intervals):
        if result and start<=result[-1][1]:result[-1]=(result[-1][0],max(result[-1][1],end))
        else:result.append((start,end))
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path('.'))
    ap.add_argument('--phase1',type=Path,default=Path('LTEE_prospective'))
    ap.add_argument('--gd-dir',type=Path,default=Path('LTEE-clone-curated'))
    ap.add_argument('--observed-dir',type=Path,default=Path('dating_inputs/observed_alignments'))
    ap.add_argument('--reference',type=Path,default=Path('reference/REL606.gbk'))
    ap.add_argument('--out',type=Path,default=None)
    args=ap.parse_args();root=args.root.resolve()
    def abs_path(p):return p if p.is_absolute() else root/p
    phase=abs_path(args.phase1);gd=abs_path(args.gd_dir);obs=abs_path(args.observed_dir);ref=abs_path(args.reference)
    out=abs_path(args.out) if args.out else phase/'phase1_2'
    manifest=phase/'manifests/observed_clone_manifest.csv'
    protocol=phase/'phase1_1/splits/temporal_protocol.csv'
    for p in (manifest,ref,gd,obs):
        if not p.exists():raise SystemExit('Missing required input: '+str(p))
    refrec=SeqIO.read(ref,'genbank');reference=str(refrec.seq).upper();L=len(reference)
    ref_id=refrec.id;valid_ids={ref_id,ref_id.split('.')[0]}
    records=read_csv(manifest);clones={r['clone_id']:r for r in records if r.get('status')=='PASS'}
    hard=[];warnings=[]
    if len(clones)!=sum(r.get('status')=='PASS' for r in records):hard.append('duplicate PASS clone IDs in manifest')
    if any(r.get('reference_sha256')!=sha_file(ref) for r in clones.values()):hard.append('reference hash mismatch in Phase 1 manifest')
    # FASTA provenance and founder hashes: stream records, keep only two alias sequences.
    founder_hashes=defaultdict(list);alias_fasta={};fasta_ids=defaultdict(list)
    for fasta in sorted(obs.glob('*.fasta')):
        for rec in SeqIO.parse(fasta,'fasta'):
            cid=rec.id;seq=str(rec.seq).upper()
            fasta_ids[cid].append(fasta.stem)
            if cid in FOUNDERS:founder_hashes[cid].append((fasta.stem,hashlib.sha256(seq.encode()).hexdigest()))
            if cid in ALIASES:alias_fasta[cid]=seq
    founder_rows=[]
    for cid,side in FOUNDERS.items():
        entries=founder_hashes[cid];hashes={h for _,h in entries}
        expected={p for p in (['Ara+1','Ara+2','Ara+3','Ara+4','Ara+5','Ara+6'] if side=='Ara+' else ['Ara-1','Ara-2','Ara-3','Ara-4','Ara-5','Ara-6'])}
        status='PASS_SHARED_IDENTICAL' if len(entries)==6 and len(hashes)==1 and {p for p,_ in entries}==expected else 'FAIL'
        if status=='FAIL':hard.append('founder FASTA mismatch: '+cid)
        founder_rows.append({'founder_id':cid,'side':side,'occurrences':len(entries),'unique_hashes':len(hashes),'populations':';'.join(p for p,_ in entries),'sha256':next(iter(hashes)) if len(hashes)==1 else '', 'status':status})
    write_csv(out/'manifests/founder_reconciliation.csv',founder_rows,['founder_id','side','occurrences','unique_hashes','populations','sha256','status'])
    # Explicit alias SNP-resolved sequence equivalence.
    alias_rows=[]
    for old,new in ALIASES.items():
        m=clones.get(new);oldseq=alias_fasta.get(old);path=gd/(new+'.gd')
        status='FAIL';n_snps=0;details=''
        if m and oldseq is not None and path.exists() and sha_file(path)==m.get('gd_sha256'):
            events,headers,malformed,evidence=gd_parse(path)
            reconstructed=bytearray(reference.encode())
            snp_positions=set();bad=[]
            for e in events:
                if e['kind']!='SNP':continue
                n_snps+=1
                if e['seq_id'] not in valid_ids or not e['args'] or e['args'][0] not in 'ACGT' or len(e['args'][0])!=1 or not 1<=e['pos']<=L or e['pos'] in snp_positions:
                    bad.append(e['line']);continue
                snp_positions.add(e['pos']);reconstructed[e['pos']-1]=ord(e['args'][0])
            status='PASS_EQUIVALENT' if not bad and not malformed and len(oldseq)==L and reconstructed==oldseq.encode() else 'FAIL'
            details=f'bad_snp_lines={bad[:10]}; malformed={len(malformed)}; old_length={len(oldseq)}'
        else:details='missing old FASTA, new manifest, GD, or GD hash mismatch'
        if status!='PASS_EQUIVALENT':hard.append('alias mismatch: '+old+' -> '+new)
        alias_rows.append({'old_id':old,'canonical_id':new,'n_snps':n_snps,'status':status,'details':details})
    write_csv(out/'audits/alias_reconciliation.csv',alias_rows,['old_id','canonical_id','n_snps','status','details'])
    # Non-SNP event-level audit and conservative per-clone excluded intervals.
    event_rows=[];mask_rows=[];clone_rows=[];types=Counter();reasons=Counter();header_keys=Counter()
    unresolved_clones=[];changed_gd=[];n_events=0
    for cid,r in sorted(clones.items()):
        p=Path(r['gd_path'])
        if not p.is_absolute():p=root/p
        if not p.exists() or sha_file(p)!=r.get('gd_sha256'):
            hard.append('missing/changed GD: '+cid);changed_gd.append(cid);continue
        events,headers,malformed,evidence=gd_parse(p)
        for h in headers:
            if h.startswith('#='):header_keys[h.split()[0]]+=1
        if malformed:hard.append('malformed mutation records in '+cid)
        spans=[];unresolved=[];counts=Counter()
        for e in events:
            if e['kind']=='SNP':continue
            n_events+=1;types[e['kind']]+=1;counts[e['kind']]+=1
            interval,reason=interval_for(e,L)
            if e['seq_id'] not in valid_ids:reason='REFERENCE_ID_MISMATCH';interval=None
            reasons[reason]+=1
            if interval is not None:spans.append(interval)
            if reason not in ('REFERENCE_INTERVAL',):unresolved.append(e)
            event_rows.append({'clone_id':cid,'line':e['line'],'type':e['kind'],'refseq':e['seq_id'],'position_1based':e['pos'],
                'first_arg':e['args'][0] if e['args'] else '', 'start_0based':interval[0] if interval else '',
                'end_exclusive':interval[1] if interval else '', 'classification':reason,
                'raw_record':e['raw'][:500]})
        intervals=merged(spans)
        for a,b in intervals:mask_rows.append({'clone_id':cid,'start_0based':a,'end_exclusive':b,'reason':'NON_SNP_EVENT_SPAN_OR_ANCHOR','coordinate_system':'REL606_REFERENCE'})
        if unresolved:unresolved_clones.append(cid)
        clone_rows.append({'clone_id':cid,'non_snp_events':sum(counts.values()),'types':';'.join(f'{k}:{v}' for k,v in sorted(counts.items())),
            'masked_reference_bases':sum(b-a for a,b in intervals),'merged_intervals':len(intervals),
            'unresolved_event_count':len(unresolved),'callability':'UNVERIFIED',
            'status':'REQUIRES_MANUAL_REVIEW' if unresolved or malformed else 'PARTIAL_COORDINATE_MASK_ONLY'})
    write_csv(out/'audits/non_snp_events.csv',event_rows,['clone_id','line','type','refseq','position_1based','first_arg','start_0based','end_exclusive','classification','raw_record'])
    write_csv(out/'data/conservative_exclusion_intervals.csv',mask_rows,['clone_id','start_0based','end_exclusive','reason','coordinate_system'])
    write_csv(out/'audits/clone_callability.csv',clone_rows,['clone_id','non_snp_events','types','masked_reference_bases','merged_intervals','unresolved_event_count','callability','status'])
    # Only audit protocol; never open evaluation target labels.
    temporal_status='UNVERIFIED'
    if protocol.exists():
        pr=read_csv(protocol)
        if any(int(x['cutoff'])>=int(x['target_generation']) for x in pr):hard.append('invalid chronological fold')
        temporal_status='CANDIDATE_ONLY'
    else:warnings.append('Phase 1.1 protocol missing')
    # GD metadata is not a chain-of-custody certificate.
    provenance={'status':'UNVERIFIED','gd_header_key_counts':dict(header_keys),
        'required_external_evidence':['clone-level sequencing accession / sample metadata',
        'variant calling pipeline and reference version','confirmation of no future-sample joint correction',
        'callable regions / coverage evidence','indel and structural event interpretation']}
    write_json(out/'audits/gd_provenance.json',provenance)
    report={'status':'BLOCKED' if hard else 'AUDITED_NOT_APPROVED',
        'hard_issues':hard,'warnings':warnings,
        'founders':founder_rows,'aliases':alias_rows,
        'non_snp':{'records':n_events,'by_type':dict(types),'by_classification':dict(reasons),
                   'clones_with_unresolved_events':len(unresolved_clones),
                   'interval_rows':len(mask_rows),
                   'mask_semantics':'EXCLUSION CANDIDATES ONLY; not a verified callable mask'},
        'temporal_status':temporal_status,'gd_provenance':'UNVERIFIED',
        'limitations':['Only SNP-resolved reference-aligned states available',
          'INS/MOB insertion sequence has no one-to-one reference coordinate mapping',
          'AMP/CON/INV extents and affected alleles need expert review',
          'A position outside exclusion intervals is NOT proven callable',
          'Full-tree-derived old FASTAs used only for reconciliation, never model features',
          'Future target labels are never loaded by this script'],
        'approval':'NO_TRAINING_AUTHORIZATION',
        'input_hashes':{'reference':sha_file(ref),'phase1_manifest':sha_file(manifest)}}
    write_json(out/'audits/phase1_2_report.json',report)
    print('PHASE 1.2',report['status'])
    print('Founders:',[(x['founder_id'],x['status']) for x in founder_rows])
    print('Aliases:',[(x['old_id'],x['status']) for x in alias_rows])
    print('Non-SNP records:',n_events,'types:',dict(types))
    print('Unresolved-event clones:',len(unresolved_clones),'exclusion intervals:',len(mask_rows))
    print('Hard issues:',len(hard),'warnings:',len(warnings))
    for x in hard[:8]:print('ISSUE',x)
    print('Saved:',out)
    print('DO NOT TRAIN: callability and GD provenance remain unverified.')
    if hard:raise SystemExit(2)

if __name__=='__main__':main()
