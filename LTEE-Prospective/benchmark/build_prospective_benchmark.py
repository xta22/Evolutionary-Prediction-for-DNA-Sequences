#!/usr/bin/env python3
"""Read-only source audit + prospective SNP-state benchmark manifests.
No reconstructed ancestral sequences, full-tree tip selection, or future-derived training features.
Requires an explicit clone-local GenomeDiff directory and a reference GenBank file.
"""
import argparse, csv, hashlib, json, re
from collections import Counter, defaultdict
from pathlib import Path
from Bio import SeqIO

PAT = re.compile(r'^(?P<population>Ara[+-][1-6])_(?P<generation>\d+)gen_(?P<clone>.+)\.gd$')
DEFAULT_FOLDS = [(10000,15000),(10000,20000),(20000,30000),(30000,40000),(40000,50000)]

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)

def read_gd(path, length, allowed_refs, reference):
    changes={};non_snp=Counter();errors=[];records=0
    with path.open() as f:
        for lineno,line in enumerate(f,1):
            s=line.strip()
            if not s or s.startswith('#'):continue
            fields=s.split()
            kind=fields[0]
            if kind!='SNP':
                non_snp[kind]+=1;continue
            records+=1
            if len(fields)<6:
                errors.append(f'line {lineno}: malformed SNP');continue
            refid=fields[3]
            try:pos=int(fields[4])
            except ValueError:
                errors.append(f'line {lineno}: invalid position');continue
            base=fields[5].upper()
            if refid not in allowed_refs:
                errors.append(f'line {lineno}: unrecognized reference {refid}');continue
            if not 1<=pos<=length or base not in 'ACGT':
                errors.append(f'line {lineno}: invalid coordinate/base');continue
            if pos in changes:
                errors.append(f'line {lineno}: duplicate SNP position {pos}');continue
            if reference[pos-1]==base:
                errors.append(f'line {lineno}: SNP equals reference at {pos}');continue
            changes[pos]=base
    return changes,non_snp,errors,records

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gd-dir',type=Path,required=True,help='Directory of clone-local .gd files (not tree-derived FASTAs)')
    p.add_argument('--reference',type=Path,default=Path('reference/REL606.gbk'))
    p.add_argument('--out',type=Path,default=Path('LTEE_prospective'))
    p.add_argument('--reference-id',action='append',default=[],help='Explicit GD reference ID, repeatable; must be verified externally')
    p.add_argument('--strict-non-snp',action='store_true',help='Reject clone if any non-SNP records are present')
    args=p.parse_args()
    if not args.gd_dir.is_dir():p.error('GD directory does not exist')
    if not args.reference.is_file():p.error('Reference GenBank does not exist')
    rec=SeqIO.read(args.reference,'genbank');reference=str(rec.seq).upper()
    allowed={rec.id,rec.name,*args.reference_id}
    files=sorted(args.gd_dir.glob('*.gd'))
    if not files:p.error('No .gd files found')
    rows=[];audits=[];states=[];unparsed=[];counts=Counter(); seen=set()
    for file in files:
        match=PAT.fullmatch(file.name)
        if not match:
            unparsed.append(file.name);continue
        g=match.groupdict();clone_id=file.stem
        if clone_id in seen:raise RuntimeError(f'duplicate clone id {clone_id}')
        seen.add(clone_id)
        changes,other,errors,nrec=read_gd(file,len(reference),allowed,reference)
        if args.strict_non_snp and other:errors.append('non-SNP variants present')
        status='PASS' if not errors else 'FAIL'
        counts[status]+=1
        rows.append(dict(clone_id=clone_id,population=g['population'],generation=int(g['generation']),gd_path=str(file.resolve()),gd_sha256=sha256(file),reference_sha256=sha256(args.reference),snp_count=len(changes),non_snp_count=sum(other.values()),status=status))
        audits.append(dict(clone_id=clone_id,status=status,errors=errors,non_snp=dict(other),snp_records=nrec,unique_snps=len(changes)))
        if status=='PASS':
            for pos,base in sorted(changes.items()):states.append(dict(clone_id=clone_id,position_1based=pos,alternate_base=base))
    out=args.out
    write_csv(out/'manifests/observed_clone_manifest.csv',rows,['clone_id','population','generation','gd_path','gd_sha256','reference_sha256','snp_count','non_snp_count','status'])
    write_csv(out/'data/snp_state_store.csv',states,['clone_id','position_1based','alternate_base'])
    with (out/'audits/variant_integrity.json').open('w') as f:json.dump({'reference_id':rec.id,'reference_length':len(reference),'allowed_reference_ids':sorted(allowed),'status_counts':dict(counts),'unparsed_files':unparsed,'per_clone':audits},f,indent=2)
    valid=[r for r in rows if r['status']=='PASS']; bypop=defaultdict(list)
    for r in valid:bypop[r['population']].append(r)
    folds=[];issues=[]
    for cutoff,target in DEFAULT_FOLDS:
        for pop in sorted(bypop):
            hist=[r for r in bypop[pop] if r['generation']<=cutoff]
            future=[r for r in bypop[pop] if r['generation']==target]
            if not hist or not future:
                issues.append({'population':pop,'cutoff':cutoff,'target':target,'reason':'missing history or future clones'})
                continue
            # Explicitly no future clone ID, genome, or mutation content in historical features.
            for r in future:
                folds.append(dict(population=pop,cutoff=cutoff,target_generation=target,horizon=target-cutoff,target_clone_id=r['clone_id'],historical_clone_count=len(hist),historical_max_generation=max(x['generation'] for x in hist),fold_status='CANDIDATE_NOT_LOCKED'))
    write_csv(out/'splits/rolling_origin_folds.csv',folds,['population','cutoff','target_generation','horizon','target_clone_id','historical_clone_count','historical_max_generation','fold_status'])
    with (out/'audits/temporal_integrity.json').open('w') as f:json.dump({'folds':len(folds),'missing_folds':issues,'rules':['historical_generation <= cutoff','target_generation > cutoff','target IDs for evaluation only','no full-tree reconstruction'],'status':'CANDIDATE_ONLY'},f,indent=2)
    provenance={'status':'UNVERIFIED','source':'clone-local GenomeDiff files','reference':str(args.reference.resolve()),'gd_directory':str(args.gd_dir.resolve()),'limitations':['GD variant calling provenance not independently verified','SNP-only states: indels/structural changes not represented','non-SNP affected sites are NOT masked','full-tree selection avoided by scanning GD files','GD file naming and sample generation require independent metadata verification','future fold targets not independent across overlapping folds','No future test period locked; do not use folds for final claims']}
    with (out/'audits/provenance_report.json').open('w') as f:json.dump(provenance,f,indent=2)
    print('REFERENCE',rec.id,len(reference),'GD files',len(files),'parsed',len(rows))
    print('STATUS',dict(counts),'unparsed',len(unparsed),'candidate forecast rows',len(folds))
    print('PROVENANCE UNVERIFIED: do not train or claim prospective validity yet')
    print('Saved',out.resolve())
    if counts['FAIL'] or unparsed:print('WARNING: failed/unparsed GD records; inspect audits before proceeding')

if __name__=='__main__':main()
