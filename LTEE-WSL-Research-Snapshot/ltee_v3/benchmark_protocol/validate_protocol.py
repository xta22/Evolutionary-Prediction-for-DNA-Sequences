#!/usr/bin/env python3
"""Validate protocol assignments and optional historical-only fold metadata. Never opens target files."""
import argparse,csv,hashlib,json
from pathlib import Path
from build_protocol import canonical,DEVELOPMENT

def read_csv(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def validate(directory,data_root=None):
    errors=[]; warnings=[]
    p=directory/'protocol.json';m=directory/'trajectory_split_manifest.csv'
    protocol=json.loads(p.read_text());manifest=read_csv(m)
    expected=hashlib.sha256(canonical(protocol)).hexdigest()
    digest=(directory/'protocol.sha256').read_text().split()[0]
    if digest!=expected:errors.append('protocol hash mismatch: protocol edited without re-freezing')
    groups=protocol['splits']; seen=set(); expected_ids=set()
    for split,seeds in groups.items():
        if split not in ('train','validation','locked_test'):errors.append('unknown split '+split)
        for seed in seeds:
            if seed in seen:errors.append(f'seed {seed} assigned across splits')
            seen.add(seed)
            for scenario in protocol['simulation']['scenarios']:expected_ids.add((scenario,str(seed),split))
    if set(groups['locked_test']) & set(DEVELOPMENT):errors.append('previously inspected seed in locked test')
    actual=set()
    for row in manifest:
        key=(row['scenario'],row['seed'],row['split'])
        if key in actual:errors.append('duplicate manifest assignment '+str(key))
        actual.add(key)
        if row['trajectory_id']!=f"{row['scenario']}:seed_{row['seed']}":errors.append('trajectory id mismatch')
    if actual!=expected_ids:errors.append('manifest does not match protocol split cartesian product')
    if protocol['approval']['locked_test_approved']:warnings.append('approval flag true: external review and data access still required')
    if data_root is not None:
        # Only check historical_inputs.csv and folds.csv; never open evaluation targets or simulator truth.
        for row in manifest:
            if row['split']=='locked_test':continue
            folder=data_root/row['scenario']/('seed_'+row['seed'])
            folds_file=folder/'folds.csv'; hist_file=folder/'historical_inputs.csv'
            if not folds_file.exists() or not hist_file.exists():
                warnings.append('missing development fold/history '+str(folder));continue
            folds=read_csv(folds_file); history=read_csv(hist_file)
            by_fold={}
            for h in history:by_fold.setdefault(h['fold_id'],[]).append(h)
            for f in folds:
                fid=f['fold_id']; cutoff=int(f['cutoff']); target=int(f['target_generation'])
                if target<=cutoff:errors.append(f'nonfuture target {folder}/{fid}')
                hh=by_fold.get(fid,[])
                if not hh:errors.append(f'missing history {folder}/{fid}');continue
                if any(int(h['generation'])>cutoff for h in hh):errors.append(f'future history {folder}/{fid}')
                if max(int(h['generation']) for h in hh)!=cutoff:errors.append(f'no cutoff observations {folder}/{fid}')
    return {'status':'FAIL' if errors else 'PASS_PROTOCOL_STRUCTURE_ONLY','errors':errors,'warnings':warnings[:30], 'n_assignments':len(manifest),'locked_test_approved':False}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol-dir',type=Path,required=True);p.add_argument('--data-root',type=Path)
    a=p.parse_args();result=validate(a.protocol_dir,a.data_root)
    print(json.dumps(result,indent=2));raise SystemExit(bool(result['errors']))
if __name__=='__main__':main()
