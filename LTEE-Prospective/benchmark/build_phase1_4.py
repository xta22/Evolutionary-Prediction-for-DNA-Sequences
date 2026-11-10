#!/usr/bin/env python3
"""Phase 1.4: fail-closed prospective benchmark specification (no model training)."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

FOLDS = 'splits/rolling_origin_folds.csv'
MANIFEST = 'manifests/observed_clone_manifest.csv'

def read_csv(p):
    with p.open(newline='') as f:
        return list(csv.DictReader(f))

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()

def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')

def integer(value):
    n=float(value)
    if not n.is_integer(): raise ValueError('noninteger')
    return int(n)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase1',default='LTEE_prospective')
    ap.add_argument('--out',default=None)
    args=ap.parse_args()
    root=Path(args.phase1).resolve()
    out=Path(args.out).resolve() if args.out else root/'phase1_4'
    fpath=root/FOLDS; mpath=root/MANIFEST
    if not fpath.exists() or not mpath.exists(): ap.error('Phase 1 manifest or fold CSV missing')
    folds=read_csv(fpath); manifest=read_csv(mpath)
    errors=[]; warnings=[]; fold_stats=Counter(); seen=set()
    for i,r in enumerate(folds):
        try:
            cutoff=integer(r['cutoff']); target=integer(r['target_generation']); horizon=integer(r['horizon']); hist=integer(r['historical_max_generation'])
            assert hist<=cutoff<target and target-cutoff==horizon
            assert r['population'].strip()
            assert r['target_clone_id'].strip()
            key=(r['population'],cutoff,target,r['target_clone_id'])
            if key in seen: errors.append(f'duplicate fold target row {i}: {key}')
            seen.add(key); fold_stats[f'{cutoff}->{target}']+=1
        except (ValueError,KeyError,AssertionError,TypeError): errors.append(f'invalid fold row {i}')
    idcol=next((k for k in ('clone_id','sample_id','id') if manifest and k in manifest[0]),None)
    if not idcol: errors.append('manifest clone ID column missing')
    else:
        ids=[r[idcol] for r in manifest]
        if len(ids)!=len(set(ids)): errors.append('duplicate canonical clone IDs in Phase 1 manifest')
        idset=set(ids)
        for i,r in enumerate(folds):
            if r.get('target_clone_id') not in idset: errors.append(f'fold target not in manifest at row {i}')
    upstream=root/'phase1_3_patched'/'audits'/'benchmark_integrity_report.json'
    if not upstream.exists(): upstream=root/'phase1_3'/'audits'/'benchmark_integrity_report.json'
    previous={}
    if upstream.exists():
        previous=json.loads(upstream.read_text())
        if previous.get('hard_issues'): errors.append('Phase 1.3 has unresolved hard issues')
    else: warnings.append('Phase 1.3 report not found')
    policy={
      'version':'1.4.0','task':'future sampled-clone SNP-resolved reference-state distribution',
      'scope':'REL606-coordinate A/C/G/T states only; structural variation not predicted',
      'forecast_unit':'population x cutoff x target generation; target clone IDs evaluator-only',
      'history':'all eligible observed clone records with generation <= cutoff',
      'origin_policy':'latest observed generation, uniform mixture of available clone origins; mixture is not a population-frequency estimate',
      'future_target_policy':'future sampled clone, not assumed to descend directly from any origin clone',
      'information_boundary':'training, preprocessing, feature selection and hyperparameter fitting must use only data available by the forecast cutoff',
      'evaluation_tracks':{
         'snp_only_exploratory':'SNP-only reference representation; not full genome, non-SNP and coverage limitations disclosed',
         'structural_exclusion_sensitivity':'candidate intervals evaluated only by evaluator after future observation; not a verified callable mask',
         'strict_callable':'BLOCKED until independently validated coverage and structural-event masks exist'
      },
      'metrics':['full-base log_loss vs identity baseline','mutation AUROC/AUPRC','precision_at_k','observed_vs_predicted_changed_sites','conditional substitution log_loss on changed sites'],
      'metric_notes':'SNP-only state comparisons are not direct lineage mutation counts; use full-genome or probability-correct sampling for rate calibration',
      'development_rule':'Only historical inner folds may select architecture and hyperparameters; candidate future folds are exploratory until pre-registered',
      'confirmatory_rule':'Not locked; no test performance claims or test-driven model selection',
      'provenance':'GD curation time-causality unverified',
      'model_training_authorized':False
    }
    mask={'version':'1.4.0','strict_callable':'NOT_AVAILABLE','candidate_reference_exclusions':'NOT_CERTIFIED',
          'future_target_masks':'evaluation-only; never fed to predictor or preprocessing fit',
          'INS_MOB':'anchor annotations do not establish full affected extent',
          'AMP_CON_INV':'complex events require event-specific review',
          'circular_intervals':'candidate coordinate splits only, not verified event resolution',
          'coverage':'not assessed','recommended_use':'exploratory SNP-only sensitivity only; no strict callability claims'}
    protocol={'version':'1.4.0','folds':dict(sorted(fold_stats.items())),'fold_row_count':len(folds),
              'development':'historical-only inner temporal validation; future targets evaluator-only',
              'locked_test':'NOT_LOCKED','rolling_origin':'refit at cutoff only with <= cutoff observations; hyperparameter policy fixed from prior history',
              'target_id_feature_allowed':False,'training_authorized':False}
    evidence={'version':'1.4.0','source_files':{str(p.relative_to(root)):sha(p) for p in (fpath,mpath)},
              'phase13_report':str(upstream) if upstream.exists() else None,
              'phase13_report_sha256':sha(upstream) if upstream.exists() else None,
              'gd_provenance':'SOURCE_TRACEABLE_ONLY; upstream calling and curation time-causality unverified',
              'coverage_based_callability':'UNVERIFIED',
              'founder_and_alias':'See Phase 1.2 and patched Phase 1.3 reports',
              'reference_coordinate_coverage':'SNP-only; non-SNP events incompletely represented'}
    report={'status':'BLOCKED' if errors else 'SPECIFIED_NOT_CERTIFIED','errors':errors,'warnings':warnings,
            'manifest_rows':len(manifest),'fold_rows':len(folds),'fold_counts':dict(fold_stats),
            'gd_provenance_certified':False,'strict_callable_mask_certified':False,
            'confirmatory_test_locked':False,'training_authorized':False,
            'next_actions':['Verify GD upstream curation provenance','Specify coverage/structural callability or keep explicit SNP-only exploratory track','Pre-register historical validation and locked final test']}
    save(out/'benchmark_spec.json',policy)
    save(out/'evaluation_mask_policy.json',mask)
    save(out/'locked_protocol.json',protocol)
    save(out/'provenance_evidence.json',evidence)
    save(out/'benchmark_readiness_report.json',report)
    print('PHASE 1.4',report['status'])
    print('Manifest rows:',len(manifest),'Forecast target rows:',len(folds),'Fold groups:',len(fold_stats))
    print('Errors:',errors[:10]);print('Warnings:',warnings[:10])
    print('Training authorized: False (by design)')
    print('Saved:',out)
    if errors: raise SystemExit(2)

if __name__=='__main__':main()
