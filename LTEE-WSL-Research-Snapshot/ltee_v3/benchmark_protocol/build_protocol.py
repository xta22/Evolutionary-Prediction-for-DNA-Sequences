#!/usr/bin/env python3
"""Create an immutable-intent protocol and trajectory assignments, without reading future labels."""
import argparse, csv, hashlib, json
from pathlib import Path

SCENARIOS = ('context','epistasis','heterogeneous','linkage_only','neutral','recombination','regime_shift','selection_only')
DEVELOPMENT = (11,23,42,67,101)

def canonical(obj):
    return json.dumps(obj,sort_keys=True,separators=(',',':')).encode()

def build(output, train_seeds, validation_seeds, locked_seeds, scenarios=SCENARIOS):
    groups = {'train':list(train_seeds),'validation':list(validation_seeds),'locked_test':list(locked_seeds)}
    if any(not v for v in groups.values()): raise ValueError('each split must have at least one trajectory seed')
    flat=[s for v in groups.values() for s in v]
    if len(flat)!=len(set(flat)): raise ValueError('overlapping seeds between splits')
    if any(s in DEVELOPMENT for s in groups['locked_test']): raise ValueError('previously inspected seed assigned to locked test')
    if len(set(scenarios))!=len(scenarios) or not scenarios: raise ValueError('invalid scenario list')
    output.mkdir(parents=True,exist_ok=True)
    manifest=[{'scenario':sc,'seed':seed,'trajectory_id':f'{sc}:seed_{seed}','split':split,'data_status':'NOT_GENERATED_OR_UNVERIFIED'} for sc in scenarios for split,seeds in groups.items() for seed in seeds]
    protocol={
      'protocol_version':'V3-C-4.1B-0.1','status':'DRAFT_LOCK_CANDIDATE_NOT_CERTIFIED',
      'task':{'unit':'future randomly sampled individual, binary two-site joint haplotype','inputs':'observed genomes at generations <= cutoff; horizon; fixed site-pair indices','target':'four probabilities P00,P01,P10,P11 at target generation','forbidden_inputs':['future genomes as features','future mutation events as features','simulator hidden states as features','future-descendant ancestral reconstructions']},
      'simulation':{'scenarios':list(scenarios),'generator':'v3b_simulate.py','generator_version_sha256':'UNVERIFIED','configuration_sha256':'UNVERIFIED','simulation_parameters':'MUST_FREEZE_BEFORE_LOCKED_TEST_GENERATION'},
      'splits':groups,'previously_inspected_seeds':list(DEVELOPMENT),
      'pair_selection':{'method':'adjacent_nonoverlapping','formula':'(0,1),(2,3),...','selected_without_target_labels':True},
      'model_comparison':['M1_independent','M5_fixed_shrinkage','M5_learned_global','M5_horizon_aware','M5_history_aware','M5_matched_shuffled'],
      'primary_metric':{'name':'joint_nll','unit':'nats per pair per future sampled individual','aggregation':'mean within trajectory then mean across independent trajectories; report scenarios separately'},
      'secondary_metrics':['joint_brier','double_mutant_mae','ld_D_mae','marginal_nll'],
      'training_policy':{'train':'fit model parameters using training trajectories and their future labels as supervised targets','validation':'select model and regularization; no test access','locked_test':'do not inspect or generate until code, configuration, and analysis plan are frozen; one final evaluation','shuffle':'match learned parameters and shrinkage; permute within-cutoff pairing while preserving marginal counts'},
      'limitations':['binary two-site task, not four-state genome','overlapping folds are correlated','neutral and linkage_only can be identical; do not count as independent mechanisms','locked test is not certified by writing this file','oracle requires separate mathematically justified implementation'],
      'approval':{'locked_test_approved':False,'human_review_required':True}}
    digest=hashlib.sha256(canonical(protocol)).hexdigest()
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    with (output/'trajectory_split_manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
    (output/'protocol.sha256').write_text(digest+'  protocol.json (canonical JSON digest)\n')
    return digest,len(manifest)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train-seeds',type=int,nargs='+',default=[11,23,67,101])
    p.add_argument('--validation-seeds',type=int,nargs='+',default=[42])
    p.add_argument('--locked-seeds',type=int,nargs='+',default=[1001,1003,1007,1009,1013])
    a=p.parse_args(); digest,n=build(a.output,a.train_seeds,a.validation_seeds,a.locked_seeds)
    print('PROTOCOL DRAFT CREATED',n,'trajectory assignments','sha256',digest)
    print('LOCKED TEST NOT APPROVED; simulation configuration and source hashes unverified')
if __name__=='__main__':main()
