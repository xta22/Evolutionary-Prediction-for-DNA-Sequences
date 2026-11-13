# LTEE Prospective Phase 1.4 + M0/M1 Synthetic Tests

## Purpose

- `build_phase1_4.py` creates a **fail-closed** benchmark specification and evidence manifest from existing Phase 1 outputs.
- `ctmc_baselines.py` implements an identity predictor and a four-state shared CTMC probability engine.
- `test_synthetic_m0_m1.py` tests the CTMC implementation on synthetic examples, including simulated transition recovery.

**Nothing here trains on LTEE or authorizes empirical model training.** No output is a certified callable mask or a prospective-validity certificate. The synthetic recovery test checks numerical behavior, not biological mechanism recovery.

## WSL usage

```bash
cd ~/LTEE-Ecoli
mkdir -p prospective_phase14_code
cp /mnt/c/Users/xiaoyuetang.DESKTOP-09PBS3S/Downloads/ltee_phase1_4_m0_m1.zip .
unzip -o ltee_phase1_4_m0_m1.zip -d prospective_phase14_code
python prospective_phase14_code/build_phase1_4.py --phase1 LTEE_prospective
python prospective_phase14_code/test_synthetic_m0_m1.py
```

Dependencies: Python 3.9+, `numpy`, `scipy`, `biopython` is **not** required for these scripts.

## Outputs

`LTEE_prospective/phase1_4/`: `benchmark_spec.json`, `evaluation_mask_policy.json`, `locked_protocol.json`, `provenance_evidence.json`, `benchmark_readiness_report.json`.

The benchmark remains **not certified** because upstream GenomeDiff curation time-causality and reference-coordinate callability are unverified. Forecast target IDs remain evaluation-only.
