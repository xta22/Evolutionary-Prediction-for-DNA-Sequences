# V3-C Stage 1 — Independent-trajectory forecasting baseline audit

This is a deliberately scoped **first stage**, NOT the full M1–M5 comparison.

## Why these baselines?

The V3-B observations are *samples of populations*, not matched parent–child
individuals. Pairing row `sample_index=0` at two time points would invent ancestry.
This script instead predicts the probability of allele 1 at each site in future
sampled individuals, conditional on sampled historical populations.

- B0_global: global historical allele frequency.
- B1_latest: shrinkage-smoothed allele frequency of latest historical sample.
- B2_shrunk_trend: pooled trend correction.
- B3_site_trend: site-wise trend correction.

They are **not** four-state CTMC models, nor M4/M5. Their purpose is to establish
leakage-safe forecasting/evaluation plumbing on the actual V3-B output.

## Data contract

Read only `folds.csv`, `historical_inputs.csv`, and
`evaluation_targets_RESTRICTED.csv` under each run. Future targets are read only
by evaluation; no `observed_genomes_RESTRICTED.csv` or simulator truth is loaded.
Disjoint trajectory seeds are required. Hyperparameters are chosen on validation
seeds only. Train seeds are reserved for future learned models, not used here.

## Run

```bash
cd ~/LTEE-Ecoli/ltee_v3
python -m unittest -v test_v3c.py
OPENBLAS_NUM_THREADS=1 python -u v3c_forecast.py \
  --input ../LTEE_prospective/v3b_simulation \
  --output ../LTEE_prospective/v3c_stage1
```

The default seed groups are train=11,23; validation=42; test=67,101.
These seeds have already been inspected in V3-B mechanism audits; the result is
**exploratory**, not an untouched confirmatory test.

## Next stage

Implement and test the true four-state M1–M3, latent-regime M4, and joint M5 on
appropriately generated four-state forward trajectories, with model-specific
negative controls and a fresh, untouched trajectory test split.
