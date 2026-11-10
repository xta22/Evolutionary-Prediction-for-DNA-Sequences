# Phase 2B: Synthetic parameter recovery, shrinkage ablation, rare-event stress test

This is **synthetic-only**. It does not read LTEE data, future labels, or Evo 2 embeddings.

## Requirements
`numpy`, `scipy` (Python 3.9+).

## Tests
`python -m unittest -v test_phase2b.py`

## Run
`OPENBLAS_NUM_THREADS=1 python -u phase2b.py --seeds 11 23 42 --output ../LTEE_prospective/phase2b_synthetic`

Start with a smoke test:
`OPENBLAS_NUM_THREADS=1 python -u phase2b.py --seeds 42 --train-n 30 --validation-n 20 --test-n 20 --maxiter 20 --output ../LTEE_prospective/phase2b_smoke`

## Outputs
- `recovery_metrics.csv`: M1 vs M2 unregularized vs M2 validation-selected shrinkage; predictive NLL, transition RMSE, centered population-rate recovery.
- `recovery_fits.json`: fitted parameters and ground truth for auditing.
- `rare_metrics.csv`: rare final-state changes with genome-wide counts, count-likelihood recovery, penalty stress tests.
- `config.json`: all run settings.

## Important limitations
- CTMC is correctly used for per-site synthetic transitions in the recovery study. The rare-event study fits a Poisson **approximation** to independent CTMC final-state change counts; it is not a complete rare-event four-state CTMC training experiment.
- Rare rates are *per synthetic time unit*, not per LTEE generation; horizon defaults to 1.
- Population effects in M2 have an intercept confounding: report centered effects and transition probabilities, not absolute effect coefficients as uniquely identified.
- The `rate_spectrum` environment is within the broad M2 model family; advantage there is expected, not independent evidence of real biological mechanism.
- Very low rates may yield zero counts and non-identifiability. The output records these cases; do not interpret high correlations from tiny numbers of events.
- The 3 seeds are illustrative, not statistical confirmation. No real-data training is authorized by this package.
