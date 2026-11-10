# Phase 2C: Held-out Rare-event Forecasting

Requirements: Python 3.9+, numpy, pandas, scipy. All data generated synthetically. No LTEE files read.

Run tests: `python -m unittest -v test_phase2c.py`

Run: `OPENBLAS_NUM_THREADS=1 python -u phase2c.py --seeds 11 23 42 67 101 --output ../LTEE_prospective/phase2c_synthetic`

Outputs: heldout_metrics.csv, fit_diagnostics.json, config.json.

Scenarios: stationary historical rates vs unknown future rate shift. Train/test counts independently sampled. M1 global Poisson, M2 per-population effects with no penalty / L2 penalties 1, 10. A shared global intercept and centered population effects are fit only to training counts.

Important limitations: this is count-only, not nucleotide CTMC, not phylogenetic simulation, not epistasis or linkage. In the stationary scenario, training and test use the same underlying population rates; in rate_shift, test rates change unpredictably. This is an identifiability and out-of-distribution stress test, not proof of evolutionary forecasting on LTEE.
