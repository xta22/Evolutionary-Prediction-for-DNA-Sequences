# V3-C Stage 2.2: Production feature & gradient audit

This read-only companion imports your actual `v3c_stage2.py`. It never edits the model or writes into Stage 2 output. It uses validation seeds from `protocol.json` only. It never loads test seeds or `simulation_truth_RESTRICTED.json`.

## Outputs
- `feature_diagnostics.csv`: variance, coefficients, coefficient-times-feature magnitude.
- `prediction_diagnostics.csv`: per-fold probability differences between models.
- `permutation_validation.csv`: fixed-model site-row permutation of history and local-context blocks, **validation only**.
- `optimization_diagnostics.json`: objective and gradients computed on validation data, at zero and saved coefficients.

**Crucial:** these are NOT training gradient norms or a recorded optimizer trajectory. The script cannot recover that information from `selected_models.json`. For true training gradients, the training objective must be instrumented in a future controlled rerun. Validation gradient magnitude is not evidence of training convergence. Permutation sensitivity is not causal feature importance and correlated features can make it misleading.

The unit tests bundled here use a copy of the Stage 2 reference implementation. The real audit imports the installed `v3c_stage2.py` via `--source` and will fail on incompatible interfaces rather than fabricate results.

## Run
```
python -m unittest -v test_v3c_stage22.py
OPENBLAS_NUM_THREADS=1 python -u audit_v3c_stage22.py \
 --source v3c_stage2.py \
 --input ../LTEE_prospective/v3b_simulation \
 --models ../LTEE_prospective/v3c_stage2/selected_models.json \
 --protocol ../LTEE_prospective/v3c_stage2/protocol.json \
 --output ../LTEE_prospective/v3c_stage22_audit
```
