# V3-C Stage 2.3: Information Availability vs Model Capacity

Requires NumPy and SciPy; no PyTorch. Does not overwrite `v3c_stage2.py`.

## What this actually does

1. Imports the production Stage 2 `collect()` and `make_features()` via `collect()`.
2. Fits four regularized binomial-offset models: shared, history, local, and a fixed nonlinear expansion of observable features.
3. Selects regularization using **validation only**, evaluates **validation only**.
4. Reports training objective and training gradient norms at zero initialization and fitted parameters.
5. Reports training feature variation and per-feature contribution dispersion, and validation probability differences.
6. Runs an **independent synthetic oracle fixture** where the generating variable, a noisy observable proxy, and uninformative noise are known.

## Limitations

- This does **not** provide actual hidden site-specific simulator truth to any model. The oracle is a **separate synthetic fixture**, not a direct upper bound on V3-B.
- The nonlinear expansion is fixed (squares and selected interactions), not a learned MLP. It tests a restricted capacity increase.
- Same simulation trajectories may share dependence across folds. Treat as exploratory.
- No test seeds or restricted truth files are loaded. Never tune models on previously examined test seeds.
- Do not infer that true biological sequence context is available from neighboring allele-frequency features.

## Commands

From `~/LTEE-Ecoli/ltee_v3`:

```bash
python -m unittest -v test_v3c_stage23.py
OPENBLAS_NUM_THREADS=1 python -u v3c_stage23.py \
  --source v3c_stage2.py \
  --input ../LTEE_prospective/v3b_simulation \
  --output ../LTEE_prospective/v3c_stage23_audit
```

Outputs: `capacity_validation.csv`, `feature_contributions.csv`, `prediction_deltas_validation.csv`, `oracle_fixture.json`, `protocol.json`.
