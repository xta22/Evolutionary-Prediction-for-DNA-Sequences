# V3-C Stage 4.2C.5B — Sampling-depth pilot

This pilot varies the number of genomes sampled per time point while preserving the leakage-safe sequence: fit rho on train seeds, choose alpha/mix on calibration seeds, then score frozen choices on disjoint confirmation seeds.

## Run in the WSL project repository

The repository must already contain `v3b_simulate.py` and `stage42c5a1.py` beside `stage42c5b.py` in `LTEE-WSL-Research-Snapshot/ltee_v3/`.

```bash
V3="LTEE-WSL-Research-Snapshot/ltee_v3"
OUT="$HOME/LTEE-Ecoli/LTEE_prospective/v3c_stage42c5b"
python -u "$V3/stage42c5b.py" --reference "$V3/v3b_simulate.py" --output "$OUT"
```

Default sample sizes: 2, 4, 8, 16, 32. Each split has 20 seeds: train 1001–1020, calibration 2001–2020, confirmation 3001–3020. Scenarios, cutoffs, horizons, and smoothing grid match C.5A.1. Each depth is calibrated separately. Seed-level bootstrap intervals treat seeds as the replicate units rather than treating correlated cutoff/horizon rows as independent.

## Pilot outcome

Mean paired gain is `independent calibrated NLL − M5 calibrated NLL`; positive values favor the dependence correction. Across scenarios, the mean gain was 0.0057, 0.0140, 0.0250, 0.0314, and 0.0370 at sample sizes 2, 4, 8, 16, and 32, respectively. The seed-bootstrap 95% intervals were [0.0022, 0.0096], [0.0098, 0.0184], [0.0201, 0.0299], [0.0263, 0.0364], and [0.0317, 0.0423].

This is development evidence on the toy simulator, not a locked test or a validation of the full M5 History model. `dependence_off` sets rho to zero; it is not a pairing permutation.
