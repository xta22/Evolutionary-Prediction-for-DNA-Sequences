# V3-C Stage 2.4: simulator-specific oracle gap

This evaluator replays the exact V3-B Wright-Fisher RNG sequence to recover **full cutoff population states**. It verifies all original sampled genomes for every generation before using any checkpoint. If parity fails, it stops without producing a comparison. This makes it deliberately dependent on the original V3-B kernel semantics.

The oracle starts from full cutoff population state, knows true selection, mutation, recombination and regime parameters, and independently simulates many futures to estimate marginal site probabilities. It does **not** see the realized future RNG draws. It is not a legitimate forecasting input, nor an exact analytic or joint-sequence oracle.

Run inside `~/LTEE-Ecoli/ltee_v3` with the original `../LTEE_prospective/v3b_simulation` present. CPU-only NumPy.

```bash
python -m unittest -v test_v3c_stage24.py
OPENBLAS_NUM_THREADS=1 python -u v3c_stage24.py \
  --input ../LTEE_prospective/v3b_simulation \
  --output ../LTEE_prospective/v3c_stage24_oracle \
  --seeds 42 --replicates 200 --max-folds-per-run 3
```

**Cautions:** this reconstructs hidden cutoff states from the *simulation seed*, for evaluator-only diagnostics. Do not expose checkpoints or mechanism truth to models. Only `B1_latest` reads historical samples. The V3-B data were already inspected, so this is exploratory. A true joint oracle and partial-information oracle require further work. If the V3-B simulator changed since the ZIP version, parity verification will fail; do not bypass it.
