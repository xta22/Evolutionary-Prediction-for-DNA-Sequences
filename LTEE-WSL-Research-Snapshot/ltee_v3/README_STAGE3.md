# V3-C Stage 3: M4 observed-history latent-regime forecasting

## Scope
This is a **binary sampled-population forecasting** experiment, not a genome-scale LTEE model. It uses V3-B's `historical_inputs.csv`, `folds.csv`, and restricted future targets *only for evaluation/training labels*. It never opens simulator truth, mutation event records, or population statistics. Do not treat `M4_latent`'s gate as a validated mutator-state estimate.

## Models
- B1_latest: Laplace-smoothed cutoff sample frequency.
- M1_shared: trainable shared log-odds correction using horizon.
- M4_static: equal-weight mixture of two experts.
- M4_latent: two experts with a learned gate depending on cutoff-history allele-frequency volatility.
- M4_shuffled: negative control with the volatility gate feature permuted across folds.

The M4 mixture is deliberately modest. It can test whether an *observable* regime proxy improves marginal forecasts; it cannot anticipate a genuinely unannounced future mutation-rate shift, and cannot identify the biological state of a mutator from these features alone.

## Run
```
cd ~/LTEE-Ecoli
unzip -o /mnt/c/Users/xiaoyuetang.DESKTOP-09PBS3S/Downloads/ltee_v3c_stage3_m4.zip
cd ltee_v3
python -m unittest -v test_v3c_stage3.py
OPENBLAS_NUM_THREADS=1 python -u v3c_stage3.py \
  --input ../LTEE_prospective/v3b_simulation \
  --output ../LTEE_prospective/v3c_stage3_m4 \
  --train-seeds 11 23 --validation-seeds 42
```

By default, **no test seeds are evaluated**. Previously examined seeds 67/101 are not a blind test. A future confirmatory test requires fresh seeds, frozen code, and a preregistered evaluation rule. Optional `--test-seeds` is explicitly labeled exploratory.

## Read results
```
python - <<'PY'
import pandas as pd
p='../LTEE_prospective/v3c_stage3_m4/metrics.csv'
df=pd.read_csv(p)
print(df.groupby(['split','scenario','model'])['nll'].agg(['mean','std','count']).round(6).to_string())
PY
```

## Important lessons carried forward
1. Full-tree ancestral reconstruction can leak future descendants; only observed history is used.
2. Strong B1 baseline: compare against recent cutoff frequencies.
3. Trend extrapolation may overfit; gate uses low-dimensional historical summary and L2 penalty.
4. Weak M2/M3 gains were not a simple optimizer failure; no assumption that M4 must win.
5. Binary 96-site high-rate V3-B simulation is not calibrated to 4.63 Mb LTEE genomes.
6. Multiple folds from the same trajectory are correlated; report per-seed uncertainty before biological claims.
7. Neutral and linkage_only may be identical; don't count them as independent mechanism evidence.
8. No joint likelihood or epistasis identification; M5 is a separate next-stage task.
9. Validation seed 42 has been repeatedly inspected: all results exploratory.

## Limitations
The mixture experts may be non-identifiable (label switching), and the volatility proxy may reflect drift or sampling noise rather than mutation regimes. The `M4_shuffled` control disrupts the historical proxy without retraining on an alternate trajectory; interpret as sensitivity, not a formal causal attribution. The simulator uses binary haplotypes, elevated mutation probabilities, and dense sampling. No real-LTEE transfer claim is warranted.
