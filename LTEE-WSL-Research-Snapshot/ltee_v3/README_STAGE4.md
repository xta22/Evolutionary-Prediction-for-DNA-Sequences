# V3-C Stage 4: M5 pairwise haplotype forecasting

This is a **binary, two-site joint distribution** benchmark. It does not recover
biological epistasis and is not yet a trainable M5 dynamics model.

Models:
- M1_independent: product of latest observed site marginals.
- M5_empirical: latest observed joint two-site frequencies.
- M5_shrunk: mixture of joint frequencies and independent product.
- M5_dynamics: historical pair-frequency extrapolation with shrinkage.
- M5_shuffled: negative control permuting the second allele among cutoff clones,
  preserving each marginal but destroying observed within-clone association.

Pair list is fixed a priori (0-1, 2-3, ...) and includes the first two sites.
No pair selection using future labels. At each fold, only cutoff and earlier
observations are used for predictions. Future samples are used only to score.

**Important:** Train seeds are reserved, not fitted in this first prototype.
Hyperparameters are chosen using validation only. This is a protocol/benchmark
baseline rather than the full trainable M5 proposed in the research plan.

## Run on Quartz

```bash
cd ~/Evolutionary-Prediction-for-DNA-Sequences
V3="LTEE-WSL-Research-Snapshot/ltee_v3"
DATA="LTEE-WSL-Research-Snapshot/LTEE_prospective"
python -m unittest discover -s "$V3" -p 'test_stage4.py' -v
OPENBLAS_NUM_THREADS=1 python -u "$V3/stage4.py" \
  --input "$DATA/v3b_simulation" \
  --output "$DATA/v3c_stage4_m5"
```

Validation is seed 42. Test seeds are intentionally omitted. To evaluate
previously inspected seeds 67 and 101 for *exploratory* analysis only, add
`--test-seeds 67 101 --allow-exploratory-test`.

Outputs: metrics.csv, selected_models.json, protocol.json.

## Interpretation

Joint NLL assesses four binary pair states 00,01,10,11. Marginal NLL is
averaged over the two sites. Double-mutant MAE and LD-D MAE assess pairwise
statistics. This is not whole-genome joint likelihood, and two-site LD cannot
identify epistasis without additional causal controls.

No simulation_truth_RESTRICTED.json or mutation-event counter is loaded.
