# V3-C Stage 2: trainable binary forecasting baselines

Requires Python 3.10+, NumPy and SciPy. No torch.

## Models

- B1_latest: empirical last-observation baseline with alpha=5.
- M1_shared: shared learned log-odds calibration and horizon correction.
- M2_history: shared model using population historical change/volatility features, regularized across trajectories. **Not a fully hierarchical random-effects model.**
- M3_local_context: adds observable neighboring allele-frequency features. **Not Evo2 embeddings or true simulator context.**

All three learned models are binomial-likelihood regression models with an offset from the latest observed allele frequencies. This is a conservative *Stage 2 interface / trainability test*, **not** a CTMC or a mechanistic Wright-Fisher likelihood. It does not infer explicit selection, recombination, epistasis or mutator states.

## Run

```bash
cd ~/LTEE-Ecoli/ltee_v3
python -m unittest -v test_v3c_stage2.py
OPENBLAS_NUM_THREADS=1 python -u v3c_stage2.py \
  --input ../LTEE_prospective/v3b_simulation \
  --output ../LTEE_prospective/v3c_stage2 \
  --train-seeds 11 23 --validation-seeds 42 --test-seeds 67 101
```

The training stage reads historical fold inputs and evaluator-only target labels for **train trajectories**. Model selection reads validation labels. Test labels are used once after selection, solely to compute evaluation metrics. No hidden simulator truth, restricted mutation-event counts, or full-trajectory files are accessed. Individual `sample_index` values are **not** treated as ancestor-child matches.

### Important limits

The 67/101 trajectories have already been inspected in prior stages; results are exploratory. Overlapping folds within one trajectory are dependent. Report uncertainty by independent trajectory, not by fold. Same seed across scenarios may share initial conditions. Binary simulated haplotypes are not four-state nucleotide genomes. The M2/M3 names denote incremental feature families, not full biological mechanism models.

Results: `metrics.csv`, `selected_models.json`, `protocol.json`.
