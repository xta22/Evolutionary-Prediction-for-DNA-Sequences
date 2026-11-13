# V3-B Forward Simulation: Mechanism Validation

Standalone NumPy forward-time haploid Wright-Fisher simulator. Does not modify V3-A `v3_simulate.py`.

**Models and interpretation**
- Binary haplotypes, not full A/C/G/T genomes. Mutation is a reversible bit flip, so this is a mechanism test, not a full biological LTEE simulation.
- `neutral` and `linkage_only`: no recombination, neutral drift. They intentionally have identical mechanisms; compare `linkage_only` against `recombination`.
- `recombination`: with probability `--recombination-probability` per offspring, copy suffix from independently sampled second parent after uniformly chosen crossover. This is not a per-base recombination rate.
- `selection_only`: selected allele at site L/3 with log fitness effect `s`.
- `epistasis`: same selection plus `epsilon * allele_a * allele_b` on log fitness. This includes selection; comparison to selection-only isolates interaction in the simulator.
- `context`: fixed site-specific relative mutation susceptibilities (normalized mean one).
- `regime_shift`: step change in mutation rate after specified generation.
- `heterogeneous`: a rate multiplier for the whole trajectory; across-seed heterogeneity is not explicitly population-randomized in this version. Do not claim it validates population-level heterogeneity.

**No leakage:** historical_inputs.csv is per-fold restricted. Files marked RESTRICTED include future sequences or simulation truth and MUST NOT be passed to model training. The directory is not an access-control boundary; enforce separation in downstream loaders.

**Run**
```
python -m unittest -v test_v3b.py
OPENBLAS_NUM_THREADS=1 python -u v3b_simulate.py --seeds 11 23 42 67 101 --output ../LTEE_prospective/v3b_simulation
python v3b_validate.py --input ../LTEE_prospective/v3b_simulation/run_summary.csv
```

**Outputs:** run_summary.csv, per scenario/seed observations, mutation event counts, population statistics, chronological folds, historical inputs, restricted targets and truth, mechanism diagnostics. `v3b_validate.py` reports descriptive paired contrasts. Use many seeds, power analyses, and dedicated ground-truth controls before claiming biological mechanism recovery. Sampled-allele frequencies and LD are noisy and can be undefined after fixation.
