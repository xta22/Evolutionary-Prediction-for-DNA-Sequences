# V3-C Stage 2.4.1 — Oracle information intervention and regime boundary audit

This is a **descriptive 2x2 information intervention**, not a rigorous decomposition of Bayes risk.
It imports the original `v3c_stage24.py` and requires exact V3-B replay parity before scoring.

## Four cells

- sampled/estimated: bootstrap the most recent sampled haplotypes; use a crude historical allele-frequency-change proxy as symmetric flip rate and neutral reproduction.
- sampled/true: bootstrap sampled haplotypes; apply true V3-B mechanism. **Evaluator-only privileged information**.
- full/estimated: use exact cutoff full population; same crude estimated neutral mechanism.
- full/true: exact cutoff population and true V3-B mechanism.

The 'estimated' mechanism is intentionally a simple diagnostic proxy, not a scientifically validated estimator of mutation, selection, recombination or epistasis. Therefore differences between cells **cannot be interpreted as uniquely attributable to information alone**: model misspecification and bootstrap approximation also contribute. In particular, the sampled/true cell is not a Bayes-optimal predictor conditional on sampled genomes.

The script checks every original sampled genome by exact replay, reads restricted targets for scoring only, and computes pre/crossing/post shift status from actual generation steps. Use all folds to include post-shift intervals.

## WSL

```bash
cd ~/LTEE-Ecoli
unzip -o '/mnt/c/Users/xiaoyuetang.DESKTOP-09PBS3S/Downloads/ltee_v3c_stage241_oracle_decomposition.zip'
cp ltee_v3c_stage241/v3c_stage241.py ltee_v3/
cp ltee_v3c_stage241/test_v3c_stage241.py ltee_v3/
cd ltee_v3
python -m unittest -v test_v3c_stage241.py
OPENBLAS_NUM_THREADS=1 python -u v3c_stage241.py \
  --source v3c_stage24.py \
  --input ../LTEE_prospective/v3b_simulation \
  --output ../LTEE_prospective/v3c_stage241 \
  --seeds 42 --scenarios neutral regime_shift \
  --replicates 100
```

The run may be CPU-intensive. Start with `--replicates 20 --max-folds-per-run 2` for a smoke test, then run full folds to inspect shift boundaries. Do not tune models against these exploratory results and call them confirmatory.
