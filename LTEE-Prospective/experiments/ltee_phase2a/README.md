# Phase 2A: Synthetic CTMC recovery (no real LTEE data)

This is an **exploratory correctness and ablation harness**, not a prospective LTEE model evaluation.

- M0: identity predictor (deterministic; log loss is large when change occurs).
- M1: shared 4x4 CTMC with one global log rate and six reverse-complement substitution classes.
- M2: M1 plus population log-rate and six-class deviations, with L2 shrinkage selected on an independent synthetic validation set.
- Scenarios: homogeneous, heterogeneous rate, heterogeneous rate+spectrum, and low-sample heterogeneity.
- Each seed/scenario has independent train/validation/test draws from the same ground-truth population parameters. Validation selects shrinkage; test is only evaluated after selection.
- Outputs: metrics.csv, fits.json, experiment.json. `fits.json` includes ground truth for M2 to permit later parameter recovery analysis.

## Important scientific caveats

This is **not** a full joint sequence simulator. Sites are sampled independently conditional on a population's CTMC. There is no linkage, selection, epistasis, phylogeny, genome sequence, Evo 2, or negative binomial count process. The time unit is arbitrary. M2 has redundant parameterization in the unpenalized model (global vs population rates and spectrum offsets), so raw coefficients should not be interpreted as identifiable biological parameters. Shrinkage stabilizes estimation but does not automatically identify them. Predictions are the primary benchmark. Rates are deliberately high enough to make a fast CPU smoke test informative; this does not mimic real LTEE prevalence. For realistic rare-event simulation, substantially more samples and an event-based implementation are required.

## WSL

```bash
cd ~/LTEE-Ecoli
unzip -o '/mnt/c/Users/xiaoyuetang.DESKTOP-09PBS3S/Downloads/ltee_phase2a_synthetic.zip' -d prospective_phase2a_code
python -m pip install numpy scipy
cd prospective_phase2a_code
python -m unittest -v test_phase2a.py
python -u phase2a.py --seeds 11 23 42
```

## Quick summary

```bash
python - <<'PY'
import pandas as pd
x=pd.read_csv('LTEE_prospective/phase2a_synthetic/metrics.csv')
print(x[x.split=='test'].groupby(['scenario','model'])[['nucleotide_nll','mutation_brier']].agg(['mean','std']).round(4))
PY
```

The output directory above is relative to the working directory. When running from `prospective_phase2a_code`, results will be created inside that directory. To put them in the main LTEE project, pass `--output ../LTEE_prospective/phase2a_synthetic`.
