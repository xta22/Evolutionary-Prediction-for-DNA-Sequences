# V3-B.1 Targeted Mechanism & Power Audit

Standalone NumPy/Pandas simulator for targeted mechanism controls. **Does not replace or patch V3-B**, and does not independently certify that `v3b_simulate.py` implements the same rules. Compare both implementations before using these findings as validation of V3-B.

- Two-locus haplotype Wright-Fisher selection and recombination.
- Additive vs positive/negative epistasis at fixed additive fitness coefficients.
- Weak selection and midpoint outcomes to reduce fixation ceiling effects.
- Direct fitness interaction identity test.
- Site-context mutation-count enrichment and population-rate heterogeneity diagnostics.
- Per-seed outputs and time trajectories; shared starting populations but subsequent RNG streams not perfectly coupled.

## Run

```bash
cd ~/LTEE-Ecoli/ltee_v3
python -m unittest -v test_v3b1.py
OPENBLAS_NUM_THREADS=1 python -u v3b1_audit.py --output ../LTEE_prospective/v3b1_audit
```

Output: `mechanism_metrics.csv`, `allele_trajectories.csv`, `audit_report.json`.

Caution: these are exploratory descriptive tests, not a full power study or causal identification of epistasis. LD can be undefined when alleles fix. To establish V3-B implementation validity, add integration tests using V3-B's own transition and fitness functions. For stronger inference, expand to 100+ seeds and confidence intervals.
