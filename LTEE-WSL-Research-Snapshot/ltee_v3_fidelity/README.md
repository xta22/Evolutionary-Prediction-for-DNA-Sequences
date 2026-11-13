# V3-C Stage 2.5: Simulation Fidelity Audit (read-only)

This is a **descriptive, non-comparability-aware** audit. It does not certify that V3-B synthetic binary haplotypes resemble real LTEE 4-state genomes.

Requirements: Python 3.10+, NumPy, Pandas, Biopython (for GenBank reference). No PyTorch.

Run from `~/LTEE-Ecoli/ltee_v3` after copying both Python scripts there:

```bash
python -m unittest -v test_fidelity.py
python audit_simulation_fidelity.py \
  --gd-dir ../LTEE-clone-curated \
  --reference ../reference/REL606.gbk \
  --simulation ../LTEE_prospective/v3b_simulation \
  --output ../LTEE_prospective/v3c_stage25_fidelity
```

Outputs: `real_clone_snp_summary.csv`, `real_temporal_summary.csv`, `simulation_sample_summary.csv`, `simulation_temporal_summary.csv`, `sampling_intervals.csv`, `fidelity_report.json`.

Important: clone SNP counts are accumulated reference-relative calls, **not** mutation events between sampled clones. Binary simulated site diversity is **not** comparable to 4-state nucleotide mutation spectrum. No automatic comparison metric is calculated because such a number would be scientifically misleading. Original input data are never modified. Real GD file provenance and callable masks remain uncertified. Results are exploratory.
