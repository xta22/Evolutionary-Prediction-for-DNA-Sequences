# V3-C Stage 1.2 — Production Trend Intervention

This audit **imports the actual WSL `v3c_forecast.py`** and calls its `predict`, `parse_history`, and `score` functions. It does not replace or edit the production script. The bundled `v3c_forecast_reference.py` is solely for self-contained tests and corresponds to the previously distributed Stage 1 version; the real audit uses `--source v3c_forecast.py`.

## Run in WSL

```bash
cd ~/LTEE-Ecoli
unzip -o '/mnt/c/Users/xiaoyuetang.DESKTOP-09PBS3S/Downloads/ltee_v3c_stage12_intervention.zip'
cp ltee_v3c_stage12/intervene_v3c_trends.py ltee_v3/
cp ltee_v3c_stage12/test_v3c_stage12.py ltee_v3/
cd ltee_v3
python -m unittest -v test_v3c_stage12.py
OPENBLAS_NUM_THREADS=1 python -u intervene_v3c_trends.py \
  --source v3c_forecast.py \
  --input ../LTEE_prospective/v3b_simulation \
  --original-results ../LTEE_prospective/v3c_stage1 \
  --output ../LTEE_prospective/v3c_stage12_intervention
```

Only validation seed 42 is used. Test seeds 67/101 are explicitly **not opened**. `simulation_truth_RESTRICTED.json` is never opened. Output is separate from original metrics. `validation_interventions.csv` contains fold-level differences in actual predicted probabilities; `validation_summary.csv` aggregates validation NLL and differences; `intervention_report.json` describes the audit.

Note: this is an exploratory intervention after original test results were already inspected. Do not use it to retune and re-report the original test as confirmatory.
