# V3-C Stage 1.1: Read-only trend baseline audit

This package **does not modify** `v3c_forecast.py` or fit any model. It audits whether B1/B2/B3 evaluation rows match exactly, locates source-code trend logic and JSON selection metadata, and tests a standalone linear-trend arithmetic fixture.

It **cannot** establish whether production per-site predictions are identical or whether alpha=0 was selected without corresponding prediction/selection artifacts. Do not interpret the standalone fixture as a production integration test.

Run from `~/LTEE-Ecoli/ltee_v3`:

```bash
python -m unittest -v test_audit_v3c_trends.py
python audit_v3c_trends.py --source v3c_forecast.py --metrics ../LTEE_prospective/v3c_stage1/metrics.csv --results-dir ../LTEE_prospective/v3c_stage1 --output ../LTEE_prospective/v3c_stage11_audit
```

Inspect `trend_audit.json` and the original `v3c_forecast.py` before deciding whether to patch or redesign B2/B3.
