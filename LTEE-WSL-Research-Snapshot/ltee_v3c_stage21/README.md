# V3-C Stage 2.1 — Read-only feature/optimization audit

This is an **honest partial audit**, not a replacement for production-model instrumentation. It consumes the existing metrics.csv, selected_models.json, and v3c_stage2.py. It does not retrain, import production code, access future labels, or claim to compute gradients.

## Run

```bash
cd ~/LTEE-Ecoli/ltee_v3
python -m unittest -v test_audit_v3c_stage21.py
python audit_v3c_stage21.py \
  --metrics ../LTEE_prospective/v3c_stage2/metrics.csv \
  --models ../LTEE_prospective/v3c_stage2/selected_models.json \
  --source v3c_stage2.py \
  --output ../LTEE_prospective/v3c_stage21_audit
```

Optional `--features PATH` expects a verified training-only feature CSV; `--predictions PATH` expects model, prediction, and sample/site identifiers. Without them the audit explicitly reports NOT_AVAILABLE. A full permutation or gradient audit requires production interfaces/optimizer traces; never infer those from aggregate metrics. Existing test data have been examined and are exploratory only.
