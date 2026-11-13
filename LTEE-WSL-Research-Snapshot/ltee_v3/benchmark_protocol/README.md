# V3-C Stage 4.1B-0: protocol draft

This is a **draft lock candidate**, not a certified benchmark. It creates a machine-readable protocol, a trajectory split manifest and a canonical JSON checksum. It does not generate trajectories, access future labels, or certify callability/biological validity.

Run:

```bash
python -m unittest -v test_protocol.py
python build_protocol.py --output ../LTEE_prospective/v3c_stage41b0_protocol
python validate_protocol.py --protocol-dir ../LTEE_prospective/v3c_stage41b0_protocol --data-root ../LTEE_prospective/v3b_simulation
```

Seeds 11, 23, 42, 67, 101 have been previously inspected. They remain development seeds. Seeds 1001, 1003, 1007, 1009, 1013 are **proposed future locked-test IDs only**. Do not generate or inspect them until simulator version, parameters, and model/analysis plan are frozen and approved. Do not claim these are confirmatory yet.

Before actual locking: record the exact simulator source SHA-256, simulation configuration, dependencies, pair selection and final scoring implementation; increase the number of independent trajectories for statistical power. Keep train/validation/test at trajectory level, and do not count neutral and linkage-only as independent mechanistic replications if their generating processes coincide. Use test targets exclusively in the final evaluator. Keep old Stage 4 and Stage 4.1A outputs unchanged.
