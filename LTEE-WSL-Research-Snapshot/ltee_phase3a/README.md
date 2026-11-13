# Phase 3A synthetic context-aware forecasting

Models: M2 (population rate, no context), M3 linear context, M3 nonlinear fixed-feature context, M3 shuffled context.

Run `python -m unittest -v test_phase3a.py` and `OPENBLAS_NUM_THREADS=1 python -u phase3a.py --output ../LTEE_prospective/phase3a_synthetic`.

Synthetic only. No real Evo 2 embeddings. This is a binary nucleotide-change probability benchmark, not full four-state sequence forecasting. Historical and future events are sampled independently conditional on the same fixed site susceptibilities. No hyperparameter tuning on test; penalty is fixed for this architecture check. Shuffled features use a fixed permutation across periods and are not an Evo 2 ablation. Low mutation rates and temporal regime shifts are not covered. The 'M3_nonlinear' model is a small fixed nonlinear feature map with a trained linear readout, not a trained MLP. No claim of causal epistasis or prospective LTEE validity.
