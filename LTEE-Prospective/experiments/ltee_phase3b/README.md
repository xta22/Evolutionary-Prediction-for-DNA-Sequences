Phase 3B NumPy/SciPy replacement. Files keep their original names: phase3b.py and test_phase3b.py.
Dependencies: numpy scipy pandas scikit-learn. No torch.
Uses scipy L-BFGS-B finite-difference gradients; slower than torch. Synthetic only.
Run: python -m unittest -v test_phase3b.py
Smoke: OPENBLAS_NUM_THREADS=1 python -u phase3b.py --seeds 42 --train 80 --validation 40 --test 50 --sites 30 --epochs 5 --scenarios homogeneous --output ../LTEE_prospective/phase3b_smoke_numpy
Full: OPENBLAS_NUM_THREADS=1 python -u phase3b.py --seeds 11 23 42 --output ../LTEE_prospective/phase3b_synthetic
The default full run may take substantial CPU time; no GPU needed.
