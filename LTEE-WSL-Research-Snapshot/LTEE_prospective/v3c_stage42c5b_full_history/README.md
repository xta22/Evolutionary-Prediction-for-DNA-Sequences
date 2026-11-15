# Stage 4.2C.5B — Full M5 History sampling-depth experiment

This runner uses the four-coefficient M5 History implementation in `stage41b.py`. At each depth it fits theta on train trajectories, selects Beta smoothing (`alpha`) and uniform mixing on separate calibration trajectories, and evaluates frozen choices on disjoint confirmation trajectories. It includes an independent calibrated baseline and a true pairing-shuffled control.

## WSL run

This ZIP includes the previously local `stage42c3_matched.py`, `stage42c4_audit.py`, and `stage42c5a.py` under `source/`. The runner also needs the existing sibling files `stage41b.py`, `stage4.py`, and `v3b_simulate.py` in `LTEE-WSL-Research-Snapshot/ltee_v3/`.

First extract the downloaded ZIP (adjust `ZIP` if the download path differs):

```bash
ZIP="/mnt/c/Users/xiaoyuetang.DESKTOP-09PBS3S/Downloads/stage42c5b_full_history.zip"
unzip -o "$ZIP" -d /tmp/stage42c5b_full_history_upload
```

Then copy the runner and its local modules into the project and run:

```bash
V3="LTEE-WSL-Research-Snapshot/ltee_v3"
OUT="$HOME/LTEE-Ecoli/LTEE_prospective/v3c_stage42c5b_full_history"
PKG="/tmp/stage42c5b_full_history_upload/stage42c5b_full_history_delivery"
cp "$PKG"/source/*.py "$V3/"
cp "$PKG/stage42c5b_full_history.py" "$V3/"
python -u "$V3/stage42c5b_full_history.py" \
  --reference "$V3/v3b_simulate.py" \
  --source "$V3/stage4.py" \
  --model-source "$V3/stage41b.py" \
  --output "$OUT"
```

Defaults: sample sizes 2, 4, 8, 16, 32; 20 seeds each for train (4101–4120), calibration (5101–5120), and confirmation (6101–6120); four scenarios; cutoffs 20 and 50; horizons 10, 20, 30. These seeds are separate from the earlier global-rho C.5B pilot.

## Pilot result

Paired gain is `independent calibrated NLL − M5 History calibrated NLL`; positive values favor M5 History. Estimates are averaged within confirmation seed across its folds, and the bootstrap resamples confirmation seeds.

| Sample size | Selected alpha / mix | Mean NLL: independent | Mean NLL: M5 History | Paired gain (95% seed bootstrap CI) |
|---:|---:|---:|---:|---:|
| 2 | 0.5 / 0.05 | 1.102084 | 1.095196 | 0.006888 [0.003816, 0.010264] |
| 4 | 0.5 / 0.05 | 1.014731 | 1.002890 | 0.011840 [0.008391, 0.015218] |
| 8 | 0.5 / 0.05 | 0.956656 | 0.935943 | 0.020713 [0.016750, 0.024993] |
| 16 | 0.5 / 0.01 | 0.918781 | 0.888334 | 0.030447 [0.025878, 0.035313] |
| 32 | 1 / 0 | 0.893313 | 0.860141 | 0.033172 [0.028795, 0.037849] |

The pairing-shuffled calibrated control scored worse than M5 History at every depth. Low-depth recombination gains were small and uncertain; they became positive at depths 16 and 32. This remains exploratory evidence from the toy simulator, not a locked test or biological validation.
