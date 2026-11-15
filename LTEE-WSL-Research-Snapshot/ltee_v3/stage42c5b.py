#!/usr/bin/env python3
"""Leakage-separated sampling-depth pilot for Stage 4.2C.5B.

Uses the V3-B simulator and the C.5A.1 prediction primitives. Each depth gets
its own train-only rho fit and calibration-only alpha/mix selection. Confirmation
seeds are never used for either choice. Depth comparisons use matched seeds and
summarize at the seed level because cutoffs/horizons within a trajectory are
correlated.
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from stage42c5a1 import load, make_examples, mean_nll, probs, score


def write_csv(path, rows, fields):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def bootstrap_ci(values, seed=8451, draws=10000):
    """Percentile CI from resampling confirmation seeds (the independent units)."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return [None, None]
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train-seeds", type=int, nargs="+", default=list(range(1001, 1021)))
    p.add_argument("--calibration-seeds", type=int, nargs="+", default=list(range(2001, 2021)))
    p.add_argument("--confirmation-seeds", type=int, nargs="+", default=list(range(3001, 3021)))
    p.add_argument("--scenarios", nargs="+", default=["neutral", "recombination", "selection_only", "epistasis"])
    p.add_argument("--cutoffs", type=int, nargs="+", default=[20, 50])
    p.add_argument("--horizons", type=int, nargs="+", default=[10, 20, 30])
    p.add_argument("--sample-sizes", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    p.add_argument("--alphas", type=float, nargs="+", default=[0, 0.1, 0.5, 1, 2, 5])
    p.add_argument("--mixes", type=float, nargs="+", default=[0, 0.001, 0.01, 0.05])
    p.add_argument("--individuals", type=int, default=120)
    p.add_argument("--sites", type=int, default=24)
    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--observe-step", type=int, default=10)
    p.add_argument("--rate", type=float, default=0.001)
    p.add_argument("--initial-frequency", type=float, default=0.25)
    p.add_argument("--context-sigma", type=float, default=1.0)
    p.add_argument("--heterogeneity-multiplier", type=float, default=3.0)
    p.add_argument("--shift-generation", type=int, default=50)
    p.add_argument("--shift-multiplier", type=float, default=4.0)
    p.add_argument("--recombination-probability", type=float, default=0.5)
    p.add_argument("--selection-strength", type=float, default=0.12)
    p.add_argument("--epistasis-strength", type=float, default=0.25)
    a = p.parse_args()

    splits = [a.train_seeds, a.calibration_seeds, a.confirmation_seeds]
    if any(set(splits[i]) & set(splits[j]) for i in range(3) for j in range(i + 1, 3)):
        p.error("train, calibration, and confirmation seeds must be disjoint")
    if len(set(a.sample_sizes)) != len(a.sample_sizes) or any(n < 1 or n > a.individuals for n in a.sample_sizes):
        p.error("sample sizes must be unique and between 1 and --individuals")
    if a.sites % 2:
        p.error("sites must be even")
    if any(x < 0 for x in a.alphas) or any(x < 0 or x >= 1 for x in a.mixes):
        p.error("invalid smoothing grid")
    if a.output.exists() and any(a.output.iterdir()):
        p.error("output must be absent or empty")

    sim = load(a.reference)
    all_rows, all_grid, all_seed_gains, per_depth = [], [], [], []

    for depth in a.sample_sizes:
        a.sample_size = depth
        train = make_examples(sim, a, a.train_seeds, a.scenarios)
        calibration = make_examples(sim, a, a.calibration_seeds, a.scenarios)
        if not train or not calibration:
            raise ValueError(f"empty train or calibration fold at sample size {depth}")

        # Persistence fit uses training seeds only; the baseline smoothing is fixed.
        opt = minimize_scalar(
            lambda rho: mean_nll(train, 1.0, 0.0, rho),
            bounds=(0, 1), method="bounded", options={"xatol": 1e-5},
        )
        if not opt.success:
            raise RuntimeError(f"rho optimization failed at sample size {depth}")
        rho = float(opt.x)

        grid = []
        for alpha in a.alphas:
            for mix in a.mixes:
                grid.append({
                    "sample_size": depth,
                    "alpha": alpha,
                    "mix": mix,
                    "calibration_nll": mean_nll(calibration, alpha, mix, rho),
                })
        chosen = min(grid, key=lambda x: (x["calibration_nll"], x["alpha"], x["mix"]))
        all_grid.extend(grid)

        # Confirmation generation/evaluation occurs only after this depth's choices freeze.
        confirm = make_examples(sim, a, a.confirmation_seeds, a.scenarios)
        if not confirm:
            raise ValueError(f"empty confirmation folds at sample size {depth}")
        rows = []
        for scenario, seed, cutoff, horizon, counts, target in confirm:
            configs = [
                ("sampled_independent_unsmoothed", 0.0, 0.0, 0.0),
                ("M5_unsmoothed", 0.0, 0.0, rho),
                ("sampled_independent_calibrated", chosen["alpha"], chosen["mix"], 0.0),
                ("M5_calibrated", chosen["alpha"], chosen["mix"], rho),
                # rho=0 is a dependence-off ablation, not a pairing permutation.
                ("M5_calibrated_dependence_off", chosen["alpha"], chosen["mix"], 0.0),
            ]
            for model, alpha, mix, model_rho in configs:
                pred = probs(counts, alpha, mix, model_rho)
                rows.append({
                    "sample_size": depth,
                    "scenario": scenario,
                    "seed": seed,
                    "cutoff": cutoff,
                    "horizon": horizon,
                    "model": model,
                    "joint_nll": score(pred, target),
                    "min_probability": float(pred.min()),
                })
        all_rows.extend(rows)

        # Average the six cutoff/horizon scores within each confirmation trajectory.
        by_seed = {}
        for row in rows:
            key = (row["scenario"], row["seed"])
            by_seed.setdefault(key, {}).setdefault(row["model"], []).append(row["joint_nll"])
        seed_gains = []
        for (scenario, seed), model_scores in sorted(by_seed.items()):
            mean_ind = float(np.mean(model_scores["sampled_independent_calibrated"]))
            mean_m5 = float(np.mean(model_scores["M5_calibrated"]))
            gain = mean_ind - mean_m5
            item = {
                "sample_size": depth,
                "scenario": scenario,
                "seed": seed,
                "independent_calibrated_mean_nll": mean_ind,
                "M5_calibrated_mean_nll": mean_m5,
                "independent_minus_M5_nll": gain,
            }
            all_seed_gains.append(item)
            seed_gains.append(item)

        scenario_summaries = []
        for scenario in a.scenarios:
            vals = [x["independent_minus_M5_nll"] for x in seed_gains if x["scenario"] == scenario]
            scenario_summaries.append({
                "scenario": scenario,
                "mean_independent_minus_M5_nll": float(np.mean(vals)),
                "seed_bootstrap_95pct_ci": bootstrap_ci(vals, seed=8451 + depth + len(scenario)),
            })
        per_depth.append({
            "sample_size": depth,
            "rho_train_only": rho,
            "chosen_calibration": chosen,
            "n_confirmation_folds": len(confirm),
            "paired_seed_gain_by_scenario": scenario_summaries,
        })
        print(f"sample_size={depth} rho={rho:.6f} alpha={chosen['alpha']} mix={chosen['mix']}", flush=True)

    a.output.mkdir(parents=True, exist_ok=True)
    write_csv(a.output / "confirmation_metrics.csv", all_rows, list(all_rows[0]))
    write_csv(a.output / "calibration_grid.csv", all_grid, list(all_grid[0]))
    write_csv(a.output / "paired_gain_by_seed.csv", all_seed_gains, list(all_seed_gains[0]))

    summary_rows = []
    for depth in a.sample_sizes:
        for scenario in a.scenarios:
            for model in sorted({r["model"] for r in all_rows}):
                vals = [r["joint_nll"] for r in all_rows
                        if r["sample_size"] == depth and r["scenario"] == scenario and r["model"] == model]
                summary_rows.append({
                    "sample_size": depth,
                    "scenario": scenario,
                    "model": model,
                    "mean_joint_nll": float(np.mean(vals)),
                    "n_folds": len(vals),
                })
    write_csv(a.output / "sample_depth_summary.csv", summary_rows, list(summary_rows[0]))

    report = {
        "status": "EXPLORATORY_HELDOUT_DEVELOPMENT",
        "reference_sha256": __import__("hashlib").sha256(a.reference.read_bytes()).hexdigest(),
        "train_seeds": a.train_seeds,
        "calibration_seeds": a.calibration_seeds,
        "confirmation_seeds": a.confirmation_seeds,
        "sample_sizes": a.sample_sizes,
        "scenario_order": a.scenarios,
        "cutoffs": a.cutoffs,
        "horizons": a.horizons,
        "results_by_depth": per_depth,
        "design_notes": [
            "Separate, non-overlapping seed splits for fitting, calibration, and confirmation.",
            "Each depth has its own train-only rho and calibration-only alpha/mix.",
            "Same seeds across depths yield paired, nested observation samples; simulator population trajectories are unchanged by sample_size.",
            "Confidence intervals resample confirmation seeds, not correlated cutoff/horizon rows.",
            "Dependence-off is rho=0; it is not a history-pairing permutation test.",
            "This remains development evidence; it is not a locked confirmatory test or the full M5 History model.",
        ],
    }
    (a.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
