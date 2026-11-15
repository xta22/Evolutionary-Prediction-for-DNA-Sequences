#!/usr/bin/env python3
"""Stage 4.2C.5B: leakage-safe sampling-depth experiment with full M5 History.

At each sample depth: fit the four-parameter M5 History model on train seeds,
select Beta smoothing and uniform mixing on separate calibration seeds, and
evaluate frozen choices on untouched confirmation seeds. The independent and
pairing-shuffled controls use the same selected smoothing parameters.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from stage42c3_matched import load, make_rows
from stage42c4_audit import dependence, from_marginals_and_D
from stage42c5a import smooth_marginals

SCENARIOS = ("neutral", "recombination", "selection_only", "epistasis")
MODELS = (
    "sampled_independent_unsmoothed",
    "M5_history_unsmoothed",
    "sampled_independent_calibrated",
    "M5_history_calibrated",
    "M5_history_pairing_shuffled_calibrated",
)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def predict_history_calibrated(row, theta, model, prod, alpha, uniform_mix, shuffle_seed=None):
    if not 0 <= uniform_mix < 1:
        raise ValueError("uniform_mix must be in [0,1)")
    if shuffle_seed is None:
        latest = row["latest"]
    else:
        # Randomize cross-site pairing while preserving the observed site marginals.
        latest = model.shuffle_joint(row, int(shuffle_seed))
    p = smooth_marginals(row["latest_matrix"], row["pairs"], alpha)
    d = dependence(latest)
    logits = model.features(row, "history") @ theta
    rho = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    q, _ = from_marginals_and_D(p, rho * d)
    q = (1.0 - uniform_mix) * q + uniform_mix / 4.0
    return prod.normalize(q)


def predict_independent_calibrated(row, prod, alpha, uniform_mix):
    p = smooth_marginals(row["latest_matrix"], row["pairs"], alpha)
    q, _ = from_marginals_and_D(p, np.zeros(len(p), dtype=float))
    q = (1.0 - uniform_mix) * q + uniform_mix / 4.0
    return prod.normalize(q)


def mean_nll(rows, predictions):
    return float(np.mean([
        predictions(row)["joint_nll"] for row in rows
    ]))


def bootstrap_ci(values, seed=8451, draws=10000):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return [None, None]
    rng = np.random.default_rng(seed)
    sample = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(sample, [0.025, 0.975])]


def stable_shuffle_seed(seed, scenario, cutoff, horizon):
    key = f"C5B-history:{seed}:{scenario}:{cutoff}:{horizon}".encode()
    return int(hashlib.sha256(key).hexdigest()[:16], 16)


def run(a):
    splits = [a.train_seeds, a.calibration_seeds, a.confirmation_seeds]
    if any(set(splits[i]) & set(splits[j]) for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("train, calibration, and confirmation seeds must be disjoint")
    if any(not s for s in splits):
        raise ValueError("all three seed splits must be nonempty")
    if len(set(a.sample_sizes)) != len(a.sample_sizes) or any(
        n < 1 or n > a.individuals for n in a.sample_sizes
    ):
        raise ValueError("sample sizes must be unique and within the population size")
    if a.sites % 2:
        raise ValueError("sites must be even for the adjacent-pair definition")
    if not a.alphas or any(x < 0 for x in a.alphas) or len(set(a.alphas)) != len(a.alphas):
        raise ValueError("invalid alpha grid")
    if not a.mixes or any(not 0 <= x < 1 for x in a.mixes) or len(set(a.mixes)) != len(a.mixes):
        raise ValueError("invalid uniform-mix grid")
    if a.penalty < 0:
        raise ValueError("penalty must be nonnegative")
    if any(c % a.observe_step or c + h > a.generations
           for c in a.cutoffs for h in a.horizons):
        raise ValueError("cutoff/horizon not available in simulator")
    if a.output.exists() and any(a.output.iterdir()):
        raise ValueError("output must be absent or empty")

    ref = load(a.reference, "ref_c5b_full_history")
    prod = load(a.source, "prod_c5b_full_history")
    model = load(a.model_source, "model_c5b_full_history")

    all_metrics, all_grid, all_seed_gains, depth_reports = [], [], [], []
    for depth in a.sample_sizes:
        a.sample_size = depth
        train = make_rows(ref, prod, a, a.scenarios, a.train_seeds, a.cutoffs, a.horizons)
        calibration = make_rows(ref, prod, a, a.scenarios, a.calibration_seeds, a.cutoffs, a.horizons)
        if not train or not calibration:
            raise ValueError(f"empty train/calibration rows at depth {depth}")

        theta, fit_info = model.fit(train, "history", prod, a.penalty)
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (4,):
            raise ValueError(f"expected four M5 History coefficients, got {theta.shape}")
        print(f"TRAIN sample_size={depth} theta={theta.tolist()} iterations={fit_info['iterations']}", flush=True)

        grid = []
        for alpha in a.alphas:
            for mix in a.mixes:
                score = mean_nll(
                    calibration,
                    lambda row: prod.score(
                        predict_history_calibrated(row, theta, model, prod, alpha, mix),
                        row["truth"],
                    ),
                )
                grid.append({"sample_size": depth, "alpha": alpha,
                             "uniform_mix": mix, "calibration_m5_history_nll": score})
        chosen = min(grid, key=lambda r: (r["calibration_m5_history_nll"], r["alpha"], r["uniform_mix"]))
        all_grid.extend(grid)

        # Generate confirmation trajectories only after the fit and smoothing choice are frozen.
        confirmation = make_rows(ref, prod, a, a.scenarios, a.confirmation_seeds, a.cutoffs, a.horizons)
        if not confirmation:
            raise ValueError(f"empty confirmation rows at depth {depth}")

        metrics = []
        for row in confirmation:
            theta_predict = model.predict(row, theta, "history", prod)
            q_ind = prod.normalize(prod.independent(row["latest"]))
            q_ind_cal = predict_independent_calibrated(
                row, prod, chosen["alpha"], chosen["uniform_mix"]
            )
            q_m5_cal = predict_history_calibrated(
                row, theta, model, prod, chosen["alpha"], chosen["uniform_mix"]
            )
            q_m5_shuffle = predict_history_calibrated(
                row, theta, model, prod, chosen["alpha"], chosen["uniform_mix"],
                shuffle_seed=stable_shuffle_seed(
                    row["seed"], row["scenario"], row["cutoff"], row["horizon"]
                ),
            )
            predictions = {
                "sampled_independent_unsmoothed": q_ind,
                "M5_history_unsmoothed": theta_predict,
                "sampled_independent_calibrated": q_ind_cal,
                "M5_history_calibrated": q_m5_cal,
                "M5_history_pairing_shuffled_calibrated": q_m5_shuffle,
            }
            for name in MODELS:
                scores = prod.score(predictions[name], row["truth"])
                metrics.append({
                    "sample_size": depth,
                    "scenario": row["scenario"],
                    "seed": row["seed"],
                    "cutoff": row["cutoff"],
                    "horizon": row["horizon"],
                    "fold_id": row["fold_id"],
                    "model": name,
                    "alpha": chosen["alpha"],
                    "uniform_mix": chosen["uniform_mix"],
                    "n_pairs": len(row["pairs"]),
                    "n_target_clones": row["n_target_clones"],
                    **scores,
                })
        all_metrics.extend(metrics)

        # Treat each independent confirmation seed as a replicate; aggregate its
        # cutoffs/horizons first, separately within each scenario.
        by_scenario_seed = {}
        for row in metrics:
            key = (row["scenario"], row["seed"])
            by_scenario_seed.setdefault(key, {}).setdefault(row["model"], []).append(row["joint_nll"])
        depth_gains = []
        for (scenario, seed), scores in sorted(by_scenario_seed.items()):
            ind = float(np.mean(scores["sampled_independent_calibrated"]))
            m5 = float(np.mean(scores["M5_history_calibrated"]))
            shuf = float(np.mean(scores["M5_history_pairing_shuffled_calibrated"]))
            item = {
                "sample_size": depth,
                "scenario": scenario,
                "seed": seed,
                "independent_calibrated_mean_nll": ind,
                "M5_history_calibrated_mean_nll": m5,
                "independent_minus_M5_nll": ind - m5,
                "M5_history_minus_pairing_shuffled_nll": shuf - m5,
            }
            all_seed_gains.append(item)
            depth_gains.append(item)

        scenario_reports = []
        for scenario in a.scenarios:
            vals = [r["independent_minus_M5_nll"] for r in depth_gains if r["scenario"] == scenario]
            shuf_vals = [r["M5_history_minus_pairing_shuffled_nll"]
                         for r in depth_gains if r["scenario"] == scenario]
            scenario_reports.append({
                "scenario": scenario,
                "mean_independent_minus_M5_nll": float(np.mean(vals)),
                "seed_bootstrap_95pct_ci": bootstrap_ci(vals, seed=9400 + depth + len(scenario)),
                "mean_M5_minus_pairing_shuffled_nll": float(np.mean(shuf_vals)),
                "pairing_shuffle_seed_bootstrap_95pct_ci": bootstrap_ci(shuf_vals, seed=9800 + depth + len(scenario)),
            })

        seed_average = {}
        for r in depth_gains:
            seed_average.setdefault(r["seed"], []).append(r["independent_minus_M5_nll"])
        overall_vals = [float(np.mean(v)) for v in seed_average.values()]
        model_means = {
            name: float(np.mean([r["joint_nll"] for r in metrics if r["model"] == name]))
            for name in MODELS
        }
        depth_reports.append({
            "sample_size": depth,
            "m5_history_theta_train_only": theta.tolist(),
            "fit_info": fit_info,
            "chosen_calibration": chosen,
            "n_confirmation_folds": len(confirmation),
            "mean_nll_by_model": model_means,
            "mean_paired_independent_minus_m5_nll": float(np.mean(overall_vals)),
            "paired_gain_seed_bootstrap_95pct_ci": bootstrap_ci(overall_vals, seed=11200 + depth),
            "paired_gain_by_scenario": scenario_reports,
        })
        print(f"CONFIRM sample_size={depth} chosen_alpha={chosen['alpha']} "
              f"mix={chosen['uniform_mix']} paired_gain={np.mean(overall_vals):.6f}", flush=True)

    a.output.mkdir(parents=True, exist_ok=True)
    write_csv(a.output / "confirmation_metrics.csv", all_metrics)
    write_csv(a.output / "calibration_grid.csv", all_grid)
    write_csv(a.output / "paired_gain_by_seed.csv", all_seed_gains)

    summary_rows = []
    for depth in a.sample_sizes:
        for scenario in a.scenarios:
            for name in MODELS:
                vals = [r["joint_nll"] for r in all_metrics if r["sample_size"] == depth
                        and r["scenario"] == scenario and r["model"] == name]
                summary_rows.append({
                    "sample_size": depth, "scenario": scenario, "model": name,
                    "mean_joint_nll": float(np.mean(vals)), "n_folds": len(vals),
                })
    write_csv(a.output / "sample_depth_summary.csv", summary_rows)

    source_hashes = {
        "reference": hashlib.sha256(a.reference.read_bytes()).hexdigest(),
        "production_source": hashlib.sha256(a.source.read_bytes()).hexdigest(),
        "model_source": hashlib.sha256(a.model_source.read_bytes()).hexdigest(),
        "stage42c3_matched": hashlib.sha256(Path(__import__("stage42c3_matched").__file__).read_bytes()).hexdigest(),
        "stage42c4_audit": hashlib.sha256(Path(__import__("stage42c4_audit").__file__).read_bytes()).hexdigest(),
    }
    report = {
        "status": "EXPLORATORY_HELDOUT_DEVELOPMENT",
        "model": "M5 History, four fitted coefficients from stage41b.fit(kind='history')",
        "train_seeds": a.train_seeds,
        "calibration_seeds": a.calibration_seeds,
        "confirmation_seeds": a.confirmation_seeds,
        "sample_sizes": a.sample_sizes,
        "scenarios": a.scenarios,
        "cutoffs": a.cutoffs,
        "horizons": a.horizons,
        "alphas": a.alphas,
        "uniform_mixes": a.mixes,
        "penalty": a.penalty,
        "source_sha256": source_hashes,
        "results_by_depth": depth_reports,
        "design_notes": [
            "M5 History theta is fit on train seeds separately at each sample depth.",
            "Alpha and uniform mix are selected on calibration seeds only, separately at each depth.",
            "Confirmation seeds are disjoint from train and calibration and are scored only after choices freeze.",
            "The independent baseline shares the M5-selected alpha and uniform mix to isolate dependence value.",
            "Pairing-shuffled control uses the original four-parameter theta but permutes site pairings in the cutoff sample.",
            "Seed is the replicate unit; cutoff/horizon folds are aggregated within seed before bootstrap intervals.",
            "Depths use nested first-N samples as implemented by the simulator; they are not random population subsamples.",
            "This is development evidence, not a locked test or external biological validation.",
        ],
    }
    (a.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "source", "model-source", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--train-seeds", type=int, nargs="+", default=list(range(4101, 4121)))
    p.add_argument("--calibration-seeds", type=int, nargs="+", default=list(range(5101, 5121)))
    p.add_argument("--confirmation-seeds", type=int, nargs="+", default=list(range(6101, 6121)))
    p.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    p.add_argument("--cutoffs", type=int, nargs="+", default=[20, 50])
    p.add_argument("--horizons", type=int, nargs="+", default=[10, 20, 30])
    p.add_argument("--sample-sizes", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    p.add_argument("--alphas", type=float, nargs="+", default=[0, 0.1, 0.5, 1, 2, 5])
    p.add_argument("--mixes", type=float, nargs="+", default=[0, 0.001, 0.01, 0.05])
    p.add_argument("--penalty", type=float, default=0.01)
    for name, typ, default in [
        ("individuals", int, 120), ("sites", int, 24), ("generations", int, 100),
        ("observe-step", int, 10), ("rate", float, 0.001),
        ("initial-frequency", float, 0.25), ("context-sigma", float, 1.0),
        ("heterogeneity-multiplier", float, 3.0), ("shift-generation", int, 50),
        ("shift-multiplier", float, 4.0), ("recombination-probability", float, 0.5),
        ("selection-strength", float, 0.12), ("epistasis-strength", float, 0.25),
    ]:
        p.add_argument("--" + name, type=typ, default=default)
    a = p.parse_args()
    run(a)


if __name__ == "__main__":
    main()
