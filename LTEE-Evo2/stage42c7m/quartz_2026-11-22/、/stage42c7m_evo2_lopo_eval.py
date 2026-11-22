#!/usr/bin/env python3
"""Evaluate a frozen Evo2 variant-context head with nested population LOPO.

Input bundle comes from stage42c7m_prepare_evo2_windows.py. Frozen Evo2
alternate-minus-reference embeddings are projected to 16 fixed random features
(projection is label-independent). A ridge head uses those features plus
cutoff frequency and historical trend. In each outer round, it is fit on the
other populations' train rows; ridge penalty and persistence-blend weight are
selected on their calibration rows; only the held-out population's
confirmation rows are scored. Persistence and a history-only model are scored
on the same held-out rows.
"""

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


SPLITS = ("TRAIN_CANDIDATE", "CALIBRATION_CANDIDATE", "CONFIRMATION_CANDIDATE")
BLENDS = (0.0, 0.25, 0.5, 0.75, 1.0)
RIDGE_GRID = (0.1, 1.0, 10.0, 100.0)
PROJECTION_DIM = 16
METRICS = ("frequency_mae", "target_clone_brier", "target_clone_log_loss")
MODELS = ("persistence", "history_only", "evo2_context")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values):
    return sum(values) / len(values) if values else None


def parse_inputs(input_dir, embeddings_dir):
    rows_path = input_dir / "forecast_rows.csv"
    index_path = input_dir / "variant_index.csv"
    embedding_path = embeddings_dir / "variant_embedding_delta.npy"
    embedding_index_path = embeddings_dir / "embedded_variant_index.csv"
    for path in (rows_path, index_path, embedding_path, embedding_index_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    rows_raw = read_csv(rows_path)
    idx_raw = read_csv(index_path)
    embedded_idx = read_csv(embedding_index_path)
    if not rows_raw or not idx_raw:
        raise ValueError("Input bundle has no forecast rows or variant index")
    required = {"evo2_variant_id", "population", "split", "cutoff_generation", "target_generation",
                "cutoff_frequency", "target_frequency", "trend_per_10k"}
    missing = required - set(rows_raw[0])
    if missing:
        raise ValueError(f"forecast_rows.csv missing columns: {sorted(missing)}")
    variant_ids = [r["variant_id"] for r in idx_raw]
    embedded_ids = [r["variant_id"] for r in embedded_idx]
    if variant_ids != embedded_ids:
        raise ValueError("Embedded variant index does not match the prepared variant index/order")
    embeddings = np.load(embedding_path, mmap_mode="r")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(variant_ids):
        raise ValueError("Embedding matrix shape does not match variant index")
    variant_lookup = {variant_id: i for i, variant_id in enumerate(variant_ids)}
    if len(variant_lookup) != len(variant_ids):
        raise ValueError("Duplicate variant IDs in variant index")
    parsed = []
    for line_no, raw in enumerate(rows_raw, 2):
        split = raw["split"].strip()
        if split not in SPLITS:
            raise ValueError(f"Unexpected split {split!r} at forecast_rows.csv line {line_no}")
        variant_id = raw["evo2_variant_id"]
        if variant_id not in variant_lookup:
            raise ValueError(f"Unknown Evo2 variant ID {variant_id!r} at line {line_no}")
        p = float(raw["cutoff_frequency"])
        q = float(raw["target_frequency"])
        trend = float(raw["trend_per_10k"])
        if not (0 <= p <= 1 and 0 <= q <= 1):
            raise ValueError(f"Frequency outside [0,1] at line {line_no}")
        population = raw["population"].strip()
        cutoff = int(raw["cutoff_generation"])
        target = int(raw["target_generation"])
        parsed.append({
            "population": population,
            "split": split,
            "cutoff": cutoff,
            "target": target,
            "fold": (population, cutoff, target),
            "p": p,
            "q": q,
            "trend": trend,
            "variant_id": variant_id,
            "variant_index": variant_lookup[variant_id],
        })
    return parsed, embeddings, variant_ids, rows_path, index_path, embedding_path


def hierarchical_weights(rows):
    folds_by_pop = defaultdict(set)
    row_counts = defaultdict(int)
    for row in rows:
        folds_by_pop[row["population"]].add(row["fold"])
        row_counts[row["fold"]] += 1
    if not folds_by_pop:
        raise ValueError("Cannot fit an empty training set")
    return np.asarray([
        1.0 / len(folds_by_pop) / len(folds_by_pop[r["population"]]) / row_counts[r["fold"]]
        for r in rows
    ], dtype=np.float64)


def score(row, prediction):
    prob = min(max(float(prediction), 1e-6), 1 - 1e-6)
    q = row["q"]
    return {
        "frequency_mae": abs(float(prediction) - q),
        "target_clone_brier": q * (1 - prediction) ** 2 + (1 - q) * prediction ** 2,
        "target_clone_log_loss": -q * math.log(prob) - (1 - q) * math.log(1 - prob),
    }


def macro_metrics(rows, predictions):
    by_fold = defaultdict(lambda: defaultdict(list))
    population_for_fold = {}
    for row, pred in zip(rows, predictions):
        fold = row["fold"]
        population_for_fold[fold] = row["population"]
        for metric, value in score(row, pred).items():
            by_fold[fold][metric].append(value)
    by_population = defaultdict(lambda: defaultdict(list))
    for fold, metrics in by_fold.items():
        pop = population_for_fold[fold]
        for metric, values in metrics.items():
            by_population[pop][metric].append(mean(values))
    return {metric: mean([mean(values[metric]) for values in by_population.values()])
            for metric in METRICS}


def fit_history(train):
    weights = hierarchical_weights(train)
    x = np.column_stack((np.ones(len(train)),
                         np.asarray([r["p"] for r in train]),
                         np.asarray([r["trend"] for r in train])))
    y = np.asarray([r["q"] for r in train])
    gram = x.T @ (weights[:, None] * x)
    rhs = x.T @ (weights * y)
    gram[1, 1] += 1e-8
    gram[2, 2] += 1e-8
    return np.linalg.solve(gram, rhs)


def pred_history(rows, coef):
    x = np.column_stack((np.ones(len(rows)),
                         np.asarray([r["p"] for r in rows]),
                         np.asarray([r["trend"] for r in rows])))
    return np.clip(x @ coef, 0.0, 1.0)


def standardize_features(train_x, other_x, train_rows):
    weights = hierarchical_weights(train_rows)
    mu = np.sum(train_x * weights[:, None], axis=0)
    var = np.sum(((train_x - mu) ** 2) * weights[:, None], axis=0)
    scale = np.sqrt(np.maximum(var, 1e-12))
    return (train_x - mu) / scale, (other_x - mu) / scale, mu, scale


def fit_ridge(train_x, train_y, weights, penalty):
    x = np.column_stack((np.ones(train_x.shape[0]), train_x))
    gram = x.T @ (weights[:, None] * x)
    rhs = x.T @ (weights * train_y)
    gram[1:, 1:] += penalty * np.eye(gram.shape[1] - 1)
    return np.linalg.solve(gram, rhs)


def ridge_predict(x, coef):
    augmented = np.column_stack((np.ones(x.shape[0]), x))
    return np.clip(augmented @ coef, 0.0, 1.0)


def calibrate_blend(rows, raw_predictions):
    grid = []
    for blend in BLENDS:
        predictions = np.clip((1 - blend) * np.asarray([r["p"] for r in rows])
                              + blend * raw_predictions, 0.0, 1.0)
        metrics = macro_metrics(rows, predictions)
        grid.append({"blend_weight": blend, **metrics})
    best = min(grid, key=lambda row: row["target_clone_log_loss"])
    return best["blend_weight"], grid


def paired_population_deltas(test_rows, predictions_by_model):
    per_fold = defaultdict(lambda: defaultdict(list))
    fold_population = {}
    for model, predictions in predictions_by_model.items():
        for row, prediction in zip(test_rows, predictions):
            fold_population[row["fold"]] = row["population"]
            for metric, value in score(row, prediction).items():
                per_fold[(model, row["fold"])][metric].append(value)
    per_pop_model = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (model, fold), metrics in per_fold.items():
        pop = fold_population[fold]
        for metric, values in metrics.items():
            per_pop_model[pop][model][metric].append(mean(values))
    result = {}
    for pop, models in per_pop_model.items():
        result[pop] = {
            model: {metric: mean(values) for metric, values in metrics.items()}
            for model, metrics in models.items()
        }
    return result


def bootstrap_ci(values, seed, replicates):
    vals = list(values)
    if len(vals) < 2:
        return None
    rng = random.Random(seed)
    draws = sorted(mean([vals[rng.randrange(len(vals))] for _ in vals])
                   for _ in range(replicates))
    return [draws[int(0.025 * replicates)], draws[min(int(0.975 * replicates), replicates - 1)]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Unzipped C7M transfer bundle")
    parser.add_argument("--embeddings-dir", type=Path, required=True,
                        help="C7M Evo2 embedding output directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--projection-dim", type=int, default=PROJECTION_DIM)
    parser.add_argument("--projection-seed", type=int, default=20261118)
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    args = parser.parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    embeddings_dir = args.embeddings_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if args.projection_dim < 1 or args.bootstrap_replicates < 100:
        raise ValueError("projection-dim must be positive and bootstrap replicates >=100")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be absent or empty: {output}")

    rows, embeddings, variant_ids, rows_path, variant_index_path, embedding_path = parse_inputs(input_dir, embeddings_dir)
    if not np.isfinite(embeddings).all():
        raise ValueError("Embedding matrix contains non-finite values")
    if args.projection_dim > embeddings.shape[1]:
        raise ValueError("projection-dim exceeds embedding dimension")
    rng = np.random.default_rng(args.projection_seed)
    projection = rng.normal(0.0, 1.0 / math.sqrt(args.projection_dim),
                            size=(embeddings.shape[1], args.projection_dim)).astype(np.float32)
    projected_variants = np.asarray(embeddings @ projection, dtype=np.float64)
    for row in rows:
        row["z"] = projected_variants[row["variant_index"]]
    populations = sorted({row["population"] for row in rows})

    population_results = []
    fold_results = []
    selection_results = []
    for heldout in populations:
        train = [r for r in rows if r["population"] != heldout and r["split"] == "TRAIN_CANDIDATE"]
        calibration = [r for r in rows if r["population"] != heldout and r["split"] == "CALIBRATION_CANDIDATE"]
        test = [r for r in rows if r["population"] == heldout and r["split"] == "CONFIRMATION_CANDIDATE"]
        if not train or not calibration or not test:
            raise ValueError(f"Held-out population {heldout} lacks a required split")

        # History-only benchmark, refit inside each outer fold.
        history_coef = fit_history(train)
        cal_history_raw = pred_history(calibration, history_coef)
        history_blend, history_grid = calibrate_blend(calibration, cal_history_raw)
        test_history_raw = pred_history(test, history_coef)
        test_history = np.clip((1 - history_blend) * np.asarray([r["p"] for r in test])
                               + history_blend * test_history_raw, 0.0, 1.0)

        # Evo2 head uses p, historical trend, and a fixed label-independent random projection.
        train_x = np.vstack([np.concatenate(([r["p"], r["trend"]], r["z"])) for r in train])
        cal_x = np.vstack([np.concatenate(([r["p"], r["trend"]], r["z"])) for r in calibration])
        test_x = np.vstack([np.concatenate(([r["p"], r["trend"]], r["z"])) for r in test])
        train_xs, cal_xs, x_mean, x_scale = standardize_features(train_x, cal_x, train)
        _, test_xs, _, _ = standardize_features(train_x, test_x, train)
        train_y = np.asarray([r["q"] for r in train], dtype=np.float64)
        train_w = hierarchical_weights(train)
        evo_calibration_grid = []
        evo_fitted = {}
        for penalty in RIDGE_GRID:
            coef = fit_ridge(train_xs, train_y, train_w, penalty)
            cal_raw = ridge_predict(cal_xs, coef)
            for blend in BLENDS:
                predictions = np.clip((1 - blend) * np.asarray([r["p"] for r in calibration])
                                      + blend * cal_raw, 0.0, 1.0)
                metrics = macro_metrics(calibration, predictions)
                evo_calibration_grid.append({
                    "ridge_penalty": penalty,
                    "blend_weight": blend,
                    **metrics,
                })
            evo_fitted[penalty] = coef
        chosen = min(evo_calibration_grid, key=lambda row: row["target_clone_log_loss"])
        evo_penalty = chosen["ridge_penalty"]
        evo_blend = chosen["blend_weight"]
        test_evo_raw = ridge_predict(test_xs, evo_fitted[evo_penalty])
        test_evo = np.clip((1 - evo_blend) * np.asarray([r["p"] for r in test])
                           + evo_blend * test_evo_raw, 0.0, 1.0)
        test_persistence = np.asarray([r["p"] for r in test], dtype=np.float64)

        models = {
            "persistence": test_persistence,
            "history_only": test_history,
            "evo2_context": test_evo,
        }
        per_pop = defaultdict(lambda: defaultdict(list))
        fold_temp = defaultdict(lambda: defaultdict(list))
        for model_name, predictions in models.items():
            for row, pred in zip(test, predictions):
                fold = row["fold"]
                for metric, value in score(row, pred).items():
                    fold_temp[(model_name, fold)][metric].append(value)
        for (model_name, fold), metrics in fold_temp.items():
            pop = fold[0]
            fold_means = {metric: mean(values) for metric, values in metrics.items()}
            n_site_rows = len(next(iter(metrics.values())))
            fold_results.append({
                "heldout_population": heldout,
                "population": pop,
                "cutoff_generation": fold[1],
                "target_generation": fold[2],
                "model": model_name,
                "n_site_allele_rows": n_site_rows,
                **fold_means,
            })
            for metric, value in fold_means.items():
                per_pop[model_name][metric].append(value)
        population_results.append({
            "heldout_population": heldout,
            **{f"{model}_{metric}": mean(per_pop[model][metric])
               for model in MODELS for metric in METRICS},
        })
        selection_results.append({
            "heldout_population": heldout,
            "n_train_populations": len({r["population"] for r in train}),
            "n_calibration_populations": len({r["population"] for r in calibration}),
            "n_test_folds": len({r["fold"] for r in test}),
            "history_intercept": float(history_coef[0]),
            "history_cutoff_slope": float(history_coef[1]),
            "history_trend_slope": float(history_coef[2]),
            "history_blend_weight": history_blend,
            "history_calibration_grid_json": json.dumps(history_grid),
            "evo2_ridge_penalty": evo_penalty,
            "evo2_blend_weight": evo_blend,
            "evo2_calibration_best_log_loss": chosen["target_clone_log_loss"],
            "evo2_calibration_grid_json": json.dumps(evo_calibration_grid),
        })

    delta_results = []
    comparisons = (("history_only", "persistence"),
                   ("evo2_context", "persistence"),
                   ("evo2_context", "history_only"))
    for metric_i, metric in enumerate(METRICS):
        for left, right in comparisons:
            deltas = [row[f"{left}_{metric}"] - row[f"{right}_{metric}"]
                      for row in population_results]
            delta_results.append({
                "comparison": f"{left}_minus_{right}",
                "metric": metric,
                "delta_definition": "negative favors the left model",
                "mean_paired_population_delta": mean(deltas),
                "population_cluster_bootstrap_95pct_ci": json.dumps(
                    bootstrap_ci(deltas, 76201 + metric_i * 10 + MODELS.index(left),
                                 args.bootstrap_replicates)
                ),
                "n_heldout_population_clusters": len(deltas),
            })
    overall_results = []
    for model_name in MODELS:
        for metric_i, metric in enumerate(METRICS):
            values = [row[f"{model_name}_{metric}"] for row in population_results]
            overall_results.append({
                "model": model_name,
                "metric": metric,
                "mean_over_heldout_populations": mean(values),
                "population_cluster_bootstrap_95pct_ci": json.dumps(
                    bootstrap_ci(values, 77201 + MODELS.index(model_name) * 10 + metric_i,
                                 args.bootstrap_replicates)
                ),
                "n_heldout_population_clusters": len(values),
            })

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "outer_population_metrics.csv", population_results, list(population_results[0]))
    write_csv(output / "outer_fold_metrics.csv", fold_results, list(fold_results[0]))
    write_csv(output / "outer_model_selections.csv", selection_results, list(selection_results[0]))
    write_csv(output / "paired_model_deltas.csv", delta_results, list(delta_results[0]))
    write_csv(output / "overall_model_metrics.csv", overall_results, list(overall_results[0]))
    report = {
        "status": "EXPLORATORY_FROZEN_EVO2_NESTED_POPULATION_LOPO",
        "outer_validation": "Hold out each population's confirmation folds; fit and calibrate on other populations only.",
        "evo2": {
            "model": "evo2_7b",
            "layer": "blocks.28.mlp.l3",
            "embedding": "alternate-window minus reference-window mean-pooled embedding",
            "random_projection_dimension": args.projection_dim,
            "random_projection_seed": args.projection_seed,
            "ridge_penalty_grid": list(RIDGE_GRID),
            "blend_grid": list(BLENDS),
        },
        "calibration_criterion": "population-macro target-clone log loss",
        "n_population_clusters": len(populations),
        "n_forecast_rows": len(rows),
        "n_unique_variants": len(variant_ids),
        "n_confirmation_population_time_folds": len({r["fold"] for r in rows if r["split"] == "CONFIRMATION_CANDIDATE"}),
        "projection_note": "Random projection is label-independent; all feature standardization is fit on outer training rows only.",
        "input_sha256": {
            "forecast_rows": sha256_file(rows_path),
            "variant_index": sha256_file(variant_index_path),
            "embedding_matrix": sha256_file(embedding_path),
        },
        "overall_metrics": overall_results,
        "paired_deltas": delta_results,
        "outputs": ["outer_population_metrics.csv", "outer_fold_metrics.csv",
                    "outer_model_selections.csv", "paired_model_deltas.csv",
                    "overall_model_metrics.csv", "report.json"],
        "limitations": [
            "Only 12 LTEE population clusters are available; intervals will be uncertain.",
            "Candidate alleles are cutoff-observed SNPs only; novel later mutations, indels, and structural variants are excluded.",
            "Evo2 sees only the REL606 focal-variant context, not other mutations in the sampled clone population.",
            "The supervised head is a low-dimensional ridge model; Evo2 is frozen and not fine-tuned.",
            "Earlier C7I-C7L results have been inspected, so this is internal development evidence, not a blind locked test.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
