#!/usr/bin/env python3
"""C7O: target clone-count weighting sensitivity for the frozen Evo2 C7M task.

Re-fits the lightweight nested-LOPO ridge heads from stored C7M embeddings,
then saves held-out per-site predictions and scores them both site-equally
and weighted by the number of target clones. This is CPU-only; it does not
load Evo2 or recompute embeddings.
"""
import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

import stage42c7m_evo2_lopo_eval as base

MODELS = ("persistence", "history_only", "evo2_only", "evo2_plus_history")
WEIGHTINGS = ("site_equal", "target_clone_weighted")


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def weighted_score(row, pred):
    pclip = min(max(float(pred), 1e-6), 1 - 1e-6)
    q = float(row["q"])
    return {"frequency_mae": abs(float(pred) - q),
            "target_clone_brier": q * (1 - pred) ** 2 + (1 - q) * pred ** 2,
            "target_clone_log_loss": -q * math.log(pclip) - (1 - q) * math.log(1 - pclip)}


def aggregate(rows, predictions, weighting):
    """Site-level scores -> equal fold -> equal population; optional n weighting within fold."""
    fold_scores = defaultdict(lambda: defaultdict(list))
    fold_weights = defaultdict(list)
    fold_pop = {}
    for row, pred in zip(rows, predictions):
        fold = row["fold"]
        fold_pop[fold] = row["population"]
        wt = row["n_target"] if weighting == "target_clone_weighted" else 1
        fold_weights[fold].append(wt)
        for metric, val in weighted_score(row, pred).items():
            fold_scores[fold][metric].append((val, wt))
    pop_vals = defaultdict(lambda: defaultdict(list))
    for fold, metrics in fold_scores.items():
        pop = fold_pop[fold]
        for metric, pairs in metrics.items():
            if weighting == "target_clone_weighted":
                value = sum(v*w for v, w in pairs) / sum(w for _, w in pairs)
            else:
                value = sum(v for v, _ in pairs) / len(pairs)
            pop_vals[pop][metric].append(value)
    return {metric: base.mean([base.mean(vals[metric]) for vals in pop_vals.values()])
            for metric in base.METRICS}


def fit_ridge_model(train, calibration, test, feature_fn):
    xtr = np.vstack([feature_fn(r) for r in train])
    xca = np.vstack([feature_fn(r) for r in calibration])
    xte = np.vstack([feature_fn(r) for r in test])
    xtr_s, xca_s, _, _ = base.standardize_features(xtr, xca, train)
    _, xte_s, _, _ = base.standardize_features(xtr, xte, train)
    y = np.asarray([r["q"] for r in train], dtype=np.float64)
    weights = base.hierarchical_weights(train)
    grid, coefs = [], {}
    for penalty in base.RIDGE_GRID:
        coef = base.fit_ridge(xtr_s, y, weights, penalty)
        coefs[penalty] = coef
        raw = base.ridge_predict(xca_s, coef)
        for blend in base.BLENDS:
            pred = np.clip((1-blend)*np.asarray([r["p"] for r in calibration]) + blend*raw, 0, 1)
            score = base.macro_metrics(calibration, pred)
            grid.append({"ridge_penalty": penalty, "blend_weight": blend, **score})
    selected = min(grid, key=lambda x: x["target_clone_log_loss"])
    raw_test = base.ridge_predict(xte_s, coefs[selected["ridge_penalty"]])
    pred_test = np.clip((1-selected["blend_weight"])*np.asarray([r["p"] for r in test])
                        + selected["blend_weight"]*raw_test, 0, 1)
    return pred_test, selected, grid


def bootstrap_ci(values, seed, reps):
    x = list(values)
    if len(x) < 2:
        return None
    rng = random.Random(seed)
    draws = sorted(sum(x[rng.randrange(len(x))] for _ in x) / len(x) for _ in range(reps))
    return [draws[int(0.025*reps)], draws[min(int(0.975*reps), reps-1)]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--embeddings-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--projection-dim", type=int, default=base.PROJECTION_DIM)
    ap.add_argument("--projection-seed", type=int, default=20261118)
    ap.add_argument("--bootstrap-replicates", type=int, default=20000)
    a = ap.parse_args()
    out = a.output.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory must be absent or empty: {out}")

    input_dir = a.input_dir.expanduser().resolve()
    rows, embeddings, variant_ids, rows_path, index_path, embedding_path = base.parse_inputs(
        input_dir, a.embeddings_dir.expanduser().resolve())
    raw_rows = base.read_csv(rows_path)
    if len(raw_rows) != len(rows):
        raise ValueError("Parsed rows do not match source forecast row count")
    for i, (r, raw) in enumerate(zip(rows, raw_rows), 2):
        if "n_target_clones" not in raw:
            raise ValueError("forecast_rows.csv needs n_target_clones for this sensitivity analysis")
        n = int(raw["n_target_clones"])
        if n < 1:
            raise ValueError(f"Invalid n_target_clones at source row {i}: {n}")
        r["n_target"] = n
        r["k_target"] = int(round(r["q"] * n))
        if not math.isclose(r["k_target"] / n, r["q"], abs_tol=1e-8):
            raise ValueError(f"target_frequency is not an integer count fraction at row {i}")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embedding matrix contains non-finite values")
    if a.projection_dim < 1 or a.projection_dim > embeddings.shape[1]:
        raise ValueError("projection-dim must be positive and no greater than embedding width")
    rng = np.random.default_rng(a.projection_seed)
    projection = rng.normal(0, 1/math.sqrt(a.projection_dim),
                            size=(embeddings.shape[1], a.projection_dim)).astype(np.float32)
    z = np.asarray(embeddings @ projection, dtype=np.float64)
    for r in rows:
        r["z"] = z[r["variant_index"]]

    prediction_rows, selection_rows = [], []
    by_model_predictions = {}
    for heldout in sorted({r["population"] for r in rows}):
        train = [r for r in rows if r["population"] != heldout and r["split"] == "TRAIN_CANDIDATE"]
        cal = [r for r in rows if r["population"] != heldout and r["split"] == "CALIBRATION_CANDIDATE"]
        test = [r for r in rows if r["population"] == heldout and r["split"] == "CONFIRMATION_CANDIDATE"]
        if not train or not cal or not test:
            raise ValueError(f"Missing required split for held-out population {heldout}")

        hcoef = base.fit_history(train)
        hcal = base.pred_history(cal, hcoef)
        hblend, hgrid = base.calibrate_blend(cal, hcal)
        hist_pred = np.clip((1-hblend)*np.asarray([r["p"] for r in test])
                            + hblend*base.pred_history(test, hcoef), 0, 1)
        evo_pred, evo_sel, evo_grid = fit_ridge_model(train, cal, test, lambda r: r["z"])
        joint_pred, joint_sel, joint_grid = fit_ridge_model(
            train, cal, test, lambda r: np.concatenate(([r["p"], r["trend"]], r["z"])))
        model_preds = {"persistence": np.asarray([r["p"] for r in test]),
                       "history_only": hist_pred, "evo2_only": evo_pred,
                       "evo2_plus_history": joint_pred}
        for model, pred in model_preds.items():
            by_model_predictions.setdefault(model, []).extend(
                {**r, "prediction": float(p), "model": model} for r, p in zip(test, pred))
            for r, p in zip(test, pred):
                prediction_rows.append({"heldout_population": heldout, "population": r["population"],
                    "cutoff_generation": r["cutoff"], "target_generation": r["target"],
                    "variant_id": r["variant_id"], "variant_index": r["variant_index"],
                    "model": model, "cutoff_frequency": r["p"], "trend_per_10k": r["trend"],
                    "target_frequency": r["q"], "target_alt_clone_count": r["k_target"],
                    "n_target_clones": r["n_target"], "predicted_frequency": float(p)})
        selection_rows.append({"heldout_population": heldout, "n_train_rows": len(train),
            "n_calibration_rows": len(cal), "n_confirmation_rows": len(test),
            "history_blend": hblend, "evo2_only_penalty": evo_sel["ridge_penalty"],
            "evo2_only_blend": evo_sel["blend_weight"], "joint_penalty": joint_sel["ridge_penalty"],
            "joint_blend": joint_sel["blend_weight"], "history_calibration_grid": json.dumps(hgrid),
            "evo2_only_calibration_grid": json.dumps(evo_grid), "joint_calibration_grid": json.dumps(joint_grid)})
        print(f"Completed held-out population {heldout}", flush=True)

    # Summaries preserve population as the independent cluster and macro unit.
    pop_metric = {}
    for model, pred_rows in by_model_predictions.items():
        # Rebuild original parsed-row alignment by unique (fold, variant) keys.
        key_to_pred = {(r["fold"], r["variant_id"]): r["prediction"] for r in pred_rows}
        ordered_rows = [r for r in rows if r["split"] == "CONFIRMATION_CANDIDATE"]
        preds = [key_to_pred[(r["fold"], r["variant_id"])] for r in ordered_rows]
        pop_metric[model] = {}
        for weighting in WEIGHTINGS:
            # population-specific macro scores are required for paired cluster bootstrap
            for pop in sorted({r["population"] for r in ordered_rows}):
                sub = [r for r in ordered_rows if r["population"] == pop]
                sub_pred = [key_to_pred[(r["fold"], r["variant_id"])] for r in sub]
                pop_metric[model][(weighting, pop)] = aggregate(sub, sub_pred, weighting)

    summary_rows = []
    for weighting in WEIGHTINGS:
        for model in MODELS:
            for metric_i, metric in enumerate(base.METRICS):
                vals = [pop_metric[model][(weighting, p)][metric]
                        for p in sorted({r["population"] for r in rows if r["split"] == "CONFIRMATION_CANDIDATE"})]
                summary_rows.append({"weighting": weighting, "model": model, "metric": metric,
                    "mean_over_populations": base.mean(vals),
                    "population_cluster_bootstrap_95pct_ci": json.dumps(
                        bootstrap_ci(vals, 99001 + WEIGHTINGS.index(weighting)*100 + MODELS.index(model)*10 + metric_i,
                                     a.bootstrap_replicates)),
                    "n_population_clusters": len(vals)})

    delta_rows = []
    comparisons = [("evo2_plus_history", "history_only"), ("evo2_plus_history", "evo2_only"),
                   ("evo2_only", "persistence"), ("history_only", "persistence"),
                   ("evo2_plus_history", "persistence")]
    pops = sorted({r["population"] for r in rows if r["split"] == "CONFIRMATION_CANDIDATE"})
    for wi, weighting in enumerate(WEIGHTINGS):
        for mi, metric in enumerate(base.METRICS):
            for ci, (left, right) in enumerate(comparisons):
                diffs = [pop_metric[left][(weighting, pop)][metric]
                         - pop_metric[right][(weighting, pop)][metric] for pop in pops]
                delta_rows.append({"weighting": weighting, "comparison": f"{left}_minus_{right}",
                    "metric": metric, "mean_paired_population_delta": base.mean(diffs),
                    "population_cluster_bootstrap_95pct_ci": json.dumps(bootstrap_ci(
                        diffs, 99101 + wi*100 + mi*10 + ci, a.bootstrap_replicates)),
                    "n_population_clusters": len(diffs), "negative_favors_left": True})

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out/"heldout_site_predictions.csv", prediction_rows, list(prediction_rows[0]))
    write_csv(out/"weighting_sensitivity_summary.csv", summary_rows, list(summary_rows[0]))
    write_csv(out/"paired_population_deltas.csv", delta_rows, list(delta_rows[0]))
    write_csv(out/"outer_model_selections.csv", selection_rows, list(selection_rows[0]))
    report = {"status": "EXPLORATORY_TARGET_CLONE_WEIGHT_SENSITIVITY",
        "task": "Re-score nested population-LOPO held-out predictions with site-equal versus target-clone-count weighting.",
        "refit": "Ridge heads refit/calibrated within the same outer population folds using the original site-macro calibration criterion; no Evo2 inference is run.",
        "weighting_definitions": {
            "site_equal": "Within each fold, each cutoff-defined SNP allele receives weight 1; folds and populations are then macro-averaged.",
            "target_clone_weighted": "Within each fold, each SNP allele score is weighted by n_target_clones; folds and populations are then macro-averaged."},
        "metrics": {"frequency_mae": "|predicted_frequency - observed_target_frequency|",
            "target_clone_brier": "q*(1-pred)^2 + (1-q)*pred^2",
            "target_clone_log_loss": "-q*log(clip(pred,1e-6,1-1e-6))-(1-q)*log(1-clip(pred,1e-6,1-1e-6))"},
        "n_population_clusters": len(pops), "n_confirmation_site_allele_rows": sum(1 for r in rows if r["split"]=="CONFIRMATION_CANDIDATE"),
        "n_confirmation_folds": len({r["fold"] for r in rows if r["split"]=="CONFIRMATION_CANDIDATE"}),
        "n_unique_variants": len(variant_ids), "projection_dimension": a.projection_dim,
        "projection_seed": a.projection_seed, "input_sha256": {
            "forecast_rows": base.sha256_file(rows_path), "variant_index": base.sha256_file(index_path),
            "embedding_matrix": base.sha256_file(embedding_path)},
        "overall_metrics": summary_rows, "paired_deltas": delta_rows,
        "outputs": ["heldout_site_predictions.csv", "weighting_sensitivity_summary.csv",
                    "paired_population_deltas.csv", "outer_model_selections.csv"],
        "limitations": ["Clone-count weighting changes the estimand; report alongside site-equal results.",
            "The 12 populations, not the SNP rows, are the population-level uncertainty clusters.",
            "This is internal development evidence, not a blind locked biological test."]}
    (out/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
