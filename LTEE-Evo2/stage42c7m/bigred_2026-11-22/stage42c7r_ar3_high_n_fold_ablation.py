#!/usr/bin/env python3
"""Nested-LOPO ablation of Ara-3 training/calibration folds with >2 clones.

Compares the original fit/calibration rows against a condition that removes
only Ara-3 target folds at generations 15k, 20k, 25k, and 30k with n>2 from
outer-training and calibration data. Held-out confirmation folds are unchanged.
This tests the influence of those folds as a bundle, not clone count alone.
CPU-only; reuses stored C7M embeddings.
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

import stage42c7m_evo2_lopo_eval as base
import stage42c7n_evo2_history_ablation as c7n

MODELS = ("persistence", "history_only", "evo2_only", "evo2_plus_history")
CONDITIONS = ("all_training_folds", "omit_ar3_high_n_train_cal")
HIGH_N_TARGETS = {15000, 20000, 25000, 30000}


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def boot_ci(values, seed, reps):
    vals = list(values)
    if len(vals) < 2:
        return None
    rng = random.Random(seed)
    draws = sorted(sum(vals[rng.randrange(len(vals))] for _ in vals) / len(vals)
                   for _ in range(reps))
    return [draws[int(.025*reps)], draws[min(int(.975*reps), reps-1)]]


def make_features(rows, mode):
    if mode == "evo2_only":
        return np.vstack([r["z"] for r in rows])
    return np.vstack([np.concatenate(([r["p"], r["trend"]], r["z"])) for r in rows])


def fit_one_condition(train, calibration, test):
    result = {}
    # Persistence
    result["persistence"] = np.asarray([r["p"] for r in test], dtype=float)

    # History-only model
    hcoef = base.fit_history(train)
    hcal_raw = base.pred_history(calibration, hcoef)
    hblend, hgrid = base.calibrate_blend(calibration, hcal_raw)
    htest_raw = base.pred_history(test, hcoef)
    result["history_only"] = np.clip((1-hblend)*np.asarray([r["p"] for r in test])
                                      + hblend*htest_raw, 0, 1)

    selected = {}
    for model in ("evo2_only", "evo2_plus_history"):
        tx = make_features(train, model)
        cx = make_features(calibration, model)
        vx = make_features(test, model)
        txs, cxs, _, _ = base.standardize_features(tx, cx, train)
        _, vxs, _, _ = base.standardize_features(tx, vx, train)
        y = np.asarray([r["q"] for r in train], dtype=float)
        weights = base.hierarchical_weights(train)
        coefs, grid = {}, []
        for penalty in base.RIDGE_GRID:
            coef = base.fit_ridge(txs, y, weights, penalty)
            coefs[penalty] = coef
            raw = base.ridge_predict(cxs, coef)
            for blend in base.BLENDS:
                pred = np.clip((1-blend)*np.asarray([r["p"] for r in calibration])
                               + blend*raw, 0, 1)
                grid.append({"ridge_penalty": penalty, "blend_weight": blend,
                             **base.macro_metrics(calibration, pred)})
        chosen = min(grid, key=lambda x: x["target_clone_log_loss"])
        raw_test = base.ridge_predict(vxs, coefs[chosen["ridge_penalty"]])
        result[model] = np.clip((1-chosen["blend_weight"])*np.asarray([r["p"] for r in test])
                                + chosen["blend_weight"]*raw_test, 0, 1)
        selected[model] = {"ridge_penalty": chosen["ridge_penalty"],
                           "blend_weight": chosen["blend_weight"],
                           "calibration_log_loss": chosen["target_clone_log_loss"],
                           "calibration_grid": grid}
    selected["history_only"] = {"blend_weight": hblend, "coefficients": hcoef.tolist(),
                                "calibration_grid": hgrid}
    return result, selected


def score_pop(test_rows, pred_by_model):
    out = {}
    for model, preds in pred_by_model.items():
        fold_metrics = c7n.metrics_by_fold(test_rows, preds)
        per_pop = {}
        for pop in sorted({fold[0] for fold in fold_metrics}):
            folds = [v for fold, v in fold_metrics.items() if fold[0] == pop]
            per_pop[pop] = {metric: base.mean([r[metric] for r in folds])
                            for metric in base.METRICS}
        out[model] = per_pop
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--embeddings-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--projection-dim", type=int, default=base.PROJECTION_DIM)
    p.add_argument("--projection-seed", type=int, default=20261118)
    p.add_argument("--bootstrap-replicates", type=int, default=20000)
    a = p.parse_args()
    out = a.output.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory must be absent or empty: {out}")

    rows, embeddings, variant_ids, rows_path, index_path, embedding_path = base.parse_inputs(
        a.input_dir.expanduser().resolve(), a.embeddings_dir.expanduser().resolve())
    raw = read_csv(rows_path)
    if len(raw) != len(rows) or "n_target_clones" not in raw[0]:
        raise ValueError("forecast_rows.csv must align with parsed rows and include n_target_clones")
    for r, rr in zip(rows, raw):
        r["n_target"] = int(rr["n_target_clones"])
    rng = np.random.default_rng(a.projection_seed)
    if a.projection_dim < 1 or a.projection_dim > embeddings.shape[1]:
        raise ValueError("Invalid projection dimension")
    projection = rng.normal(0, 1/np.sqrt(a.projection_dim),
                           size=(embeddings.shape[1], a.projection_dim)).astype(np.float32)
    projected = np.asarray(embeddings @ projection, dtype=np.float64)
    for r in rows:
        r["z"] = projected[r["variant_index"]]

    # Identify high-count source folds directly from row metadata and verify the expected list.
    high_folds = sorted({(r["population"], r["cutoff"], r["target"])
                         for r in rows if r["population"] == "Ara-3"
                         and r["target"] in HIGH_N_TARGETS and r["n_target"] > 2})
    high_fold_counts = {}
    for fold in high_folds:
        rs = [r for r in rows if r["fold"] == fold]
        high_fold_counts[f"{fold[0]}:{fold[1]}->{fold[2]}"] = {
            "n_target_clones": rs[0]["n_target"], "n_site_allele_rows": len(rs)}

    population_metrics = []
    fold_metrics = []
    selections = []
    for heldout in sorted({r["population"] for r in rows}):
        outer_train = [r for r in rows if r["population"] != heldout and r["split"] == "TRAIN_CANDIDATE"]
        outer_cal = [r for r in rows if r["population"] != heldout and r["split"] == "CALIBRATION_CANDIDATE"]
        test = [r for r in rows if r["population"] == heldout and r["split"] == "CONFIRMATION_CANDIDATE"]
        if not outer_train or not outer_cal or not test:
            raise ValueError(f"Missing split for held-out population {heldout}")
        scores_by_condition = {}
        for condition in CONDITIONS:
            if condition == "all_training_folds":
                train, cal = outer_train, outer_cal
            else:
                drop = lambda r: (r["population"] == "Ara-3" and r["target"] in HIGH_N_TARGETS
                                  and r["n_target"] > 2)
                train = [r for r in outer_train if not drop(r)]
                cal = [r for r in outer_cal if not drop(r)]
            if not train or not cal:
                raise ValueError(f"Condition {condition} empties train/cal for heldout {heldout}")
            predictions, chosen = fit_one_condition(train, cal, test)
            scores_by_condition[condition] = score_pop(test, predictions)
            # Fold-level scores preserve the test fold structure.
            for model, pred in predictions.items():
                fm = c7n.metrics_by_fold(test, pred)
                for fold, vals in fm.items():
                    fold_metrics.append({"heldout_population": heldout, "condition": condition,
                        "model": model, "population": fold[0], "cutoff_generation": fold[1],
                        "target_generation": fold[2], "n_site_allele_rows": sum(r["fold"] == fold for r in test),
                        **vals})
            drop_train = [r for r in outer_train if condition == "omit_ar3_high_n_train_cal"
                          and r["population"] == "Ara-3" and r["target"] in HIGH_N_TARGETS and r["n_target"] > 2]
            drop_cal = [r for r in outer_cal if condition == "omit_ar3_high_n_train_cal"
                        and r["population"] == "Ara-3" and r["target"] in HIGH_N_TARGETS and r["n_target"] > 2]
            selections.append({"heldout_population": heldout, "condition": condition,
                "n_train_rows": len(train), "n_calibration_rows": len(cal), "n_test_rows": len(test),
                "n_removed_train_high_n_rows": len(drop_train), "n_removed_cal_high_n_rows": len(drop_cal),
                "n_removed_high_n_folds": len({r["fold"] for r in drop_train + drop_cal}),
                "selected_models_json": json.dumps(chosen)})
        # score_pop returns model -> population -> metric. Iterate the
        # population keys under one model, not the outer model keys.
        for pop in sorted(scores_by_condition["all_training_folds"]["persistence"]):
            row = {"heldout_population": heldout, "scored_population": pop}
            for condition in CONDITIONS:
                for model in MODELS:
                    for metric in base.METRICS:
                        row[f"{condition}__{model}__{metric}"] = scores_by_condition[condition][model][pop][metric]
            population_metrics.append(row)

    # Paired cluster bootstrap for each model-condition comparison and for the change in the key Evo2 gap.
    scored_pops = sorted({r["scored_population"] for r in population_metrics})
    # One row per heldout/scored population; in LOPO these are the same population identifiers.
    # Build macro population scores by averaging repeated records if present.
    pop_scores = {}
    for condition in CONDITIONS:
        for model in MODELS:
            for metric in base.METRICS:
                vals_by_pop = defaultdict(list)
                for r in population_metrics:
                    vals_by_pop[r["scored_population"]].append(r[f"{condition}__{model}__{metric}"])
                pop_scores[(condition, model, metric)] = {
                    pop: base.mean(vals) for pop, vals in vals_by_pop.items()}
    delta_rows = []
    comparisons = [("evo2_plus_history", "history_only"), ("evo2_only", "persistence"),
                   ("evo2_plus_history", "persistence")]
    for metric_i, metric in enumerate(base.METRICS):
        for ci, (left, right) in enumerate(comparisons):
            for condition in CONDITIONS:
                ds = [pop_scores[(condition, left, metric)][pop] - pop_scores[(condition, right, metric)][pop]
                      for pop in scored_pops]
                delta_rows.append({"analysis": "model_comparison", "condition": condition,
                    "comparison": f"{left}_minus_{right}", "metric": metric,
                    "mean_paired_population_delta": base.mean(ds),
                    "population_cluster_bootstrap_95pct_ci": json.dumps(
                        boot_ci(ds, 78101 + metric_i*100 + ci*10 + CONDITIONS.index(condition), a.bootstrap_replicates)),
                    "n_population_clusters": len(ds), "negative_favors_left": True})
            # Change in Evo2+history incremental gain: full condition delta minus ablated condition delta.
            d_full = [pop_scores[("all_training_folds", left, metric)][pop]
                      - pop_scores[("all_training_folds", right, metric)][pop] for pop in scored_pops]
            d_drop = [pop_scores[("omit_ar3_high_n_train_cal", left, metric)][pop]
                      - pop_scores[("omit_ar3_high_n_train_cal", right, metric)][pop] for pop in scored_pops]
            changes = [x-y for x, y in zip(d_full, d_drop)]
            delta_rows.append({"analysis": "change_in_model_gap_full_minus_ablated", "condition": "paired_change",
                "comparison": f"change_{left}_minus_{right}", "metric": metric,
                "mean_paired_population_delta": base.mean(changes),
                "population_cluster_bootstrap_95pct_ci": json.dumps(
                    boot_ci(changes, 78201 + metric_i*100 + ci, a.bootstrap_replicates)),
                "n_population_clusters": len(changes), "negative_favors_full_condition": True})

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out/"population_metrics_by_condition.csv", population_metrics, list(population_metrics[0]))
    write_csv(out/"confirmation_fold_metrics.csv", fold_metrics, list(fold_metrics[0]))
    write_csv(out/"outer_fit_selections.csv", selections, list(selections[0]))
    write_csv(out/"paired_deltas.csv", delta_rows, list(delta_rows[0]))
    report = {"status": "EXPLORATORY_AR3_HIGH_N_TRAIN_CAL_FOLD_ABLATION",
        "question": "Do Ara-3 target folds with more than two sampled clones influence held-out predictions?",
        "conditions": {"all_training_folds": "Original train/calibration rows.",
            "omit_ar3_high_n_train_cal": "Remove Ara-3 target-generation folds 15k, 20k, 25k, and 30k only when n_target_clones>2; confirmation rows unchanged."},
        "high_n_folds_found": high_fold_counts,
        "outer_validation": "Nested leave-one-population-out; each held-out population confirmation folds are identical between conditions.",
        "n_population_clusters": len(scored_pops), "n_confirmation_folds": len({r["fold"] for r in rows if r["split"]=="CONFIRMATION_CANDIDATE"}),
        "n_forecast_rows": len(rows), "n_unique_variants": len(variant_ids),
        "projection_dimension": a.projection_dim, "projection_seed": a.projection_seed,
        "calibration_criterion": "population-macro target-clone log loss",
        "interpretation_limit": "This removes complete Ara-3 folds; it cannot separate clone-count precision from Ara-3 population, generation, or fold-specific biology. A true clone-count-only test would downsample the original clone genotypes within those same folds.",
        "outputs": ["population_metrics_by_condition.csv", "confirmation_fold_metrics.csv",
                    "outer_fit_selections.csv", "paired_deltas.csv"],
        "limitations": ["Only 12 population clusters.", "Previously viewed confirmation outcomes make this development evidence, not a blind test.",
            "No Evo2 inference is run; stored frozen embeddings are reused."]}
    (out/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
