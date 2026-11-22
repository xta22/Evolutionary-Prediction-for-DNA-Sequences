#!/usr/bin/env python3
"""C7S: one CPU job for error breakdown and Evo2-embedding shuffle control.

Part A stratifies the already saved held-out C7O predictions by population,
time fold, and observed target frequency. Part B reruns the same nested LOPO
Evo2+history ridge pipeline using true variant embeddings and 20 mappings
that randomly reassign embeddings across variant IDs. No Evo2 inference runs.
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

METRICS = base.METRICS
MODEL_DIAGNOSTICS = ("persistence", "history_only", "evo2_only", "evo2_plus_history")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def bootstrap_ci(values, seed, reps=20000):
    vals = list(values)
    if len(vals) < 2:
        return None
    rng = random.Random(seed)
    draws = sorted(sum(vals[rng.randrange(len(vals))] for _ in vals) / len(vals)
                   for _ in range(reps))
    return [draws[int(.025*reps)], draws[min(int(.975*reps), reps-1)]]


def observed_score_diagnostics(prediction_path):
    rows = read_csv(prediction_path)
    required = {"population", "cutoff_generation", "target_generation", "model",
                "target_frequency", "predicted_frequency"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Prediction file must contain {sorted(required)}")
    grouped = defaultdict(lambda: {"pred": [], "q": [], "score": defaultdict(list)})
    for r in rows:
        pop = r["population"]
        fold = (int(r["cutoff_generation"]), int(r["target_generation"]))
        model = r["model"]
        q = float(r["target_frequency"])
        pred = float(r["predicted_frequency"])
        key = (pop, fold[0], fold[1], q, model)
        rec = grouped[key]
        rec["pred"].append(pred)
        rec["q"].append(q)
        fake = {"q": q}
        for metric, val in base.score(fake, pred).items():
            rec["score"][metric].append(val)
    out = []
    for (pop, cutoff, target, q, model), rec in sorted(grouped.items()):
        out.append({"population": pop, "cutoff_generation": cutoff,
            "target_generation": target, "observed_target_frequency": q, "model": model,
            "n_site_allele_rows": len(rec["pred"]),
            "mean_predicted_frequency": float(np.mean(rec["pred"])),
            **{metric: float(np.mean(rec["score"][metric])) for metric in METRICS}})
    return out, len(rows)


def macro_by_population(test, pred):
    fold_metrics = c7n.metrics_by_fold(test, pred)
    by_pop = defaultdict(lambda: defaultdict(list))
    for fold, metrics in fold_metrics.items():
        pop = fold[0]
        for metric, value in metrics.items():
            by_pop[pop][metric].append(value)
    return {pop: {metric: float(np.mean(vals[metric])) for metric in METRICS}
            for pop, vals in by_pop.items()}


def fit_joint_for_outer(rows, projected, heldout, permutations, seed):
    train = [r for r in rows if r["population"] != heldout and r["split"] == "TRAIN_CANDIDATE"]
    cal = [r for r in rows if r["population"] != heldout and r["split"] == "CALIBRATION_CANDIDATE"]
    test = [r for r in rows if r["population"] == heldout and r["split"] == "CONFIRMATION_CANDIDATE"]
    if not train or not cal or not test:
        raise ValueError(f"Missing split for held-out population {heldout}")

    def score_mapping(z_by_variant, condition):
        for r in rows:
            r["z_diag"] = z_by_variant[r["variant_index"]]
        pred, chosen, _grid = c7n.fit_calibrate_predict(
            train, cal, test,
            lambda r: np.concatenate(([r["p"], r["trend"]], r["z_diag"])))
        return pred, chosen

    result = []
    pred_true, selected_true = score_mapping(projected, "true")
    for pop, metrics in macro_by_population(test, pred_true).items():
        result.extend({"condition": "true_embedding", "permutation": 0,
                       "population": pop, "metric": metric, "score": value,
                       "selected_penalty": selected_true["ridge_penalty"],
                       "selected_blend": selected_true["blend_weight"]}
                      for metric, value in metrics.items())

    rng = np.random.default_rng(seed)
    n_variants = projected.shape[0]
    for perm_i in range(1, permutations + 1):
        shuffled = projected[rng.permutation(n_variants)]
        pred, selected = score_mapping(shuffled, f"shuffle_{perm_i}")
        for pop, metrics in macro_by_population(test, pred).items():
            result.extend({"condition": "shuffled_embedding", "permutation": perm_i,
                           "population": pop, "metric": metric, "score": value,
                           "selected_penalty": selected["ridge_penalty"],
                           "selected_blend": selected["blend_weight"]}
                          for metric, value in metrics.items())
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--embeddings-dir", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True,
                   help="C7O heldout_site_predictions.csv")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-permutations", type=int, default=20)
    p.add_argument("--seed", type=int, default=74821)
    p.add_argument("--projection-dim", type=int, default=base.PROJECTION_DIM)
    p.add_argument("--projection-seed", type=int, default=20261118)
    p.add_argument("--bootstrap-replicates", type=int, default=20000)
    a = p.parse_args()
    out = a.output.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory must be absent or empty: {out}")
    if a.n_permutations < 5:
        raise ValueError("Use at least five embedding permutations")

    stratified, n_prediction_rows = observed_score_diagnostics(a.predictions.expanduser().resolve())
    rows, embedding, variant_ids, rows_path, index_path, embedding_path = base.parse_inputs(
        a.input_dir.expanduser().resolve(), a.embeddings_dir.expanduser().resolve())
    if not np.isfinite(embedding).all():
        raise ValueError("Embedding matrix contains non-finite values")
    if a.projection_dim < 1 or a.projection_dim > embedding.shape[1]:
        raise ValueError("Invalid projection dimension")
    projection_rng = np.random.default_rng(a.projection_seed)
    projection = projection_rng.normal(0, 1/np.sqrt(a.projection_dim),
        size=(embedding.shape[1], a.projection_dim)).astype(np.float32)
    projected = np.asarray(embedding @ projection, dtype=np.float64)
    for r in rows:
        r["z_diag"] = projected[r["variant_index"]]

    permutation_rows = []
    populations = sorted({r["population"] for r in rows})
    for h, heldout in enumerate(populations):
        print(f"Evaluating held-out population {heldout} ({h+1}/{len(populations)})", flush=True)
        permutation_rows.extend(fit_joint_for_outer(rows, projected, heldout,
                                                     a.n_permutations,
                                                     a.seed + h*100003))

    # Compare actual embeddings with the permutation distribution at the population level.
    pop_value = defaultdict(dict)
    for r in permutation_rows:
        pop_value[(r["condition"], r["permutation"], r["metric"])][r["population"]] = float(r["score"])
    summaries = []
    for metric_i, metric in enumerate(METRICS):
        true_by_pop = pop_value[("true_embedding", 0, metric)]
        perm_ids = sorted({r["permutation"] for r in permutation_rows
                           if r["condition"] == "shuffled_embedding" and r["metric"] == metric})
        mean_perm_by_pop = {pop: float(np.mean([
            pop_value[("shuffled_embedding", perm_i, metric)][pop] for perm_i in perm_ids]))
            for pop in populations}
        deltas = [true_by_pop[pop] - mean_perm_by_pop[pop] for pop in populations]
        true_macro = float(np.mean([true_by_pop[pop] for pop in populations]))
        permutation_macros = [float(np.mean([
            pop_value[("shuffled_embedding", perm_i, metric)][pop] for pop in populations]))
            for perm_i in perm_ids]
        p_lower = (1 + sum(x <= true_macro for x in permutation_macros)) / (len(permutation_macros)+1)
        summaries.append({"metric": metric, "true_embedding_macro_score": true_macro,
            "mean_shuffled_macro_score": float(np.mean(permutation_macros)),
            "shuffled_macro_q025": float(np.quantile(permutation_macros, .025)),
            "shuffled_macro_q975": float(np.quantile(permutation_macros, .975)),
            "true_minus_mean_shuffled": float(np.mean(deltas)),
            "population_cluster_bootstrap_95pct_ci_for_true_minus_shuffle_mean": json.dumps(
                bootstrap_ci(deltas, 79821+metric_i, a.bootstrap_replicates)),
            "permutation_p_lower_better": p_lower,
            "n_embedding_permutations": len(permutation_macros),
            "n_population_clusters": len(populations)})

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out/"stratified_error_diagnostics.csv", stratified, list(stratified[0]))
    write_csv(out/"embedding_shuffle_population_scores.csv", permutation_rows,
              list(permutation_rows[0]))
    write_csv(out/"embedding_shuffle_summary.csv", summaries, list(summaries[0]))
    report = {"status": "EXPLORATORY_EVO2_ERROR_AND_EMBEDDING_SHUFFLE_DIAGNOSTICS",
        "parts": {"error_breakdown": "Existing held-out predictions, grouped by population, fold, and observed target frequency.",
            "embedding_control": "Refit/calibrate the same nested-LOPO Evo2+history ridge head after shuffling the embedding-to-variant mapping; history features and folds are unchanged."},
        "n_prediction_rows_for_error_breakdown": n_prediction_rows,
        "n_outer_populations": len(populations), "n_confirmation_folds": len({r["fold"] for r in rows if r["split"]=="CONFIRMATION_CANDIDATE"}),
        "n_forecast_rows": len(rows), "n_unique_variants": len(variant_ids),
        "n_embedding_permutations": a.n_permutations,
        "permutation_seed": a.seed, "projection_dimension": a.projection_dim,
        "projection_seed": a.projection_seed,
        "permutation_interpretation": "Lower true score than shuffled scores suggests variant-specific Evo2 representations carry information beyond the fixed history features under this pipeline. Similar scores mean no detectable added signal here.",
        "p_value_note": "Monte Carlo lower-tail permutation p-value over embedding shuffles; not a population-level biological p-value.",
        "limitations": ["Only 12 population clusters; intervals are exploratory.",
            "Embedding shuffling tests variant-to-embedding association, not whether Evo2 is useful in every possible representation.",
            "No Evo2 inference is run; stored 7B embeddings are reused.",
            "Confirmation outcomes have already been inspected; this is development diagnosis, not a blind test."],
        "input_sha256": {"forecast_rows": base.sha256_file(rows_path),
            "variant_index": base.sha256_file(index_path), "embedding_matrix": base.sha256_file(embedding_path)},
        "outputs": ["stratified_error_diagnostics.csv", "embedding_shuffle_population_scores.csv",
                    "embedding_shuffle_summary.csv"]}
    (out/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
