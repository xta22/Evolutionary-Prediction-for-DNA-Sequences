#!/usr/bin/env python3
"""Nested population-LOPO test of incremental Evo2 and history signal.

Compares persistence, history-only, Evo2 embedding-only, and Evo2 plus history.
Uses the exact C7M forecast bundle and frozen embeddings. All scaling, model
fitting, and calibration happen within each outer training-population set.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import stage42c7m_evo2_lopo_eval as base


MODELS = ("persistence", "history_only", "evo2_only", "evo2_plus_history")


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fit_calibrate_predict(train, calibration, test, feature_fn):
    tx = np.vstack([feature_fn(r) for r in train])
    cx = np.vstack([feature_fn(r) for r in calibration])
    vx = np.vstack([feature_fn(r) for r in test])
    txs, cxs, _, _ = base.standardize_features(tx, cx, train)
    _, vxs, _, _ = base.standardize_features(tx, vx, train)
    y = np.asarray([r["q"] for r in train], dtype=np.float64)
    weights = base.hierarchical_weights(train)
    grid = []
    models = {}
    for penalty in base.RIDGE_GRID:
        coef = base.fit_ridge(txs, y, weights, penalty)
        raw_cal = base.ridge_predict(cxs, coef)
        models[penalty] = (coef, raw_cal)
        for blend in base.BLENDS:
            pred = np.clip((1 - blend) * np.asarray([r["p"] for r in calibration])
                           + blend * raw_cal, 0.0, 1.0)
            grid.append({"ridge_penalty": penalty, "blend_weight": blend,
                         **base.macro_metrics(calibration, pred)})
    chosen = min(grid, key=lambda x: x["target_clone_log_loss"])
    coef, _ = models[chosen["ridge_penalty"]]
    raw_test = base.ridge_predict(vxs, coef)
    pred_test = np.clip((1 - chosen["blend_weight"]) * np.asarray([r["p"] for r in test])
                        + chosen["blend_weight"] * raw_test, 0.0, 1.0)
    return pred_test, chosen, grid


def metrics_by_fold(rows, pred):
    by_fold = {}
    for row, p in zip(rows, pred):
        by_fold.setdefault(row["fold"], {m: [] for m in base.METRICS})
        for m, value in base.score(row, p).items():
            by_fold[row["fold"]][m].append(value)
    return {fold: {m: base.mean(vs) for m, vs in vals.items()}
            for fold, vals in by_fold.items()}


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
    rows, embedding, variant_ids, rows_path, index_path, emb_path = base.parse_inputs(
        a.input_dir.expanduser().resolve(), a.embeddings_dir.expanduser().resolve())
    if not np.isfinite(embedding).all():
        raise ValueError("Non-finite embeddings")
    if a.projection_dim < 1 or a.projection_dim > embedding.shape[1]:
        raise ValueError("Invalid projection dimension")
    rng = np.random.default_rng(a.projection_seed)
    proj = rng.normal(0, 1 / np.sqrt(a.projection_dim),
                      size=(embedding.shape[1], a.projection_dim)).astype(np.float32)
    projected = np.asarray(embedding @ proj, dtype=np.float64)
    for r in rows:
        r["z"] = projected[r["variant_index"]]

    pop_results, fold_results, selections = [], [], []
    for heldout in sorted({r["population"] for r in rows}):
        tr = [r for r in rows if r["population"] != heldout and r["split"] == "TRAIN_CANDIDATE"]
        ca = [r for r in rows if r["population"] != heldout and r["split"] == "CALIBRATION_CANDIDATE"]
        te = [r for r in rows if r["population"] == heldout and r["split"] == "CONFIRMATION_CANDIDATE"]
        if not tr or not ca or not te:
            raise ValueError(f"Missing split for outer heldout population {heldout}")

        hcoef = base.fit_history(tr)
        hcal_raw = base.pred_history(ca, hcoef)
        hblend, hgrid = base.calibrate_blend(ca, hcal_raw)
        hp = np.clip((1-hblend)*np.array([r["p"] for r in te]) + hblend*base.pred_history(te,hcoef),0,1)
        ep, esel, egrid = fit_calibrate_predict(tr, ca, te, lambda r: r["z"])
        jx, jsel, jgrid = fit_calibrate_predict(
            tr, ca, te, lambda r: np.concatenate(([r["p"], r["trend"]], r["z"])))
        preds = {"persistence": np.asarray([r["p"] for r in te]),
                 "history_only": hp, "evo2_only": ep, "evo2_plus_history": jx}
        by_model_fold = {m: metrics_by_fold(te, p) for m, p in preds.items()}
        pop_summary = {"heldout_population": heldout, "n_test_folds": len(by_model_fold["persistence"])}
        for model in MODELS:
            for metric in base.METRICS:
                vals = [fm[metric] for fm in by_model_fold[model].values()]
                pop_summary[f"{model}_{metric}"] = base.mean(vals)
            for fold, fm in by_model_fold[model].items():
                fold_results.append({"heldout_population": heldout, "population": fold[0],
                                     "cutoff_generation": fold[1], "target_generation": fold[2],
                                     "model": model,
                                     "n_site_allele_rows": sum(1 for r in te if r["fold"] == fold),
                                     **fm})
        pop_results.append(pop_summary)
        selections.append({"heldout_population": heldout, "n_train_rows": len(tr),
                           "n_calibration_rows": len(ca), "n_test_rows": len(te),
                           "history_blend": hblend, "history_coef": json.dumps(hcoef.tolist()),
                           "evo2_only_penalty": esel["ridge_penalty"],
                           "evo2_only_blend": esel["blend_weight"],
                           "joint_penalty": jsel["ridge_penalty"],
                           "joint_blend": jsel["blend_weight"],
                           "history_calibration_grid": json.dumps(hgrid),
                           "evo2_only_calibration_grid": json.dumps(egrid),
                           "joint_calibration_grid": json.dumps(jgrid)})

    comparisons = [("evo2_plus_history", "history_only"),
                   ("evo2_plus_history", "evo2_only"),
                   ("evo2_only", "persistence"),
                   ("history_only", "persistence"),
                   ("evo2_plus_history", "persistence")]
    deltas = []
    for mi, metric in enumerate(base.METRICS):
        for ci, (left, right) in enumerate(comparisons):
            d = [r[f"{left}_{metric}"]-r[f"{right}_{metric}"] for r in pop_results]
            deltas.append({"comparison": f"{left}_minus_{right}", "metric": metric,
                           "mean_paired_population_delta": base.mean(d),
                           "population_cluster_bootstrap_95pct_ci": json.dumps(
                               base.bootstrap_ci(d, 88201+mi*100+ci, a.bootstrap_replicates)),
                           "n_heldout_population_clusters": len(d),
                           "negative_favors_left": True})
    overall = []
    for mi, model in enumerate(MODELS):
        for j, metric in enumerate(base.METRICS):
            vals = [r[f"{model}_{metric}"] for r in pop_results]
            overall.append({"model": model, "metric": metric,
                            "mean_over_heldout_populations": base.mean(vals),
                            "population_cluster_bootstrap_95pct_ci": json.dumps(
                                base.bootstrap_ci(vals, 89201+mi*10+j, a.bootstrap_replicates)),
                            "n_heldout_population_clusters": len(vals)})

    out.mkdir(parents=True, exist_ok=True)
    base.write_csv(out/"outer_population_metrics.csv", pop_results, list(pop_results[0]))
    base.write_csv(out/"outer_fold_metrics.csv", fold_results, list(fold_results[0]))
    base.write_csv(out/"outer_model_selections.csv", selections, list(selections[0]))
    base.write_csv(out/"paired_model_deltas.csv", deltas, list(deltas[0]))
    base.write_csv(out/"overall_model_metrics.csv", overall, list(overall[0]))
    report = {"status": "EXPLORATORY_FROZEN_EVO2_HISTORY_ABLATION_NESTED_LOPO",
              "question": "Does Evo2 context add forecast value beyond explicit history, and does history add value beyond Evo2?",
              "models": list(MODELS), "outer_validation": "population-level nested LOPO; held-out confirmation only",
              "n_population_clusters": len(pop_results), "n_forecast_rows": len(rows),
              "n_unique_variants": len(variant_ids),
              "n_confirmation_population_time_folds": len({r["fold"] for r in rows if r["split"]=="CONFIRMATION_CANDIDATE"}),
              "evo2": {"model":"evo2_7b", "layer":"blocks.28.mlp.l3",
                       "feature":"frozen alternate-minus-reference pooled embedding",
                       "projection_dimension":a.projection_dim,"projection_seed":a.projection_seed,
                       "ridge_grid":list(base.RIDGE_GRID),"persistence_blend_grid":list(base.BLENDS)},
              "calibration_criterion":"population-macro target-clone log loss",
              "input_sha256":{"forecast_rows":sha256_file(rows_path),"variant_index":sha256_file(index_path),"embedding":sha256_file(emb_path)},
              "overall_metrics":overall,"paired_deltas":deltas,
              "metric_definitions":{"frequency_mae":"mean across cutoff-defined site-allele rows within fold of |p_hat - q|, then equal-weight folds within population and populations overall",
                "target_clone_brier":"for target frequency q, q*(1-p_hat)^2 + (1-q)*p_hat^2; equals expected binary Brier over target clones",
                "target_clone_log_loss":"-q*ln(clip(p_hat,1e-6,1-1e-6))-(1-q)*ln(1-clip(p_hat,1e-6,1-1e-6)); equals mean binary cross-entropy over target clones"},
              "limitations":["Not whole-sequence likelihood or sequence generation; only cutoff-observed SNP alternate alleles are scored.","Only 12 population clusters; bootstrap reflects population-level variation.","Frozen local-context embeddings use focal SNP substitution on REL606 background; not full clone genome context.","Development evidence, not a blind locked biological test."]}
    (out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__ == "__main__":
    main()
