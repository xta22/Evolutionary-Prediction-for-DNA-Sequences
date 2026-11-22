#!/usr/bin/env python3
"""Estimate target-frequency uncertainty from the observed held-out clones.

Uses Jeffreys-prior Beta-Binomial uncertainty per population-time-SNP row and
re-scores fixed held-out predictions against posterior draws of latent allele
frequency. This is a sensitivity analysis, not a new independent test.
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

MODELS = ("persistence", "history_only", "evo2_only", "evo2_plus_history")
METRICS = ("frequency_mae", "target_clone_brier", "target_clone_log_loss")
TARGETS = ("latent_frequency", "replicate_sample_frequency")
KEYS = ("population", "cutoff_generation", "target_generation", "variant_id")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def quantile_ci(vals):
    return [float(x) for x in np.quantile(np.asarray(vals), [0.025, 0.5, 0.975])]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, required=True,
                    help="C7O heldout_site_predictions.csv")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=734201)
    a = ap.parse_args()
    src = a.predictions.expanduser().resolve()
    out = a.output.expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(src)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory must be absent or empty: {out}")
    if a.draws < 500:
        raise ValueError("Use at least 500 posterior draws")

    raw = read_csv(src)
    required = set(KEYS) | {"model", "target_frequency", "target_alt_clone_count",
                            "n_target_clones", "predicted_frequency"}
    if not raw:
        raise ValueError("Prediction table is empty")
    missing = required - set(raw[0])
    if missing:
        raise ValueError(f"Prediction table missing columns: {sorted(missing)}")

    grouped = defaultdict(dict)
    meta = {}
    for line, r in enumerate(raw, 2):
        key = tuple(r[k] for k in KEYS)
        model = r["model"]
        if model not in MODELS:
            raise ValueError(f"Unexpected model {model!r} at line {line}")
        n = int(r["n_target_clones"])
        k = int(r["target_alt_clone_count"])
        q = float(r["target_frequency"])
        p = float(r["predicted_frequency"])
        if n < 1 or not 0 <= k <= n or not 0 <= p <= 1:
            raise ValueError(f"Invalid count or prediction at line {line}")
        if not math.isclose(q, k / n, abs_tol=1e-8):
            raise ValueError(f"target_frequency != k/n at line {line}")
        if model in grouped[key]:
            raise ValueError(f"Duplicate model row for {key}, {model}")
        grouped[key][model] = p
        current = (n, k, q)
        if key in meta and meta[key] != current:
            raise ValueError(f"Target count/frequency mismatch across model rows for {key}")
        meta[key] = current

    keys = sorted(grouped)
    for key in keys:
        if set(grouped[key]) != set(MODELS):
            raise ValueError(f"Missing model prediction(s) for {key}: {set(MODELS)-set(grouped[key])}")
    pops = sorted({k[0] for k in keys})
    folds = sorted({(k[0], k[1], k[2]) for k in keys})
    idx_by_fold = {f: np.asarray([i for i, k in enumerate(keys)
                                  if (k[0], k[1], k[2]) == f], dtype=int) for f in folds}
    folds_by_pop = {p: [f for f in folds if f[0] == p] for p in pops}
    n = np.asarray([meta[k][0] for k in keys], dtype=int)
    successes = np.asarray([meta[k][1] for k in keys], dtype=int)
    q_obs = successes / n
    predictions = {m: np.asarray([grouped[k][m] for k in keys], dtype=float) for m in MODELS}

    # Jeffreys posterior for latent frequency theta_i | k_i,n_i.
    # Note: independent site-wise draws ignore linkage/correlation among loci.
    rng = np.random.default_rng(a.seed)
    draws_by_model = {target: {m: {metric: np.empty((a.draws, len(pops)), dtype=float)
                          for metric in METRICS} for m in MODELS} for target in TARGETS}
    delta_pairs = [("evo2_plus_history", "history_only"),
                   ("evo2_plus_history", "evo2_only"),
                   ("evo2_only", "persistence"),
                   ("history_only", "persistence"),
                   ("evo2_plus_history", "persistence")]
    delta_draws = {(target, l, r, metric): np.empty((a.draws, len(pops)), dtype=float)
                   for target in TARGETS for l, r in delta_pairs for metric in METRICS}

    for b in range(a.draws):
        theta = rng.beta(successes + 0.5, n - successes + 0.5)
        replicate_q = rng.binomial(n, theta) / n
        pop_scores = {target: {m: {metric: [] for metric in METRICS}
                               for m in MODELS} for target in TARGETS}
        for pi, pop in enumerate(pops):
            fold_scores = {target: {m: {metric: [] for metric in METRICS}
                                    for m in MODELS} for target in TARGETS}
            for fold in folds_by_pop[pop]:
                ix = idx_by_fold[fold]
                ys = {"latent_frequency": theta[ix],
                      "replicate_sample_frequency": replicate_q[ix]}
                for target, y in ys.items():
                    for model in MODELS:
                        p_raw = predictions[model][ix]
                        p = np.clip(p_raw, 1e-6, 1 - 1e-6)
                        per_metric = {
                            "frequency_mae": np.abs(p_raw - y),
                            "target_clone_brier": y * (1-p)**2 + (1-y) * p**2,
                            "target_clone_log_loss": -y*np.log(p) - (1-y)*np.log(1-p),
                        }
                        for metric, vals in per_metric.items():
                            fold_scores[target][model][metric].append(float(np.mean(vals)))
            for target in TARGETS:
                for model in MODELS:
                    for metric in METRICS:
                        pop_scores[target][model][metric].append(
                            float(np.mean(fold_scores[target][model][metric])))
                        draws_by_model[target][model][metric][b, pi] = pop_scores[target][model][metric][-1]
                for left, right in delta_pairs:
                    for metric in METRICS:
                        delta_draws[(target, left, right, metric)][b, pi] = (
                            pop_scores[target][left][metric][-1] - pop_scores[target][right][metric][-1])

    summary = []
    for target in TARGETS:
        for model in MODELS:
            for metric in METRICS:
                overall_draws = draws_by_model[target][model][metric].mean(axis=1)
                lo, med, hi = quantile_ci(overall_draws)
                summary.append({"scoring_target": target, "model": model, "metric": metric,
                    "posterior_predictive_mean_macro_score": float(overall_draws.mean()),
                    "posterior_draw_q025": lo, "posterior_draw_median": med,
                    "posterior_draw_q975": hi, "n_population_clusters": len(pops)})

    deltas_out = []
    for (target, left, right, metric), vals in delta_draws.items():
        overall_draws = vals.mean(axis=1)
        lo, med, hi = quantile_ci(overall_draws)
        deltas_out.append({"scoring_target": target, "comparison": f"{left}_minus_{right}", "metric": metric,
            "posterior_mean_macro_delta": float(overall_draws.mean()),
            "posterior_draw_q025": lo, "posterior_draw_median": med,
            "posterior_draw_q975": hi, "negative_favors_left": True,
            "n_population_clusters": len(pops)})

    site_out = []
    for i, key in enumerate(keys):
        n_i, k_i, q_i = meta[key]
        site_out.append({**dict(zip(KEYS, key)), "k_target": k_i, "n_target": n_i,
            "observed_frequency": q_i,
            "posterior_mean_frequency": (k_i+0.5)/(n_i+1),
            **{f"prediction_{m}": float(predictions[m][i]) for m in MODELS}})

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out/"posterior_frequency_by_site.csv", site_out, list(site_out[0]))
    write_csv(out/"posterior_macro_scores.csv", summary, list(summary[0]))
    write_csv(out/"posterior_macro_model_deltas.csv", deltas_out, list(deltas_out[0]))
    count_summary = {str(v): int(np.sum(n == v)) for v in sorted(set(n.tolist()))}
    report = {"status": "EXPLORATORY_BETA_BINOMIAL_TARGET_FREQUENCY_UNCERTAINTY",
        "input": str(src), "n_site_fold_allele_rows": len(keys),
        "n_population_clusters": len(pops), "n_population_time_folds": len(folds),
        "target_clone_count_distribution_by_site_rows": count_summary,
        "posterior": "Independent site-wise Jeffreys posterior: theta_i | k_i,n_i ~ Beta(k_i+0.5,n_i-k_i+0.5).",
        "scoring_targets": {"latent_frequency": "Posterior draws of the latent population allele frequency.",
            "replicate_sample_frequency": "A new target clone panel simulated from each posterior-drawn latent frequency, using that site's observed target clone count."},
        "aggregation": "Sites averaged within fold; folds averaged within population; populations equally averaged. Posterior intervals describe only target-frequency sampling uncertainty under the stated model, not a population-bootstrap confidence interval.",
        "independence_assumption": "Conditional binomial sampling of clones and independent posterior draws across SNP alleles. LTEE clones/loci may be correlated through shared ancestry/linkage, so intervals may understate uncertainty.",
        "n_posterior_draws": a.draws, "seed": a.seed,
        "metrics": {"frequency_mae": "|predicted frequency - latent theta|",
            "target_clone_brier": "theta*(1-p)^2 + (1-theta)*p^2",
            "target_clone_log_loss": "-theta*log(clip(p,1e-6,1-1e-6))-(1-theta)*log(1-clip(p,1e-6,1-1e-6))"},
        "outputs": ["posterior_frequency_by_site.csv", "posterior_macro_scores.csv", "posterior_macro_model_deltas.csv"],
        "interpretation_warning": "This is a plug-in Bayesian sensitivity analysis using held-out observed counts to estimate latent frequencies. It is not a new blind test and does not remove uncertainty from only 12 populations or model selection."}
    (out/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
