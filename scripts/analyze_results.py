from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd


PAIR_METHODS = [
    "selectivity_adaptive",
    "density_adaptive",
    "inverted_control",
    "exact_cache",
]


def bootstrap_mean_ci(values, confidence=0.95, samples=10000, seed=12345):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for i in range(samples):
        means[i] = rng.choice(values, size=values.size, replace=True).mean()
    alpha = 1.0 - confidence
    return (
        float(values.mean()),
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def analyze(input_csv: Path, output_dir: Path, bootstrap_samples=10000):
    runs = pd.read_csv(input_csv)
    required = {
        "seed", "budget", "method", "test_calls", "test_wrong_hit_rate",
        "test_semantic_hits", "test_exact_reuses", "test_wrong_hits",
        "test_decisions", "tune_calls", "tune_wrong_hit_rate",
    }
    missing = sorted(required - set(runs.columns))
    if missing:
        raise ValueError(f"runs.csv is missing required columns: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Each seed and budget should have one selected configuration per method.
    duplicates = runs.duplicated(["seed", "budget", "method"], keep=False)
    if duplicates.any():
        sample = runs.loc[duplicates, ["seed", "budget", "method"]].head(10)
        raise ValueError(
            "Duplicate seed/budget/method rows found. Example:\n"
            + sample.to_string(index=False)
        )

    # Summaries include the number of test runs that exceeded their nominal budget.
    summary = runs.groupby(["budget", "method"], as_index=False).agg(
        seeds=("seed", "nunique"),
        calls_mean=("test_calls", "mean"),
        calls_sd=("test_calls", "std"),
        error_mean=("test_wrong_hit_rate", "mean"),
        error_sd=("test_wrong_hit_rate", "std"),
        wrong_hits_mean=("test_wrong_hits", "mean"),
        decisions_mean=("test_decisions", "mean"),
        semantic_hits_mean=("test_semantic_hits", "mean"),
        exact_reuses_mean=("test_exact_reuses", "mean"),
    )
    # Compute budget violation rate explicitly because it depends on the group key.
    violation = (
        runs.assign(violates=runs["test_wrong_hit_rate"] > runs["budget"])
        .groupby(["budget", "method"], as_index=False)["violates"]
        .mean()
        .rename(columns={"violates": "test_budget_violation_rate"})
    )
    summary = summary.merge(
        violation, on=["budget", "method"], how="left"
    )
    summary.to_csv(output_dir / "summary_checked.csv", index=False)

    # Paired comparisons use only seeds for which both methods have results at
    # the same nominal budget. Negative differences mean adaptive uses fewer calls.
    paired_rows = []
    static = runs[runs["method"] == "static"][
        ["seed", "budget", "test_calls", "test_wrong_hit_rate"]
    ].rename(columns={
        "test_calls": "static_calls",
        "test_wrong_hit_rate": "static_error",
    })
    for method in PAIR_METHODS:
        candidate = runs[runs["method"] == method][
            ["seed", "budget", "test_calls", "test_wrong_hit_rate"]
        ].rename(columns={
            "test_calls": "candidate_calls",
            "test_wrong_hit_rate": "candidate_error",
        })
        paired = static.merge(candidate, on=["seed", "budget"], how="inner")
        if paired.empty:
            continue
        paired["method"] = method
        paired["call_difference_candidate_minus_static"] = (
            paired["candidate_calls"] - paired["static_calls"]
        )
        paired["error_difference_candidate_minus_static"] = (
            paired["candidate_error"] - paired["static_error"]
        )
        for budget, group in paired.groupby("budget"):
            call_mean, call_low, call_high = bootstrap_mean_ci(
                group["call_difference_candidate_minus_static"],
                samples=bootstrap_samples,
                seed=12345 + int(round(float(budget) * 10000)),
            )
            err_mean, err_low, err_high = bootstrap_mean_ci(
                group["error_difference_candidate_minus_static"],
                samples=bootstrap_samples,
                seed=54321 + int(round(float(budget) * 10000)),
            )
            paired_rows.append({
                "budget": budget,
                "method": method,
                "paired_seeds": int(group["seed"].nunique()),
                "candidate_calls_mean": float(group["candidate_calls"].mean()),
                "static_calls_mean": float(group["static_calls"].mean()),
                "call_difference_mean": call_mean,
                "call_difference_ci95_low": call_low,
                "call_difference_ci95_high": call_high,
                "candidate_error_mean": float(group["candidate_error"].mean()),
                "static_error_mean": float(group["static_error"].mean()),
                "error_difference_mean": err_mean,
                "error_difference_ci95_low": err_low,
                "error_difference_ci95_high": err_high,
                "candidate_budget_violation_rate": float(
                    (group["candidate_error"] > float(budget)).mean()
                ),
                "static_budget_violation_rate": float(
                    (group["static_error"] > float(budget)).mean()
                ),
                "candidate_fewer_calls_seeds": int(
                    (group["call_difference_candidate_minus_static"] < 0).sum()
                ),
                "candidate_equal_calls_seeds": int(
                    (group["call_difference_candidate_minus_static"] == 0).sum()
                ),
                "candidate_more_calls_seeds": int(
                    (group["call_difference_candidate_minus_static"] > 0).sum()
                ),
            })
    paired_summary = pd.DataFrame(paired_rows)
    paired_summary.to_csv(output_dir / "paired_vs_static.csv", index=False)

    metadata = {
        "input": str(input_csv),
        "rows_in_runs_file": int(len(runs)),
        "unique_seeds": int(runs["seed"].nunique()),
        "budgets": sorted(float(x) for x in runs["budget"].unique()),
        "bootstrap_samples": int(bootstrap_samples),
        "confidence_interval": "percentile bootstrap 95% CI over paired seed-level differences",
        "interpretation": {
            "call_difference": "candidate calls minus static calls; negative favours candidate",
            "error_difference": "candidate wrong-hit rate minus static wrong-hit rate; negative means lower error",
            "budget_violation_rate": "fraction of test seeds whose observed wrong-hit rate exceeded the nominal budget",
        },
        "warning": "These are offline simulation results using dataset labels as oracle labels, not real LLM calls.",
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    return summary, paired_summary


def main():
    parser = argparse.ArgumentParser(
        description="Analyze paired semantic-cache benchmark results."
    )
    parser.add_argument("--input", required=True, help="Path to runs.csv")
    parser.add_argument("--output", required=True, help="Directory for analysis outputs")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")
    summary, paired = analyze(
        Path(args.input), Path(args.output), args.bootstrap_samples
    )
    print(f"Wrote {args.output}")
    print("\nPaired comparison against static:")
    if paired.empty:
        print("No paired rows found.")
    else:
        cols = [
            "budget", "method", "call_difference_mean",
            "call_difference_ci95_low", "call_difference_ci95_high",
            "error_difference_mean", "candidate_budget_violation_rate",
        ]
        print(paired[cols].to_string(index=False))


if __name__ == "__main__":
    main()
