import pandas as pd
import pytest

from scripts.analyze_results import analyze, bootstrap_mean_ci


def test_bootstrap_mean_ci_contains_constant_value():
    mean, low, high = bootstrap_mean_ci([4, 4, 4], samples=500)
    assert mean == 4
    assert low == 4
    assert high == 4


def test_analysis_outputs_paired_comparison(tmp_path):
    rows = []
    for seed, static_calls, adaptive_calls, static_error, adaptive_error in [
        (0, 100, 105, 0.01, 0.02),
        (1, 110, 100, 0.03, 0.01),
        (2, 90, 95, 0.00, 0.01),
    ]:
        for method, calls, error in [
            ("static", static_calls, static_error),
            ("selectivity_adaptive", adaptive_calls, adaptive_error),
        ]:
            rows.append({
                "seed": seed,
                "budget": 0.02,
                "method": method,
                "test_calls": calls,
                "test_wrong_hit_rate": error,
                "test_semantic_hits": 1,
                "test_exact_reuses": 10,
                "test_wrong_hits": 1,
                "test_decisions": 100,
                "tune_calls": calls + 1,
                "tune_wrong_hit_rate": error,
            })
    input_csv = tmp_path / "runs.csv"
    pd.DataFrame(rows).to_csv(input_csv, index=False)

    summary, paired = analyze(input_csv, tmp_path / "out", bootstrap_samples=500)

    assert (tmp_path / "out" / "summary_checked.csv").exists()
    assert (tmp_path / "out" / "paired_vs_static.csv").exists()
    assert (tmp_path / "out" / "analysis_metadata.json").exists()
    row = paired[
        (paired["budget"] == 0.02)
        & (paired["method"] == "selectivity_adaptive")
    ].iloc[0]
    assert row["paired_seeds"] == 3
    assert row["call_difference_mean"] == pytest.approx(0.0)
    assert row["candidate_budget_violation_rate"] == pytest.approx(0.0)
