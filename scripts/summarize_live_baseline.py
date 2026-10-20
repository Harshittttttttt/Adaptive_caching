"""Offline analysis script to summarize saved live LLM baseline predictions.

Recomputes accuracy, precision, recall, F1 score, confusion matrix, latency,
and token usage statistics from a frozen baseline CSV without calling any API.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def summarize_baseline(input_path: Path, output_dir: Path | None = None) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    total_records = len(df)
    if total_records == 0:
        raise ValueError(f"Input file is empty: {input_path}")

    # Consider records with a valid prediction / ground truth
    valid_mask = df["prediction"].notna() & df["ground_truth"].notna()
    valid_df = df[valid_mask].copy()
    valid_count = len(valid_df)

    if valid_count == 0:
        raise ValueError("No valid predictions found in dataset.")

    y_true = valid_df["ground_truth"].astype(int).to_numpy()
    y_pred = valid_df["prediction"].astype(int).to_numpy()

    correct_count = int((y_true == y_pred).sum())
    incorrect_count = int((y_true != y_pred).sum())
    accuracy = float(accuracy_score(y_true, y_pred))

    # Binary metrics (pos_label=1: negative review label in benchmark)
    prec_1 = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec_1 = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1_1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    # Class 0 metrics (pos_label=0: positive review label in benchmark)
    prec_0 = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    rec_0 = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    f1_0 = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))

    # Macro & Weighted metrics
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    weighted_prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Confusion matrix: rows = actual, columns = predicted
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"actual_{lbl}" for lbl in labels],
        columns=[f"pred_{lbl}" for lbl in labels],
    )

    # Latency statistics
    latency_series = valid_df["latency_seconds"].dropna().astype(float) if "latency_seconds" in valid_df.columns else pd.Series(dtype=float)
    latency_stats = {
        "count": int(len(latency_series)),
        "mean_seconds": float(latency_series.mean()) if len(latency_series) else None,
        "std_seconds": float(latency_series.std()) if len(latency_series) > 1 else None,
        "median_seconds": float(latency_series.median()) if len(latency_series) else None,
        "min_seconds": float(latency_series.min()) if len(latency_series) else None,
        "max_seconds": float(latency_series.max()) if len(latency_series) else None,
    }

    # Token usage statistics
    prompt_tokens_sum = int(valid_df["prompt_tokens"].dropna().sum()) if "prompt_tokens" in valid_df.columns else 0
    completion_tokens_sum = int(valid_df["completion_tokens"].dropna().sum()) if "completion_tokens" in valid_df.columns else 0
    total_tokens_sum = int(valid_df["total_tokens"].dropna().sum()) if "total_tokens" in valid_df.columns else 0

    model_name = valid_df["model"].iloc[0] if "model" in valid_df.columns else "unknown"

    summary = {
        "source_file": str(input_path),
        "model": str(model_name),
        "total_records_in_file": total_records,
        "valid_predictions_analyzed": valid_count,
        "correct_predictions": correct_count,
        "incorrect_predictions": incorrect_count,
        "accuracy": accuracy,
        "accuracy_percent": round(accuracy * 100, 2),
        "classification_metrics": {
            "binary_class_1": {
                "precision": prec_1,
                "recall": rec_1,
                "f1_score": f1_1,
                "support": int((y_true == 1).sum()),
            },
            "binary_class_0": {
                "precision": prec_0,
                "recall": rec_0,
                "f1_score": f1_0,
                "support": int((y_true == 0).sum()),
            },
            "macro_avg": {
                "precision": macro_prec,
                "recall": macro_rec,
                "f1_score": macro_f1,
            },
            "weighted_avg": {
                "precision": weighted_prec,
                "recall": weighted_rec,
                "f1_score": weighted_f1,
            },
        },
        "confusion_matrix": cm.tolist(),
        "latency": latency_stats,
        "tokens": {
            "prompt_tokens_sum": prompt_tokens_sum,
            "completion_tokens_sum": completion_tokens_sum,
            "total_tokens_sum": total_tokens_sum,
            "mean_tokens_per_call": (
                float(total_tokens_sum / valid_count) if valid_count else 0.0
            ),
        },
        "live_api_calls_made_during_analysis": 0,
    }

    print("=" * 60)
    print(f"OFFLINE BASELINE SUMMARY: {input_path}")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Total rows in CSV: {total_records}")
    print(f"Analyzed predictions: {valid_count}")
    print(f"Correct: {correct_count} | Incorrect: {incorrect_count}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f}")
    if latency_stats["mean_seconds"] is not None:
        print(f"Mean Latency: {latency_stats['mean_seconds']:.4f}s (std: {latency_stats['std_seconds']:.4f}s)")
    print(f"Total Tokens: {total_tokens_sum:,} (Prompt: {prompt_tokens_sum:,}, Completion: {completion_tokens_sum:,})")
    print("\nConfusion Matrix (Rows: Actual [0, 1], Columns: Predicted [0, 1]):")
    print(cm_df)
    print("=" * 60)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        cm_df.to_csv(output_dir / "confusion_matrix.csv")

        # Flat summary table
        flat_summary = {
            "source_file": str(input_path),
            "model": str(model_name),
            "total_records": total_records,
            "valid_records": valid_count,
            "correct": correct_count,
            "incorrect": incorrect_count,
            "accuracy": accuracy,
            "class_0_f1": f1_0,
            "class_1_f1": f1_1,
            "macro_f1": macro_f1,
            "mean_latency_seconds": latency_stats["mean_seconds"],
            "total_tokens": total_tokens_sum,
        }
        pd.DataFrame([flat_summary]).to_csv(output_dir / "summary.csv", index=False)
        print(f"Saved analysis artifacts to: {output_dir}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Offline summary analysis of saved live LLM baseline predictions without API calls."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/real_groq/baseline_500.csv"),
        help="Path to saved baseline CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/real_groq/baseline_500_analysis"),
        help="Directory to save summary analysis metrics, tables, and confusion matrix.",
    )
    args = parser.parse_args()

    summarize_baseline(input_path=args.input, output_dir=args.output)


if __name__ == "__main__":
    main()
