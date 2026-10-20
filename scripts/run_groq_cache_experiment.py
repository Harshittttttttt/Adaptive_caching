import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def run_replay(
    input_path: Path,
    output_dir: Path,
    static_threshold: float = 0.85,
    density_low: float = 0.65,
    density_high: float = 0.95,
    density_reference: float = 0.80,
):
    if not input_path.exists():
        raise FileNotFoundError(f"Missing baseline file: {input_path}")

    df = pd.read_csv(input_path)
    if len(df) == 0:
        raise ValueError(f"Baseline file {input_path} is empty.")
    if df["prediction"].isna().any():
        raise ValueError("Baseline file contains missing predictions.")

    from nltk.corpus import movie_reviews

    texts = []
    for row_id in df["row_id"]:
        texts.append(" ".join(movie_reviews.words(row_id))[:2500])

    vectors = TfidfVectorizer(
        stop_words="english",
        min_df=1,
        max_features=5000,
        sublinear_tf=True,
    ).fit_transform(texts)
    vectors = normalize(vectors).toarray()

    labels = df["ground_truth"].astype(int).to_numpy()
    predictions = df["prediction"].astype(int).to_numpy()
    ids = df["row_id"].tolist()
    texts_key = [text.strip().lower() for text in texts]
    model_name = df["model"].iloc[0] if "model" in df.columns else "unknown"

    methods = ["no_cache", "exact_cache", "static_semantic", "density_adaptive"]
    logs = []
    summaries = []

    for method in methods:
        exact_cache = {}  # query_key -> (predicted, source_id)
        semantic_ids = []
        calls = 0
        semantic_hits = 0
        exact_hits = 0
        wrong_semantic_hits = 0
        correct = 0

        for pass_number in (1, 2):
            for i in range(len(df)):
                query_key = texts_key[i]
                predicted = None
                hit_type = "miss"
                source_id = None
                similarity = None
                threshold = None
                density = None

                if method == "no_cache":
                    hit_type = "miss"
                elif method == "exact_cache":
                    if query_key in exact_cache:
                        predicted, source_id = exact_cache[query_key]
                        hit_type = "exact"
                        exact_hits += 1
                elif method in ("static_semantic", "density_adaptive"):
                    # Exact repeated queries take priority over similarity lookup
                    if query_key in exact_cache:
                        predicted, source_id = exact_cache[query_key]
                        hit_type = "exact"
                        exact_hits += 1
                    elif semantic_ids:
                        sims = vectors[semantic_ids] @ vectors[i]
                        best_pos = int(np.argmax(sims))
                        best_sim = float(sims[best_pos])
                        source_idx = semantic_ids[best_pos]
                        similarity = best_sim

                        if method == "static_semantic":
                            threshold = static_threshold
                        else:
                            density = float((sims >= density_reference).mean())
                            threshold = density_high - (
                                density_high - density_low
                            ) * density

                        if best_sim >= threshold:
                            predicted = int(predictions[source_idx])
                            hit_type = "semantic"
                            source_id = ids[source_idx]
                            semantic_hits += 1
                            wrong_semantic_hits += int(predicted != labels[i])

                if hit_type == "miss":
                    calls += 1
                    predicted = int(predictions[i])
                    source_id = ids[i]

                    if method in ("exact_cache", "static_semantic", "density_adaptive"):
                        exact_cache[query_key] = (predicted, ids[i])
                    if method in ("static_semantic", "density_adaptive"):
                        semantic_ids.append(i)

                is_correct = int(predicted == labels[i])
                correct += is_correct

                logs.append({
                    "method": method,
                    "pass": pass_number,
                    "row_id": ids[i],
                    "ground_truth": int(labels[i]),
                    "prediction": predicted,
                    "correct": is_correct,
                    "decision": hit_type,
                    "source_row_id": source_id,
                    "similarity": similarity,
                    "threshold": threshold,
                    "density": density,
                    "counterfactual_api_call": int(hit_type == "miss"),
                })

        total = 2 * len(df)
        summaries.append({
            "method": method,
            "queries_replayed": total,
            "counterfactual_api_calls": calls,
            "counterfactual_calls_saved": total - calls,
            "exact_hits": exact_hits,
            "semantic_hits": semantic_hits,
            "wrong_semantic_hits": wrong_semantic_hits,
            "accuracy": correct / total,
            "static_threshold": static_threshold if method == "static_semantic" else None,
            "density_low": density_low if method == "density_adaptive" else None,
            "density_high": density_high if method == "density_adaptive" else None,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(logs).to_csv(output_dir / "decisions.csv", index=False)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    metadata = {
        "experiment": "Groq compact cache counterfactual replay",
        "provider": "Groq",
        "model": str(model_name),
        "source_predictions": str(input_path),
        "unique_reviews": len(df),
        "passes": 2,
        "queries_replayed": 2 * len(df),
        "additional_api_calls": 0,
        "actual_api_calls_in_source_baseline": len(df),
        "parameters": {
            "static_threshold": static_threshold,
            "density_low": density_low,
            "density_high": density_high,
            "density_reference": density_reference,
        },
        "replay_only": True,
        "warning": (
            "Counterfactual API calls are replay estimates, not measured live calls. "
            "Every cache miss reuses the frozen baseline prediction. Results on exploratory "
            "replays do not establish general semantic-cache performance."
        ),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(summary_df.to_string(index=False))
    print(f"\nSaved replay outputs to: {output_dir}")
    print("Additional API calls made: 0 (pure counterfactual replay)")
    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description="Replay frozen LLM predictions through exact and semantic cache configurations without live API calls."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/real_groq/baseline_compact.csv"),
        help="Path to input baseline CSV containing frozen model predictions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/real_groq/cache_replay"),
        help="Directory to save replay decisions, summary, and metadata.",
    )
    parser.add_argument(
        "--static-threshold",
        type=float,
        default=0.85,
        help="Cosine similarity threshold for static semantic caching.",
    )
    parser.add_argument(
        "--density-low",
        type=float,
        default=0.65,
        help="Minimum similarity threshold for density-adaptive caching.",
    )
    parser.add_argument(
        "--density-high",
        type=float,
        default=0.95,
        help="Maximum similarity threshold for density-adaptive caching.",
    )
    parser.add_argument(
        "--density-reference",
        type=float,
        default=0.80,
        help="Reference similarity threshold used to compute local neighborhood density.",
    )
    args = parser.parse_args()

    run_replay(
        input_path=args.input,
        output_dir=args.output,
        static_threshold=args.static_threshold,
        density_low=args.density_low,
        density_high=args.density_high,
        density_reference=args.density_reference,
    )


if __name__ == "__main__":
    main()
