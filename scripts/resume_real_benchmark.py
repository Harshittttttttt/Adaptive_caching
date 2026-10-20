import argparse
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from nltk.corpus import movie_reviews

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_real_benchmark import classify

DEFAULT_INPUT = Path("results/real_groq/baseline_500.csv")
TARGET = 500
MAX_RETRIES = 12


def resume_benchmark(input_path: Path = DEFAULT_INPUT, target: int = TARGET):
    if not input_path.exists():
        raise SystemExit(f"Missing saved results: {input_path}")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY is not set.")

    output_path = input_path
    df = pd.read_csv(input_path)
    done = set(df["row_id"].dropna().astype(str))
    records = df.to_dict("records")

    all_ids = list(movie_reviews.fileids())
    remaining = [fid for fid in all_ids if fid not in done]
    random.Random(42).shuffle(remaining)

    successful = sum(pd.notna(r.get("correct")) for r in records)
    print(f"Saved rows: {len(records)}; successful: {successful}; remaining candidates: {len(remaining)}")

    session = requests.Session()

    for fileid in remaining:
        if successful >= target:
            break

        text = " ".join(movie_reviews.words(fileid))
        label = int(movie_reviews.categories(fileid)[0] == "neg")

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = classify(session, api_key, text)
                record = {
                    "row_id": fileid,
                    "ground_truth": label,
                    **result,
                    "correct": int(result["prediction"] == label),
                    "error": None,
                }
                records.append(record)
                successful += 1
                pd.DataFrame(records).to_csv(output_path, index=False)
                print(f"{successful}/{target}: {fileid} prediction={result['prediction']} correct={record['correct']}", flush=True)
                break
            except KeyboardInterrupt:
                pd.DataFrame(records).to_csv(output_path, index=False)
                raise
            except Exception as exc:
                message = str(exc)
                retryable = any(x in message.lower() for x in [
                    "rate_limit", "rate limit", "429", "timeout",
                    "connection", "temporarily", "remote end closed",
                    "read timed out"
                ])
                if retryable and attempt < MAX_RETRIES:
                    wait = 60 if ("rate_limit" in message.lower() or "429" in message) else min(5 * attempt, 30)
                    print(f"Temporary error for {fileid}: {message[:180]}. Waiting {wait}s ({attempt}/{MAX_RETRIES})", flush=True)
                    time.sleep(wait)
                    continue

                print(f"Stopping on error for {fileid}: {message[:300]}", flush=True)
                pd.DataFrame(records).to_csv(output_path, index=False)
                raise SystemExit(1)

    pd.DataFrame(records).to_csv(output_path, index=False)
    final_df = pd.read_csv(output_path)
    valid = final_df[final_df["correct"].notna()]
    print(f"\nSaved: {output_path}")
    print(f"Rows saved: {len(final_df)}")
    print(f"Successful predictions: {len(valid)}")
    print(f"Accuracy: {valid['correct'].astype(float).mean():.4f}" if len(valid) else "Accuracy: N/A")


def main():
    parser = argparse.ArgumentParser(description="Resume live Groq movie review benchmark.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to checkpointed CSV.")
    parser.add_argument("--target", type=int, default=TARGET, help="Target total successful predictions.")
    args = parser.parse_args()
    resume_benchmark(input_path=args.input, target=args.target)


if __name__ == "__main__":
    main()
