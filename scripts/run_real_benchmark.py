import argparse
import os
import time
from pathlib import Path

import nltk
import pandas as pd
import requests
from nltk.corpus import movie_reviews

API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"


def classify(session, api_key, text):
    prompt = (
        "Classify the movie review as positive or negative. "
        "Reply with exactly one word: Positive or Negative.\n\n"
        f"Review:\n{text[:2500]}"
    )

    started = time.perf_counter()
    response = session.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 100,
            "reasoning_effort": "low",
        },
        timeout=60,
    )
    latency = time.perf_counter() - started

    if response.status_code == 429:
        raise RuntimeError("RATE_LIMIT: " + response.text[:500])

    response.raise_for_status()
    body = response.json()
    message = body["choices"][0]["message"]
    content = (message.get("content") or "").strip().lower()

    if content not in {"positive", "negative"}:
        raise ValueError(f"Invalid classification output: {content[:200]!r}")

    usage = body.get("usage") or {}
    return {
        "prediction": int(content == "negative"),
        "raw_output": content,
        "latency_seconds": latency,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "model": body.get("model", MODEL),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", default="results/real_groq/baseline_compact.csv")
    args = parser.parse_args()

    if not 1 <= args.limit <= 500:
        parser.error("--limit must be between 1 and 500")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Set GROQ_API_KEY in this PowerShell session.")

    try:
        movie_reviews.fileids()
    except LookupError:
        nltk.download("movie_reviews", quiet=True)

    records = []
    session = requests.Session()
    fileids = sorted(movie_reviews.fileids())
    negative = [f for f in fileids if movie_reviews.categories(f)[0] == "neg"]
    positive = [f for f in fileids if movie_reviews.categories(f)[0] == "pos"]

    import random

    rng = random.Random(42)
    rng.shuffle(negative)
    rng.shuffle(positive)

    selected = []
    n_negative = (args.limit + 1) // 2
    n_positive = args.limit // 2
    selected.extend(negative[:n_negative])
    selected.extend(positive[:n_positive])
    rng.shuffle(selected)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for index, fileid in enumerate(selected, start=1):
        text = " ".join(movie_reviews.words(fileid))
        label = int(movie_reviews.categories(fileid)[0] == "neg")

        while True:
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
                print(
                    f"{index}/{len(selected)}: {fileid} "
                    f"prediction={result['prediction']} "
                    f"correct={record['correct']} "
                    f"latency={result['latency_seconds']:.2f}s",
                    flush=True,
                )
                pd.DataFrame(records).to_csv(output, index=False)
                break

            except Exception as exc:
                if "RATE_LIMIT:" in str(exc):
                    print(f"Rate limit reached. Waiting 60 seconds before retrying {fileid}...", flush=True)
                    time.sleep(60)
                    continue

                record = {
                    "row_id": fileid,
                    "ground_truth": label,
                    "prediction": None,
                    "raw_output": None,
                    "latency_seconds": None,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "model": MODEL,
                    "correct": None,
                    "error": str(exc),
                }
                records.append(record)
                pd.DataFrame(records).to_csv(output, index=False)
                print(f"{index}/{len(selected)}: ERROR {exc}", flush=True)
                break

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False)

    successful = [r for r in records if r["correct"] is not None]
    print("\nSaved:", output)
    print("Successful predictions:", len(successful), "/", len(records))
    print("API calls attempted:", len(records))

    if successful:
        accuracy = sum(r["correct"] for r in successful) / len(successful)
        print("Accuracy on successful predictions:", round(accuracy, 4))
        print("Total prompt tokens:", sum(r["prompt_tokens"] or 0 for r in successful))
        print("Total completion tokens:", sum(r["completion_tokens"] or 0 for r in successful))
        print(
            "Mean latency (seconds):",
            round(sum(r["latency_seconds"] for r in successful) / len(successful), 3),
        )


if __name__ == "__main__":
    main()

