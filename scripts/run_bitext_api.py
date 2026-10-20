import os
import time
import random
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"
SOURCE = Path("data/bitext_sample_500.csv")
OUTPUT = Path("results/bitext_groq/baseline_200.csv")
TARGET = 200

def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY is not set in this PowerShell session.")
    if not SOURCE.exists():
        raise SystemExit(f"Dataset not found: {SOURCE}")

    df = pd.read_csv(SOURCE).dropna(subset=["instruction", "intent"]).copy()
    df["row_id"] = df.index.astype(str)
    df["instruction"] = df["instruction"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()
    df = df[df["instruction"] != ""].drop_duplicates(subset=["instruction"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df.head(TARGET)

    intents = sorted(df["intent"].unique())
    allowed = "\n".join(intents)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT.exists():
        saved = pd.read_csv(OUTPUT)
        completed = set(saved.loc[saved["correct"].notna(), "row_id"].astype(str))
        records = saved.to_dict("records")
    else:
        completed = set()
        records = []

    session = requests.Session()
    print(f"Dataset rows selected: {len(df)}")
    print(f"Intent labels: {len(intents)}")
    print(f"Already completed: {len(completed)}")
    print(f"Target successful responses: {TARGET}")
    print(f"Output: {OUTPUT}", flush=True)

    for _, row in df.iterrows():
        row_id = str(row["row_id"])
        if row_id in completed:
            continue
        if len(completed) >= TARGET:
            break

        prompt = (
            "Classify the customer-support request into exactly one intent label. "
            "Choose only from the allowed labels below. Return only the label, "
            "with no punctuation or explanation.\n\n"
            f"Allowed labels:\n{allowed}\n\n"
            f"Customer request:\n{row['instruction'][:2500]}"
        )

        while True:
            try:
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
                    timeout=90,
                )
                latency = time.perf_counter() - started

                if response.status_code == 429:
                    print("Rate limit. Waiting 60 seconds; completed calls are saved.", flush=True)
                    time.sleep(60)
                    continue

                response.raise_for_status()
                body = response.json()
                raw = (body["choices"][0]["message"].get("content") or "").strip()
                normalized = raw.strip().strip("`'\"").strip().lower()
                matches = [label for label in intents if label.lower() == normalized]

                if len(matches) != 1:
                    raise ValueError(f"Model returned a label outside the allowed set: {raw[:160]!r}")

                prediction = matches[0]
                usage = body.get("usage") or {}
                record = {
                    "row_id": row_id,
                    "instruction": row["instruction"],
                    "category": row["category"],
                    "ground_truth": row["intent"],
                    "prediction": prediction,
                    "correct": int(prediction == row["intent"]),
                    "raw_output": raw,
                    "latency_seconds": latency,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "model": body.get("model", MODEL),
                    "error": None,
                }
                records.append(record)
                completed.add(row_id)
                pd.DataFrame(records).to_csv(OUTPUT, index=False)
                print(
                    f"{len(completed)}/{TARGET} correct={record['correct']} "
                    f"latency={latency:.2f}s intent={row['intent']}",
                    flush=True,
                )
                break

            except KeyboardInterrupt:
                pd.DataFrame(records).to_csv(OUTPUT, index=False)
                print("\nInterrupted. Successful responses are checkpointed. Rerun to resume.")
                return
            except Exception as exc:
                print(f"Request/parse error: {exc}. Retrying in 10 seconds.", flush=True)
                time.sleep(10)

    result = pd.read_csv(OUTPUT)
    successful = result[result["correct"].notna()]
    print("\nFINAL RESULTS")
    print("Successful API responses:", len(successful), "/", TARGET)
    if len(successful):
        print("Accuracy:", round(successful["correct"].mean(), 4))
        print("Total prompt tokens:", int(successful["prompt_tokens"].fillna(0).sum()))
        print("Total completion tokens:", int(successful["completion_tokens"].fillna(0).sum()))
        print("Mean latency:", round(successful["latency_seconds"].mean(), 3), "seconds")
    print("Saved:", OUTPUT)

if __name__ == "__main__":
    main()
