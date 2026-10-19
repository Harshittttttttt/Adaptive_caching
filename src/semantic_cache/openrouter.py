"""Minimal OpenRouter chat client for a later real-predicate experiment.

This client is not used by the offline benchmark. Record model, date, token usage,
latency, and provider-reported cost in any experiment that uses it.
"""
import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()


def classify_text(text: str, predicate: str, model: str | None = None):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY in .env before using the real API client.")
    model = model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    prompt = (
        "Classify the text for the predicate below. Return only JSON with one key, label, "
        "whose value is true or false. Do not include any other text.\n"
        f"Predicate: {predicate}\nText: {text}"
    )
    started = time.perf_counter()
    response = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0},
        timeout=60,
    )
    response.raise_for_status()
    body = response.json()
    content = body["choices"][0]["message"]["content"]
    elapsed = time.perf_counter() - started
    usage = body.get("usage", {})
    return {"content": content, "model": model, "latency_seconds": elapsed, "usage": usage, "raw": body}
