from __future__ import annotations
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .cache import SemanticCache
from .controllers import static_threshold, selectivity_threshold, density_threshold, inverted_threshold
from .datasets import load_dataset
from .embeddings import make_embeddings


def make_stream(indices, word_counts, embeddings, seed, queries, selectivities, density_ref):
    rng = np.random.default_rng(seed)
    ordered = indices[np.argsort(word_counts[indices], kind="stable")]
    stream = []
    for _ in range(queries):
        sel = float(rng.choice(selectivities))
        width = max(2, min(len(ordered), int(round(sel * len(ordered)))))
        start = int(rng.integers(0, len(ordered) - width + 1))
        rows = ordered[start:start + width].copy()
        rng.shuffle(rows)
        if len(rows) > 1:
            sims = embeddings[rows] @ embeddings[rows].T
            np.fill_diagonal(sims, -1.0)
            density = float((sims.max(axis=1) >= density_ref).mean())
        else:
            density = 0.0
        stream.append((rows, width / len(ordered), density))
    return stream


def run_config(stream, embeddings, labels, method, params):
    if method == "static":
        fn = lambda s, d: static_threshold(s, d, params["threshold"])
    elif method == "selectivity_adaptive":
        fn = lambda s, d: selectivity_threshold(s, d, params["low"], params["high"])
    elif method == "density_adaptive":
        fn = lambda s, d: density_threshold(s, d, params["low"], params["high"])
    elif method == "inverted_control":
        fn = lambda s, d: inverted_threshold(s, d, params["low"], params["high"])
    elif method == "exact_cache":
        fn = lambda s, d: 2.0
    elif method == "no_cache":
        return {"calls": sum(len(rows) for rows, _, _ in stream), "semantic_hits": 0, "exact_reuses": 0, "wrong_hits": 0, "decisions": sum(len(rows) for rows, _, _ in stream), "wrong_hit_rate": 0.0}
    cache = SemanticCache(embeddings, labels, fn)
    for rows, sel, density in stream:
        cache.process_query(rows, sel, density)
    return cache.metrics()


def configurations():
    configs = [("no_cache", {}) , ("exact_cache", {})]
    for t in np.round(np.arange(0.50, 1.00, 0.01), 2):
        configs.append(("static", {"threshold": float(t)}))
    for low, high in itertools.product([0.50, 0.60, 0.70, 0.80, 0.90], [0.90, 0.95, 0.98, 1.01]):
        if low >= high:
            continue
        for method in ("selectivity_adaptive", "density_adaptive", "inverted_control"):
            configs.append((method, {"low": low, "high": high}))
    return configs


def non_dominated(frame):
    points = frame[["calls", "wrong_hit_rate"]].drop_duplicates().to_numpy()
    keep = []
    for i, (calls, err) in enumerate(points):
        dominated = any((c <= calls and e <= err and (c < calls or e < err)) for j, (c, e) in enumerate(points) if i != j)
        if not dominated:
            keep.append((calls, err))
    return frame[frame.apply(lambda r: (r["calls"], r["wrong_hit_rate"]) in keep, axis=1)]


def run_benchmark(dataset="nltk_movie_reviews", seeds=10, queries=60, output="results/run", budgets=None):
    budgets = budgets or [0.02, 0.05, 0.10, 0.15, 0.20]
    outdir = Path(output)
    outdir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(dataset)
    embeddings = make_embeddings(df.text.tolist())
    labels = df.label.to_numpy(dtype=int)
    word_counts = df.n_words.to_numpy()
    rows_out = []
    selectivities = [0.02, 0.05, 0.10, 0.20, 0.40, 0.70, 1.0]
    for seed in range(seeds):
        rng = np.random.default_rng(1000 + seed)
        perm = rng.permutation(len(df))
        tune_ids, test_ids = perm[:len(df)//2], perm[len(df)//2:]
        tune_stream = make_stream(tune_ids, word_counts, embeddings, seed * 10 + 1, queries, selectivities, 0.8)
        test_stream = make_stream(test_ids, word_counts, embeddings, seed * 10 + 2, queries, selectivities, 0.8)
        scored = []
        for method, params in configurations():
            metrics = run_config(tune_stream, embeddings, labels, method, params)
            scored.append((method, params, metrics))
        for budget in budgets:
            eligible = [(m, p, x) for m, p, x in scored if x["wrong_hit_rate"] <= budget]
            # Exact/no-cache are valid fallbacks. Tie-break by fewer calls, then lower error.
            chosen_by_method = {}
            for method in ("no_cache", "exact_cache", "static", "selectivity_adaptive", "density_adaptive", "inverted_control"):
                candidates = [(m, p, x) for m, p, x in eligible if m == method]
                if candidates:
                    chosen_by_method[method] = min(candidates, key=lambda item: (item[2]["calls"], item[2]["wrong_hit_rate"]))
            for method, params, tune_metrics in chosen_by_method.values():
                test_metrics = run_config(test_stream, embeddings, labels, method, params)
                rows_out.append({"seed": seed, "budget": budget, "method": method, "params": json.dumps(params, sort_keys=True), **{f"tune_{k}": v for k, v in tune_metrics.items()}, **{f"test_{k}": v for k, v in test_metrics.items()}})
        print(f"seed {seed + 1}/{seeds} complete", flush=True)
    result = pd.DataFrame(rows_out)
    result.to_csv(outdir / "runs.csv", index=False)
    summary = result.groupby(["budget", "method"]).agg(
        calls_mean=("test_calls", "mean"), calls_std=("test_calls", "std"),
        wrong_hit_rate_mean=("test_wrong_hit_rate", "mean"), wrong_hit_rate_std=("test_wrong_hit_rate", "std"),
        semantic_hits_mean=("test_semantic_hits", "mean"), exact_reuses_mean=("test_exact_reuses", "mean")
    ).reset_index()
    summary.to_csv(outdir / "summary.csv", index=False)
    pareto = non_dominated(result.rename(columns={"test_calls": "calls", "test_wrong_hit_rate": "wrong_hit_rate"}))
    pareto.to_csv(outdir / "pareto.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    for method, group in result.groupby("method"):
        ax.scatter(group["test_wrong_hit_rate"] * 100, group["test_calls"], label=method, alpha=0.75)
    ax.set_xlabel("Wrong cache hits (% of cache decisions)")
    ax.set_ylabel("Simulated model calls on test stream")
    ax.set_title(f"Semantic-cache trade-off: {dataset}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "pareto.png", dpi=160)
    plt.close(fig)
    metadata = {"dataset": dataset, "rows": len(df), "seeds": seeds, "queries_per_stream": queries, "budgets": budgets, "embedding": "TF-IDF + truncated SVD", "label_source": "dataset oracle labels", "real_llm_calls": 0, "warning": "Offline simulation only; call counts are not measured API calls."}
    (outdir / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return result, summary
