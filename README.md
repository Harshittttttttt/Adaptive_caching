# Adaptive Semantic Caching for SQL + LLM Queries

A reproducible research repository to test whether query-selectivity-aware semantic-cache thresholds outperform a tuned fixed threshold.

**Research status:** this repository is an experiment harness, not evidence that adaptive caching works. Results must be generated and reported from each run. Simulation results and real API results must be labelled separately.

## Research questions

1. How much work does exact-repeat caching remove?
2. At a fixed wrong-hit budget, does a selectivity-adaptive threshold reduce model calls versus a static threshold tuned on held-out tuning data?
3. Does a direct local-density signal perform better than selectivity?
4. Do conclusions hold across datasets, predicates, embedding models, and query workloads?

## Methods

- `no_cache`: evaluate every row every time it appears in a query.
- `exact_cache`: reuse only the answer for the same row ID.
- `static`: reuse the nearest cached row if cosine similarity is at least a fixed threshold.
- `selectivity_adaptive`: threshold changes with the fraction of rows left after the SQL filter.
- `density_adaptive`: threshold changes with the measured local similarity density in the current candidate set.
- `inverted_control`: control with the opposite selectivity direction.

The offline benchmark uses dataset labels as oracle answers. It does **not** make LLM calls. This makes it cheap and repeatable, but it is not a real LLM evaluation. The optional API module supports later real-predicate tests and must be used only after you configure a provider and define a labelled evaluation set.

## Quick start (Windows CMD)

Use Python 3.12 or 3.13 for the best compatibility with scientific Python packages. Python 3.14 may not yet have wheels for every dependency.

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -e .
python scripts\run_benchmark.py --dataset synthetic --seeds 10 --output results\synthetic
```

If `py -3.12` is not installed, install Python 3.12 first, or try your installed Python with `python -m venv .venv`.

## Run with movie reviews

The movie-review dataset is downloaded by NLTK on first use. This needs internet access once.

```bat
python scripts\run_benchmark.py --dataset nltk_movie_reviews --seeds 10 --output results\movie_reviews
```

If the NLTK download fails, run the synthetic benchmark first. Do not report synthetic data as real-world data.

## Outputs

Each run writes:

- `runs.csv`: per-seed, per-budget test results.
- `pareto.csv`: configurations with non-dominated error/call counts.
- `summary.json`: run settings and mean/standard deviation.
- `pareto.png`: error versus model-call plot.

The benchmark tunes each method on a tuning split and evaluates the selected configuration on a separate test split. It reports actual wrong cache hits divided by cache decisions. The code records exact-repeat reuse separately.

## Optional OpenRouter setup

The core benchmark does not need an API key. For real model calls, copy `.env.example` to `.env`, set `OPENROUTER_API_KEY`, and use the API client in your own experiment script. Never commit `.env` or keys. Provider pricing and response formats change; record the model ID and date in results. Do not use the offline oracle benchmark's call counts as measured API calls or cost.

## Repository layout

```text
src/semantic_cache/  cache logic, controllers, datasets, benchmark, API client
scripts/             command-line entry points
configs/             documented experiment configuration
 tests/              unit tests
results/             generated outputs (not committed by default)
docs/                experiment protocol and limitations
```

## Reproducibility checklist

Before using results in a paper:

1. Run tests: `python -m pytest`.
2. Keep the seed list, package versions, dataset source, and exact configuration.
3. Use at least 10 seeds for initial results; increase this for final analysis.
4. Tune on the tuning split only. Do not choose parameters after viewing test results.
5. Compare methods at the same error budget and report both error and calls.
6. Add a second dataset and a second predicate before making broad claims.
7. For real LLM claims, measure API calls, token usage, latency, and price directly. Include embedding and cache lookup overhead.
8. Compare against VectorQ/vCache or explain clearly why this was not possible.

## Important interpretation

A negative result is valid only for the tested setup. It does not prove that adaptive semantic caching is universally infeasible. Similarity thresholds can return incorrect labels, and row-count selectivity is only an indirect proxy for local label agreement.
