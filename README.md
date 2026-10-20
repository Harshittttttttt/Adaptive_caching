# Adaptive Semantic Caching for SQL + LLM Queries

A reproducible research repository investigating whether query-selectivity-aware and local-density-adaptive semantic-cache thresholds reduce LLM calls compared to tuned static thresholds under controlled wrong-hit budgets.

---

## 1. Project Goal & Research Questions

When Large Language Models (LLMs) evaluate predicates over rows returned by SQL queries, semantic caching can avoid redundant inference by reusing previous predictions on semantically similar texts. However, setting a fixed similarity threshold presents a difficult trade-off between cache hit rate and erroneous cache hits (wrong hits).

This repository evaluates four primary research questions:
1. **Exact-Repeat Workload Reduction**: How much inference work does exact-text caching remove before semantic similarity is evaluated?
2. **Selectivity-Adaptive Thresholds**: At a fixed wrong-hit error budget, does adapting the similarity threshold based on query selectivity (the fraction of table rows passing SQL filters) reduce model calls compared to a tuned static threshold?
3. **Local-Density-Adaptive Thresholds**: Does adapting the threshold directly from the local embedding density of candidate vectors provide a more reliable error-control signal than row selectivity?
4. **Generalization & Stability**: Do these conclusions hold across datasets, predicates, embedding representations, and query distribution shifts?

---

## 2. Experimental Protocols & Separation of Concerns

To maintain scientific integrity, experimental results in this repository are strictly categorized into three tiers:

| Tier | Description | Cost / API Calls | Ground Truth & Behavior |
|---|---|---|---|
| **1. Offline Oracle Simulation** | Evaluates cache controllers across randomized query streams using dataset labels as ground truth answers. Embeddings are generated offline (TF-IDF + Truncated SVD). | **0 API calls** (free, offline, repeatable). | Uses dataset labels as perfect oracle responses to measure algorithmic cache dynamics. |
| **2. Frozen Live API Baselines** | Measures real LLM latency, token counts, and classification accuracy by querying hosted inference APIs (e.g. Groq). Predictions are frozen into CSV artifacts. | **Frozen artifacts** (re-analyzed offline without API calls). | Real model outputs from `openai/gpt-oss-20b`. |
| **3. Counterfactual Cache Replay** | Replays frozen model predictions through tiered exact and semantic caches over multiple passes. | **0 additional API calls**. | Counterfactual estimates of calls saved; cache misses reuse frozen outputs. Exact repeated texts are resolved by exact cache before similarity lookup. |

> [!WARNING]
> Do not treat offline simulation call counts as measured API billing costs, and do not conflate counterfactual replay estimates with live adaptive cache deployment. VectorQ integration and a live end-to-end adaptive cache comparison remain unverified in this repository.

---

## 3. Repository Layout

```text
adaptive-semantic-caching/
├── data/
│   ├── README.md                     # Dataset documentation, licenses, and provenance
│   └── bitext_sample_500.csv         # Permitted 500-sample intent classification subset
├── docs/
│   ├── EXPERIMENT_PROTOCOL.md        # Mathematical formulation and evaluation protocol
│   └── REPOSITORY_STATUS.md          # Architectural status and research notes
├── results/
│   ├── movie_reviews/                # Offline 10-seed simulation (runs.csv, pareto.csv, plots)
│   │   └── analysis/                 # Paired statistical comparisons vs static baseline
│   ├── real_groq/                    # Live Groq API baseline (291 evaluated reviews)
│   │   ├── baseline_500.csv          # Frozen predictions, latencies, and token counts
│   │   ├── baseline_compact.csv      # 10-review compact baseline
│   │   ├── baseline_500_analysis/    # Offline classification report and confusion matrix
│   │   └── cache_replay/             # Counterfactual exact + semantic cache replay
│   ├── bitext/                       # Bitext offline simulation benchmark results
│   ├── bitext_groq/                  # Bitext Groq 200-query baseline and replay
│   ├── quick/                        # Exploratory rapid offline simulation artifacts
│   └── real_nvidia/                  # Exploratory pilot baseline outputs
├── scripts/
│   ├── run_benchmark.py              # CLI runner for offline simulation benchmarks
│   ├── analyze_results.py            # Paired bootstrap CI analysis of offline simulation
│   ├── summarize_live_baseline.py    # Offline recomputation of live Groq baseline metrics
│   ├── run_groq_cache_experiment.py  # Counterfactual cache replay runner
│   ├── run_real_benchmark.py         # Live Groq baseline collector (requires API key)
│   ├── resume_real_benchmark.py      # Resumption harness for interrupted Groq runs
│   ├── run_bitext_api.py             # Live Groq runner on Bitext dataset
│   └── build_manifest.py             # Reproducibility checksum and metadata manifest builder
├── src/semantic_cache/
│   ├── benchmark.py                  # Offline tuning, evaluation stream, and Pareto frontier
│   ├── cache.py                      # Core ExactCache and SemanticCache implementations
│   ├── controllers.py                # Static, Selectivity, and Density threshold controllers
│   ├── datasets.py                   # Synthetic and NLTK movie-review data loaders
│   └── openrouter.py                 # Optional real-model API interface
├── tests/
│   ├── test_cache.py                 # Unit tests for exact and semantic cache logic
│   ├── test_controllers.py           # Unit tests for threshold adaptation bounds
│   └── test_analyze_results.py       # Unit tests for bootstrap analysis routines
├── pyproject.toml                    # Package configuration and pytest settings
└── reproducibility_manifest.json     # SHA-256 hashes, byte sizes, and environment specs
```

---

## 4. Setup & Dependency Installation

Use Python 3.10 to 3.13 (Python 3.12+ recommended). Install the package in editable mode with test dependencies:

```powershell
# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install package and test dependencies
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Verify the installation and test suite:
```powershell
python -m pytest -q
```
Expected output: `8 passed`.

---

## 5. Reproducing Offline Benchmarks & Statistical Analysis

The offline benchmark evaluates candidate caching configurations on a tuning split and measures test performance under nominal wrong-hit budgets: `[0.01, 0.02, 0.05, 0.10, 0.20]`.

### Run the offline simulation
```powershell
# Run with NLTK movie reviews (10 seeds, 60 queries per stream)
python .\scripts\run_benchmark.py --dataset nltk_movie_reviews --seeds 10 --queries 60 --output results\movie_reviews

# Or run with synthetic data (does not require external downloads)
python .\scripts\run_benchmark.py --dataset synthetic --seeds 10 --queries 60 --output results\synthetic
```

### Run paired statistical analysis
Compute paired differences, bootstrap 95% confidence intervals, and budget violation rates:
```powershell
python .\scripts\analyze_results.py --input results\movie_reviews\runs.csv --output results\movie_reviews\analysis
```

Outputs written to `results/movie_reviews/analysis/`:
- `summary_checked.csv`: Means, standard deviations, and test budget violation rates.
- `paired_vs_static.csv`: Paired bootstrap differences (calls and error) relative to tuned static caching.
- `analysis_metadata.json`: Timestamp, sample count, and configuration metadata.

---

## 6. Reproducing Saved Live Groq Baseline (Zero API Calls)

The repository includes a frozen baseline dataset at `results/real_groq/baseline_500.csv` containing live predictions from Groq (`openai/gpt-oss-20b`).

You can recompute all classification metrics, latency distributions, and token usage without an API key or network connection:
```powershell
python .\scripts\summarize_live_baseline.py --input results\real_groq\baseline_500.csv --output results\real_groq\baseline_500_analysis
```

### Verified Frozen Baseline Metrics
- **Model**: `openai/gpt-oss-20b` (Groq API, temperature=0, max_tokens=100, reasoning_effort="low")
- **Total Evaluated Records**: 291
- **Correct Predictions**: 248
- **Incorrect Predictions**: 43
- **Classification Accuracy**: 85.22% (`248 / 291`)
- **Macro F1 Score**: 0.8506
- **Mean Latency**: 0.5492 seconds (std: 0.1722s)
- **Total Recorded Tokens**: 184,912 (Prompt: 178,830, Completion: 6,082)
- **Confusion Matrix**:
  - Actual Class 0 (positive review): 109 true negative, 41 false positive
  - Actual Class 1 (negative review): 2 false negative, 139 true positive

---

## 7. Replaying Compact Cache Pilot (Zero API Calls)

To evaluate cache behavior without incurring API costs or nondeterminism, replay the 10-review compact baseline through exact and semantic cache models over 2 passes:

```powershell
python .\scripts\run_groq_cache_experiment.py --input results\real_groq\baseline_compact.csv --output results\real_groq\cache_replay
```

### Key Replay Architecture & Fixes:
- **Exact-Repeat Priority**: When an exact query string match is found in the cache, it is resolved as an `exact` hit immediately. It is not passed to vector similarity search or mislabeled as a semantic hit.
- **Counterfactual Decisions**: Pass 1 caches incoming reviews on misses; Pass 2 resolves identical repeated reviews as 100% exact hits (10 exact hits, 0 calls saved beyond exact cache).
- **Metadata Output**: Generates `results/real_groq/cache_replay/metadata.json` documenting model, input paths, thresholds, and confirming 0 live API calls were made.

---

## 8. Datasets & Provenance

### Bitext Customer Support Dataset
- **Upstream Source**: [Hugging Face (`bitext/Bitext-customer-support-llm-chatbot-training-dataset`)](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Sample (`data/bitext_sample_500.csv`)**: 500-sample representative subset (~348 KB) included for offline testing.
- **Full Corpus (`data/bitext_customer_support.csv`)**: Excluded from version control (~19.2 MB) via `.gitignore` to maintain lightweight cloning. See [`data/README.md`](data/README.md) for full provenance details.

### NLTK Movie Reviews Dataset
- **Source**: Pang & Lee (2004) via `nltk.corpus.movie_reviews` (2,000 sentiment-annotated reviews).
- **Usage**: Used for offline oracle simulation (TF-IDF + truncated SVD embeddings) and live sentiment classification benchmarks.

---

## 9. Reproducibility Manifest

A cryptographic audit trail of all result CSVs, scripts, datasets, and environment dependencies is generated via:

```powershell
python .\scripts\build_manifest.py
```

This writes [`reproducibility_manifest.json`](reproducibility_manifest.json), recording:
- SHA-256 checksums and byte lengths for all tracked data, code, and result files.
- CSV row counts and column schemas for all result tables.
- Exact Python interpreter version and installed versions of `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `requests`, and `pytest`.

---

## 10. Verification Suite

Run this full sequence to verify end-to-end repository reproducibility:

```powershell
python -m pip install -e ".[test]"
python -m pytest -q
python .\scripts\run_benchmark.py --help
python .\scripts\run_groq_cache_experiment.py --help
python .\scripts\summarize_live_baseline.py --input results\real_groq\baseline_500.csv --output results\real_groq\baseline_500_analysis
python .\scripts\run_groq_cache_experiment.py --input results\real_groq\baseline_compact.csv --output results\real_groq\cache_replay
python .\scripts\build_manifest.py
git diff --check
```
