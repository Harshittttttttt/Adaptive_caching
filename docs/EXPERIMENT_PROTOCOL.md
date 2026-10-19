# Experiment protocol

## Claim under test

The proposed claim is that a threshold based on SQL-filter selectivity can reduce semantic-predicate evaluations at a fixed wrong-hit rate compared with a tuned static threshold. This is a hypothesis, not an assumed result.

## Current offline protocol

1. Load a labelled corpus.
2. Build TF-IDF + truncated-SVD embeddings.
3. Split each random seed into independent tuning and test row sets.
4. Generate query streams by selecting contiguous windows after sorting rows by word count. The window size controls selectivity.
5. Tune each method on the tuning stream for each error budget.
6. Evaluate the selected configuration on the test stream.
7. Report calls, wrong hits, semantic hits, and exact-repeat reuse.

## Important limitations

- Dataset labels stand in for an LLM's labels. No LLM calls are made.
- TF-IDF + SVD is not a modern neural embedding.
- The current word-count window is a controlled workload, not a production SQL workload.
- Density is measured on the candidate set. This uses information that a live system may not have without extra work.
- The benchmark uses one corpus unless the user runs multiple datasets.
- The cache is row-level. Predicate-specific cache keys and data freshness are not modelled yet.
- Threshold selection can be noisy. Use more seeds and report uncertainty.
- Cost and latency are not API measurements.

## Before making a paper claim

- Add a second dataset and a second semantic predicate.
- Add a neural embedding baseline and document its exact model/version.
- Evaluate predicate-specific cache keys.
- Include embedding, vector search, and controller overhead.
- Use real LLM labels or clearly label the offline oracle-label simulation.
- Compare with VectorQ/vCache if feasible. Do not claim a comparison that was not run.
- Save raw results and exact configuration for every run.
