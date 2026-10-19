# Repository status

Initial repository scaffold. Unit tests and a one-seed synthetic smoke test passed in the build environment.

This does not mean the scientific hypothesis is validated. The full benchmark, NLTK dataset run, real LLM integration, provider cost measurement, and comparison with VectorQ/vCache have not been completed.

Known implementation limits for the next iteration:

- The offline benchmark uses dataset labels as oracle answers.
- The OpenRouter client is a helper only; it is not connected to the benchmark loop.
- The current query generator creates word-count windows, not arbitrary SQL predicates.
- The current cache is row-level and does not key entries by predicate.
- The density calculation is an offline diagnostic and may add significant overhead in larger datasets.
