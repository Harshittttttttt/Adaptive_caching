from __future__ import annotations
import numpy as np


class SemanticCache:
    """Row-level nearest-neighbour cache with oracle labels for offline evaluation."""
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray, threshold_fn):
        self.embeddings = embeddings
        self.labels = labels
        self.threshold_fn = threshold_fn
        self.cached_ids: set[int] = set()
        self.cache_ids: list[int] = []
        self.calls = 0
        self.wrong_hits = 0
        self.hits = 0
        self.exact_reuses = 0
        self.decisions = 0

    def process_query(self, row_ids: np.ndarray, selectivity: float, density: float):
        for row_id in row_ids:
            row_id = int(row_id)
            if row_id in self.cached_ids:
                self.exact_reuses += 1
                continue
            self.decisions += 1
            threshold = self.threshold_fn(selectivity, density)
            if self.cache_ids:
                ids = np.asarray(self.cache_ids, dtype=int)
                sims = self.embeddings[ids] @ self.embeddings[row_id]
                pos = int(np.argmax(sims))
                if float(sims[pos]) >= threshold:
                    self.hits += 1
                    self.wrong_hits += int(self.labels[ids[pos]] != self.labels[row_id])
                    continue
            self.calls += 1
            self.cached_ids.add(row_id)
            self.cache_ids.append(row_id)

    def metrics(self):
        return {
            "calls": self.calls,
            "semantic_hits": self.hits,
            "exact_reuses": self.exact_reuses,
            "wrong_hits": self.wrong_hits,
            "decisions": self.decisions,
            "wrong_hit_rate": self.wrong_hits / self.decisions if self.decisions else 0.0,
        }
