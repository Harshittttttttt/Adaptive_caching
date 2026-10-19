import numpy as np
from semantic_cache.cache import SemanticCache


def test_exact_repeat_is_reused_without_decision():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    labels = np.array([1, 0])
    cache = SemanticCache(vectors, labels, lambda s, d: 0.99)
    cache.process_query(np.array([0, 0]), 1.0, 0.0)
    metrics = cache.metrics()
    assert metrics["calls"] == 1
    assert metrics["exact_reuses"] == 1
    assert metrics["decisions"] == 1


def test_wrong_semantic_hit_is_counted():
    vectors = np.array([[1.0, 0.0], [0.999, 0.001]], dtype=np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    labels = np.array([1, 0])
    cache = SemanticCache(vectors, labels, lambda s, d: 0.9)
    cache.process_query(np.array([0, 1]), 1.0, 1.0)
    metrics = cache.metrics()
    assert metrics["calls"] == 1
    assert metrics["semantic_hits"] == 1
    assert metrics["wrong_hits"] == 1
