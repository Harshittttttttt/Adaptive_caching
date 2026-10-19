import pytest
from semantic_cache.controllers import static_threshold, selectivity_threshold, density_threshold, inverted_threshold


def test_static_is_constant():
    assert static_threshold(0.1, 0.2, 0.91) == pytest.approx(0.91)
    assert static_threshold(0.9, 0.8, 0.91) == pytest.approx(0.91)


def test_selectivity_rule_direction():
    assert selectivity_threshold(0.0, 0.0, 0.6, 0.95) == pytest.approx(0.95)
    assert selectivity_threshold(1.0, 0.0, 0.6, 0.95) == pytest.approx(0.6)


def test_density_rule_direction():
    assert density_threshold(0.5, 0.0, 0.6, 0.95) == pytest.approx(0.95)
    assert density_threshold(0.5, 1.0, 0.6, 0.95) == pytest.approx(0.6)


def test_inverted_rule_direction():
    assert inverted_threshold(0.0, 0.0, 0.6, 0.95) == pytest.approx(0.6)
    assert inverted_threshold(1.0, 0.0, 0.6, 0.95) == pytest.approx(0.95)
