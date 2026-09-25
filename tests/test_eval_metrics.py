import pytest

from scripts.eval_retrieval import hit_at_k, recall_at_k, reciprocal_rank


def test_hit_at_k_finds_relevant_within_k():
    assert hit_at_k(["a", "b", "c"], {"c"}, k=3) == 1


def test_hit_at_k_misses_relevant_outside_k():
    assert hit_at_k(["a", "b", "c"], {"c"}, k=2) == 0


def test_hit_at_k_empty_gold_is_zero():
    assert hit_at_k(["a"], set(), k=5) == 0


def test_reciprocal_rank_uses_first_hit_position():
    assert reciprocal_rank(["x", "y", "gold"], {"gold"}) == pytest.approx(1 / 3)


def test_reciprocal_rank_zero_when_absent():
    assert reciprocal_rank(["x"], {"gold"}) == 0.0


def test_recall_at_k_fraction_of_gold_found():
    assert recall_at_k(["a", "b", "c"], {"a", "c", "z"}, k=3) == pytest.approx(2 / 3)


def test_recall_at_k_empty_gold_is_zero_not_division_error():
    assert recall_at_k(["a"], set(), k=3) == 0.0


def test_metrics_are_order_sensitive_for_rr_but_not_hit():
    assert hit_at_k(["a", "b"], {"a"}, 2) == hit_at_k(["b", "a"], {"a"}, 2)
    assert reciprocal_rank(["a", "b"], {"a"}) > reciprocal_rank(["b", "a"], {"a"})
