from evaluation.ranking_metrics import dcg_at_k, hit_rate_at_k, mrr, ndcg_at_k, recall_at_k, relative_improvement


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k([3, 2, 1, 0], k=10) == 1.0


def test_ndcg_inverted_ranking_is_worse():
    perfect = ndcg_at_k([3, 2, 1, 0], k=4)
    inverted = ndcg_at_k([0, 1, 2, 3], k=4)
    assert inverted < perfect


def test_ndcg_all_zeros_is_zero():
    assert ndcg_at_k([0, 0, 0], k=10) == 0.0


def test_dcg_position_discount():
    high_first = dcg_at_k([3, 0], k=2)
    high_second = dcg_at_k([0, 3], k=2)
    assert high_first > high_second


def test_recall_at_k():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"c", "z"}
    assert recall_at_k(retrieved, relevant, k=10) == 0.5
    assert recall_at_k(retrieved, relevant, k=2) == 0.0


def test_hit_rate_and_mrr():
    retrieved = ["x", "y", "rel"]
    relevant = {"rel"}
    assert hit_rate_at_k(retrieved, relevant, k=3) == 1.0
    assert hit_rate_at_k(retrieved, relevant, k=2) == 0.0
    assert mrr(retrieved, relevant) == 1.0 / 3.0


def test_relative_improvement():
    assert abs(relative_improvement(1.11, 1.0) - 0.11) < 1e-12
    assert relative_improvement(1.0, 0.0) == 0.0
