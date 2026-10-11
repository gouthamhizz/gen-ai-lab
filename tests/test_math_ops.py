import numpy as np
from genai_platform.embeddings.math_ops import (cosine_loop, cosine_matrix, euclid_loop, euclid_matrix,
                                                 l2_normalize, top_k, verify_against_numpy)


def test_hand_written_matches_numpy():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = rng.normal(size=32), rng.normal(size=32)
        assert verify_against_numpy(a, b)["ok"]


def test_cosine_identity_with_euclid_on_unit_vectors():
    rng = np.random.default_rng(1)
    a, b = l2_normalize(rng.normal(size=(1, 16)))[0], l2_normalize(rng.normal(size=(1, 16)))[0]
    cos = cosine_loop(a.tolist(), b.tolist())
    euc = euclid_loop(a.tolist(), b.tolist())
    assert abs(euc**2 - (2 - 2 * cos)) < 1e-6


def test_cosine_matrix_matches_loop():
    rng = np.random.default_rng(2)
    q, d = rng.normal(size=(3, 8)), rng.normal(size=(5, 8))
    mat = cosine_matrix(q, d)
    for i in range(3):
        for j in range(5):
            assert abs(mat[i, j] - cosine_loop(q[i].tolist(), d[j].tolist())) < 1e-4


def test_euclid_matrix_matches_loop():
    rng = np.random.default_rng(3)
    q, d = rng.normal(size=(2, 6)), rng.normal(size=(4, 6))
    mat = euclid_matrix(q, d)
    for i in range(2):
        for j in range(4):
            assert abs(mat[i, j] - euclid_loop(q[i].tolist(), d[j].tolist())) < 1e-4


def test_top_k_orders_correctly():
    scores = np.array([0.1, 0.9, 0.5, 0.3])
    assert top_k(scores, 2) == [1, 2]
    assert top_k(scores, 2, largest=False) == [0, 3]


def test_zero_vector_cosine_is_defined_as_zero():
    assert cosine_loop([0, 0], [1, 1]) == 0.0
