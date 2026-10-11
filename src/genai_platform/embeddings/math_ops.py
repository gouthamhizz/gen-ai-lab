"""STEP 2a - the maths, by hand, BEFORE any library abstraction.

    dot(A, B)      = sum_i A_i * B_i
    ||A||          = sqrt(sum_i A_i^2)
    cos(theta)     = (A . B) / (||A|| * ||B||)          in [-1, 1]; 1 = same direction
    euclid(A, B)   = sqrt(sum_i (A_i - B_i)^2)

Key identity (why vector DBs can use either metric on normalised vectors):
    for unit vectors:  ||A - B||^2 = 2 - 2*cos(theta)

Functions come in two flavours:
  * *_loop : plain Python loops - exactly the formula, slow, used to *understand*
  * vectorised NumPy versions - what the pipeline actually uses (matrix-matrix in one shot)
`verify_against_numpy` proves the two agree, the same "gradient-check" mindset as the DL labs.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


# ---- flavour 1: literal formulas ------------------------------------------------------
def dot_loop(a: Sequence[float], b: Sequence[float]) -> float:
    assert len(a) == len(b), "vectors must have equal dimension"
    total = 0.0
    for x, y in zip(a, b):
        total += x * y
    return total


def norm_loop(a: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def cosine_loop(a: Sequence[float], b: Sequence[float]) -> float:
    na, nb = norm_loop(a), norm_loop(b)
    if na == 0 or nb == 0:  # cos undefined for the zero vector
        return 0.0
    return dot_loop(a, b) / (na * nb)


def euclid_loop(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ---- flavour 2: vectorised (batch) ----------------------------------------------------
def l2_normalize(m: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    m = np.atleast_2d(m).astype(np.float32)
    return m / np.maximum(np.sqrt((m * m).sum(axis=1, keepdims=True)), eps)


def cosine_matrix(queries: np.ndarray, docs: np.ndarray) -> np.ndarray:
    """(q, d) x (n, d) -> (q, n) cosine similarities. After L2-normalising, cosine == dot product."""
    return l2_normalize(queries) @ l2_normalize(docs).T


def euclid_matrix(queries: np.ndarray, docs: np.ndarray) -> np.ndarray:
    """||q-d||^2 = ||q||^2 + ||d||^2 - 2 q.d   (avoids materialising an (q, n, dim) tensor)."""
    q, d = np.atleast_2d(queries).astype(np.float64), np.atleast_2d(docs).astype(np.float64)
    sq = (q * q).sum(1, keepdims=True) + (d * d).sum(1) - 2 * q @ d.T
    return np.sqrt(np.maximum(sq, 0.0))


def top_k(scores: np.ndarray, k: int, largest: bool = True) -> list[int]:
    """Brute-force exact nearest neighbours for ONE query row - what an ANN index approximates."""
    order = np.argsort(-scores if largest else scores)
    return order[:k].tolist()


def verify_against_numpy(a: np.ndarray, b: np.ndarray, tol: float = 1e-5) -> dict:
    """Compare hand-written loops with NumPy; returns the absolute errors (should be ~1e-7)."""
    cos_np = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    euc_np = float(np.linalg.norm(a - b))
    res = {
        "cosine_err": abs(cosine_loop(a.tolist(), b.tolist()) - cos_np),
        "euclid_err": abs(euclid_loop(a.tolist(), b.tolist()) - euc_np),
    }
    res["ok"] = all(v < tol for v in res.values())
    return res
