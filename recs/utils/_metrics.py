"""
Метрики для подсчёта расстояния между векторами (эмбеддингами).
"""


import numpy as np


def _cosine_distance(
        array1: np.ndarray,
        array2: np.ndarray,
) -> float:
    """Косинусова расстояние."""

    similarity = np.dot(array1, array2) / \
        (np.linalg.norm(array1) * np.linalg.norm(array2) + 0.0001)
    return similarity


def _euclidean_distance(
        array1: np.ndarray,
        array2: np.ndarray,
) -> float:
    """Евклидово расстояние."""

    similarity = np.linalg.norm(array1 - array2)
    return similarity


def _inner_product_distance(
        array1: np.ndarray,
        array2: np.ndarray,
) -> float:
    """Внутреннее произведение."""

    similarity = np.inner(array1, array2)
    return similarity
