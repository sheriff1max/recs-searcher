"""
Алгоритмы для подсчёта расстояния двух векторов.
"""


import numpy as np


def cosine_distance(a, b):
    """Подсчёт косинусого расстояние между вектрами."""
    if isinstance(a, np.ndarray):
        a = a.reshape(-1)
    if isinstance(b, np.ndarray):
        b = b.reshape(-1)

    dot_product = np.dot(a, b)  # x * y
    norm_a = np.linalg.norm(a)  # |x|
    norm_b = np.linalg.norm(b)  # |y|
    return 1 - dot_product / (norm_a * norm_b)


def euclidean_distance(a, b):
    """Подсчёт евклидова расстояния между вектрами."""
    return np.linalg.norm(a - b)
