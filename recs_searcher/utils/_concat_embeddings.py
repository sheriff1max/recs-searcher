"""
Алгоритмы для объединения эмбеддингов текстовых данных.
"""

from typing import List, Optional
import numpy as np


def concat_embeddings(
    embeddings: List[np.ndarray],
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """Конкатенация эмбеддингов, полученных разными алгоритмами.

        Параметры:
        ----------
        embeddings : List[np.ndarray]
            Список массивов-эмбеддингов.

        weights : List[float]
            Список весов важности каждого эмбеддинга при конкатенации.

        Returns
        -------
        embedding: np.ndarray
            Массив, объединяющий входные эмббединги `embeddings`.
    """
    if weights is None:
        weights = [1 for _ in range(len(embeddings))]

    if len(embeddings) < 2 or len(weights) < 2:
        raise ValueError('Error argument: len(weights) < 2 or len(embeddings) < 2')
    elif len(embeddings) != len(weights):
        raise ValueError('Error argument: len(weights) != len(embeddings)')

    array = [embeddings[i] * weights[i] for i in range(1, len(weights))]
    array.insert(0, embeddings[0] * weights[0])

    embedding = np.concatenate(array, axis=1)
    return embedding


if __name__ == '__main__':
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6, 7, 8], [9, 10, 11, 12]])
    k = np.array([[0], [0]])
    print(concat_embeddings([a, b, k], [2, 1, 1]))
