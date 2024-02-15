"""
Алгоритмы для объединения эмбеддингов текстовых данных.
"""

from typing import List
import numpy as np

# TODO: доделать всё.

def concat_embeddings(
    embeddings: List[np.ndarray],
    weights: List[float],
) -> np.ndarray:
    """"""
    array = [embeddings[i].T * weights[i] for i in range(1, len(weights))]
    array.insert(0, embeddings[0] * weights[0])

    result = np.concatenate(array, axis=1)
    return result
print(concat_embeddings())

a = np.array([[1, 2]])
b = np.array([[5, 6, 7, 8]])

print(np.concatenate([a, b], axis=1))
# embeddings = [
#     np.array([[1, 2]]),
#     np.array([[5, 6, 7, 8]]),
# ]
# print(concat_embeddings())
