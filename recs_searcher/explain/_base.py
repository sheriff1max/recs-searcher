"""
Алгоритмы для объяснение сходства одного текста на другой.
"""

from ..base import BaseExplain, BaseEmbedding, BaseTransformation
from ..utils import cosine_distance, euclidean_distance

from typing import List, Union, Callable, Literal
import pandas as pd
import numpy as np


class DistanceExplain(BaseExplain):
    """Класс для интерпретации сходства двух текстовых данных
    путём взятия окна из n-токенов из оригинального текста и подсчёта
    расстояния их эмбеддингов с оригинальным."""

    def __init__(
        self,
        model: BaseEmbedding,
        preprocessing: List[BaseTransformation] = [],
        distance: Union[
            Literal['cosine', 'euclidean'],
            Callable[[np.ndarray, np.ndarray], float],
        ] = 'cosine',
    ):
        super().__init__(
            model=model,
            preprocessing=preprocessing,
        )
        self._distance = self.__define_distance(distance)

    def __define_distance(
        self,
        distance: Union[
            Literal['cosine', 'euclidean'],
            Callable[[np.ndarray, np.ndarray], float],
        ],
    ) -> Callable[[np.ndarray, np.ndarray], float]:
        """"""
        if distance == 'cosine':
            return cosine_distance
        elif distance == 'euclidean':
            return euclidean_distance
        else:
            return distance

    def _explain(
        self,
        clear_compared_text: str,
        clear_original_text: str,
        n_grams: int = 1,
    ) -> pd.DataFrame:
        """"""
        tokens_list = clear_compared_text.split(' ')

        text_list = []
        similarity_list = []

        clear_original_embedding = self._model.transform([clear_original_text])
        for i in range(len(tokens_list) - n_grams + 1):
            n_tokens_list = tokens_list[i:i + n_grams]
            cut_text = ' '.join(n_tokens_list)
            cut_text_embedding = self._model.transform([cut_text])

            distance = self._distance(clear_original_embedding, cut_text_embedding)

            text_list.append(cut_text)
            similarity_list.append(distance)

        if len(text_list) == 0:
            raise ValueError(f'The `n_grams` parameter must be <= {len(tokens_list)}')

        df = pd.DataFrame({'text': text_list, 'similarity': similarity_list})
        return df
