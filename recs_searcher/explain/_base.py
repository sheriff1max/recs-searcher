"""
Алгоритмы для объяснение сходства одного текста на другой.
"""

from ..base import BaseExplain, BaseEmbedding, BaseTransformation
from ..utils import cosine_distance, euclidean_distance

from typing import List, Union, Callable, Literal, Tuple, Optional
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
        """
        Определение функции для подсчёта расстояния между 2 векторами.

        Параметры
        ----------
        distance : Union[Literal['cosine', 'euclidean'], Callable[[np.ndarray, np.ndarray], float]]
            Текст - для получения функции, реализованной в данном проекта.
            Функция - для встраивания своей функции для подсчёта расстояния.

        Returns
        -------
        distance: Callable[[np.ndarray, np.ndarray], float]
            Функция для подсчёта расстояния между двумя векторами.
        """
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
        n_grams: Optional[int] = 1,
        analyzer: Literal['word', 'char'] = 'word',
        sep: Optional[str] = ' ',
    ) -> Tuple[List[str], List[float], List[Tuple[int, int]]]:
        if analyzer == 'word':
            n_grams_list, indeces_n_grams_list = self._split_by_words(
                clear_text=clear_compared_text,
                n_grams=n_grams,
                sep=sep,
            )
        elif analyzer == 'char':
            n_grams_list, indeces_n_grams_list = self._split_by_chars(
                clear_text=clear_compared_text,
                n_grams=n_grams,
                sep=sep,
            )
        else:
            raise ValueError(f'The `analyzer` parameter cannot take a value {analyzer}')

        text_list = []
        similarity_list = []

        clear_original_embedding = self._model.transform([clear_original_text])
        for n_gram in n_grams_list:
            n_gram_embedding = self._model.transform([n_gram])

            distance = self._distance(clear_original_embedding, n_gram_embedding)

            text_list.append(n_gram)
            similarity_list.append(distance)

        if len(text_list) == 0:
            raise ValueError(f'The `n_grams` parameter must be <= {len(n_grams_list)}')

        return text_list, similarity_list, indeces_n_grams_list
