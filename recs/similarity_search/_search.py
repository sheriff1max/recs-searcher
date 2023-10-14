"""
Алгоритмы для получения результатов моделей.
"""

from typing import Iterable, Union, Callable, List

import numpy as np
import pandas as pd

from base import BaseSearch, BaseModel


class ForCycleSearch(BaseSearch):
    """Класс поиска ближайших N-векторов в базе данных
    для одного входного вектора."""

    def __init__(
            self,
            model: BaseModel,
            embedding_data: np.ndarray,
            name_data: Iterable[str],
            metric: Union[str, Callable] = '_cosine_distance',
    ):
        super().__init__(
            model=model,
            embedding_data=embedding_data,
            name_data=name_data,
            metric=metric
        )

    def search(self, text: str, k: int) -> pd.DataFrame:
        """"""

        text = [text]
        array = self._model.transform(text)
        array = array[0]

        lst_name = []
        lst_similarity = []

        for i, emb in enumerate(self._embedding_data):
            similarity = self._metric(emb, array)

            lst_name.append(self._name_data[i])
            lst_similarity.append(similarity)

        df = pd.DataFrame({'name': lst_name,'similarity': lst_similarity})
        df = df.sort_values(by=['similarity'], ascending=False)
        df = df.head(k)

        return df


# class FaissSearch(BaseSearch):
#     """"""

#     def __init__(
#             self,
#             model: BaseModel,
#             embedding_data: np.ndarray,
#             name_data: pd.DataFrame,
#             metric: Union[str, Callable] = 'cosine_distance',
#     ):
#         super().__init__(
#             model=model,
#             embedding_data=embedding_data,
#             name_data=name_data,
#             metric=metric
#         )

#     def search(self, text: str, k: int) -> pd.DataFrame:
#         """"""
