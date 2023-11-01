"""
Алгоритмы для получения результатов моделей.
"""

from typing import Iterable, Union, Callable, List

import numpy as np
import pandas as pd

from base import BaseSearch, BaseEmbeddingSearch, BaseModel
from thefuzz import process
import faiss


class TheFuzzSearch(BaseSearch):
    """Класс поиска ближайших N-векторов в базе данных
    с помощью алгоритмов библиотеки TheFuzz."""

    def __init__(
            self,
            original_array: Iterable[str],
    ):
        self._original_array = original_array


    def search(self, text: str, k: int) -> pd.DataFrame:
        """"""

        results = process.extract(text, self._original_array, limit=k)

        lst_name = []
        lst_similarity = []
        for result in results:
            lst_name.append(result[0])
            lst_similarity.append(result[1])

        df = pd.DataFrame({'name': lst_name, 'similarity': lst_similarity})
        df = df.sort_values(by=['similarity'], ascending=False)
        df = df.head(k)
        return df


class ForCycleSearch(BaseEmbeddingSearch):
    """Класс поиска ближайших N-векторов в базе данных
    с помощью обычного цикла полного перебора."""

    def __init__(
            self,
            model: BaseModel,
            embedding_database: np.ndarray,
            original_array: Iterable[str],
            metric: Union[str, Callable] = '_cosine_distance',
    ):
        super().__init__(
            model=model,
            embedding_database=embedding_database,
            original_array=original_array,
            metric=metric
        )

    def search(self, text: str, k: int) -> pd.DataFrame:
        """"""

        text = [text]
        array = self._model.transform(text)
        array = array[0]

        lst_name = []
        lst_similarity = []

        for i, emb in enumerate(self._embedding_database):
            similarity = self._metric(emb, array)

            lst_name.append(self._original_array[i])
            lst_similarity.append(similarity)

        df = pd.DataFrame({'name': lst_name, 'similarity': lst_similarity})
        df = df.sort_values(by=['similarity'], ascending=False)
        df = df.head(k)
        return df


class FaissSearch(BaseEmbeddingSearch):
    """"""

    def __init__(
            self,
            model: BaseModel,
            embedding_database: np.ndarray,
            original_array: Iterable[str],
            metric: Union[str, Callable] = '_euclidean_distance',
    ):
        if metric == '_euclidean_distance':
            faiss_database = faiss.IndexFlatL2(embedding_database.shape[1])
        else:
            faiss_database = faiss.IndexFlatIP(embedding_database.shape[1])
        faiss_database.add(embedding_database)

        super().__init__(
            model=model,
            embedding_database=faiss_database,
            original_array=original_array,
            metric=metric
        )

    def search(self, text: str, k: int) -> pd.DataFrame:
        """"""

        text = [text]
        array = self._model.transform(text)

        distances, indices = self._embedding_database.search(array, k)
        distances, indices = distances[0], indices[0]

        lst_name = []
        lst_similarity = []
        for i in range(indices.shape[0]):
            lst_name.append(self._original_array[indices[i]])
            lst_similarity.append(distances[i])

        df = pd.DataFrame({'name': lst_name, 'similarity': lst_similarity})
        df = df.sort_values(by=['similarity'])
        return df
