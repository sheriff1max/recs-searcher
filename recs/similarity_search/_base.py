"""
Алгоритмы для получения результатов моделей.
"""

from typing import Iterable, Union, Callable, List, Literal
import datetime

import numpy as np
import pandas as pd

from utils import _create_date_name

from base import BaseSearch, BaseEmbeddingSearch, BaseModel, BaseTransformation
from thefuzz import process
import faiss
import chromadb
from sklearn.neighbors import NearestNeighbors


class TheFuzzSearch(BaseSearch):
    """Класс поиска ближайших N-векторов в базе данных
    с помощью алгоритмов библиотеки TheFuzz."""

    def __init__(
            self,
            original_array: Iterable[str],
            preprocessing: List[BaseTransformation],
    ):
        self._original_array = original_array
        super().__init__(
            original_array=original_array,
            preprocessing=preprocessing,
        )

    def search(self, text: str, k: int) -> pd.DataFrame:
        """"""

        for transformator in self._preprocessing:
            tmp_text = transformator.transform([text])[0]
            if tmp_text:
                text = tmp_text

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


class NearestNeighborsSearch(BaseEmbeddingSearch):
    """
    Реализация: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors
    """

    def __init__(
            self,
            model: BaseModel,
            embedding_database: np.ndarray,
            original_array: Iterable[str],
            preprocessing: List[BaseTransformation],

            radius: float = 1.0,
            algorithm: str = 'auto',
            leaf_size: int = 30,
            metric: Literal['cosine', 'l1', 'l2', 'minkowski', 'manhattan', 'cosine', 'haversine'] = 'cosine',
            p: float = 2,
            metric_params: dict = None,
            n_jobs: int = None,
    ):
        super().__init__(
            model=model,
            embedding_database=embedding_database,
            original_array=original_array,
            preprocessing=preprocessing,
            metric=metric,
        )

        self._knn = None
        self._radius = radius
        self._algorithm = algorithm
        self._leaf_size = leaf_size
        self._p = p
        self._metric_params = metric_params
        self._n_jobs = n_jobs

    def search(self, text: str, k: int) -> pd.DataFrame:
        """"""

        if not self._knn or self._knn.n_neighbors != k:
            self._knn = NearestNeighbors(
                n_neighbors=k,
                radius=self._radius,
                algorithm=self._algorithm,
                leaf_size=self._leaf_size,
                metric=self._metric,  # l1, l2, minkowski, manhattan, cosine, haversine and etc from sklearn.
                p=self._p,
                metric_params=self._metric_params,
                n_jobs=self._n_jobs,
            ).fit(self._embedding_database)

        for transformator in self._preprocessing:
            tmp_text = transformator.transform([text])[0]
            if tmp_text:
                text = tmp_text

        text = [text]
        array = self._model.transform(text)

        lst_similarity, lst_name = self._knn.kneighbors(array)
        lst_similarity = lst_similarity[0]
        lst_name = self._original_array[lst_name[0]]
        if self._metric == 'cosine':
            lst_similarity = list(map(lambda x: 1 - x, lst_similarity))

        df = pd.DataFrame({'name': lst_name, 'similarity': lst_similarity})
        df = df.sort_values(by=['similarity'], ascending=False)
        return df


class FaissSearch(BaseEmbeddingSearch):
    """"""

    def __init__(
            self,
            model: BaseModel,
            embedding_database: np.ndarray,
            original_array: Iterable[str],
            preprocessing: List[BaseTransformation],
            metric: Literal['l2', 'ip'] = 'l2',
    ):
        if metric == 'l2':
            faiss_database = faiss.IndexFlatL2(embedding_database.shape[1])
        elif metric == 'ip':
            faiss_database = faiss.IndexFlatIP(embedding_database.shape[1])
        faiss_database.add(embedding_database)

        super().__init__(
            model=model,
            embedding_database=faiss_database,
            original_array=original_array,
            preprocessing=preprocessing,
            metric=metric
        )

    def search(self, text: str, k: int) -> pd.DataFrame:
        """"""

        for transformator in self._preprocessing:
            tmp_text = transformator.transform([text])[0]
            if tmp_text:
                text = tmp_text

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


class ChromaDBSearch(BaseEmbeddingSearch):
    """"""

    _MAX_BATCH_CHROMA_DB = 5000

    def __init__(
            self,
            model: BaseModel,
            embedding_database: np.ndarray,
            original_array: Iterable[str],
            preprocessing: List[BaseTransformation],
            metric: Literal['l2', 'ip', 'cosine'] = 'cosine',
    ):
        chroma_client = chromadb.Client()
        name_database = _create_date_name('database')
        chroma_database = chroma_client.create_collection(
            name=name_database,
            metadata={"hnsw:space": metric}
        )

        if original_array.shape[0] > self._MAX_BATCH_CHROMA_DB:
            for i in range(0, original_array.shape[0]-self._MAX_BATCH_CHROMA_DB, self._MAX_BATCH_CHROMA_DB):
                chroma_database.add(
                    embeddings=embedding_database[i:i+self._MAX_BATCH_CHROMA_DB].tolist(),
                    ids=original_array[i:i+self._MAX_BATCH_CHROMA_DB].tolist(),
                )
        else:
            chroma_database.add(
                embeddings=embedding_database.tolist(),
                ids=original_array.tolist(),
            )

        super().__init__(
            model=model,
            embedding_database=chroma_database,
            original_array=original_array,
            preprocessing=preprocessing,
            metric=metric
        )

    def search(self, text: str, k: int) -> pd.DataFrame:
        """"""

        for transformator in self._preprocessing:
            tmp_text = transformator.transform([text])[0]
            if tmp_text:
                text = tmp_text

        text = [text]
        array = self._model.transform(text)

        result = self._embedding_database.query(
            query_embeddings=array.tolist(),
            n_results=k,
        )
        lst_name, lst_similarity = result['ids'][0], result['distances'][0]
        if self._metric == 'cosine':
            lst_similarity = list(map(lambda x: 1 - x, lst_similarity))

        df = pd.DataFrame({'name': lst_name, 'similarity': lst_similarity})
        return df
