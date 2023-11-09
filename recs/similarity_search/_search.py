"""
Алгоритмы для получения результатов моделей.
"""

from typing import Iterable, Union, Callable, List
import datetime

import numpy as np
import pandas as pd

from utils import _create_date_name

from base import BaseSearch, BaseEmbeddingSearch, BaseModel
from thefuzz import process
import faiss
import chromadb
from sklearn.neighbors import NearestNeighbors


class ForCycleSearch(BaseEmbeddingSearch):
    """Класс поиска ближайших N-векторов в базе данных
    с помощью обычного цикла полного перебора.
    
    Реализация крайне медленная из-за полного перебора
    базы данных. Советуется данную реализацию использовать
    только на небольших датасетах (меньше 4к записей)."""

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


class NearestNeighborsSearch(BaseEmbeddingSearch):
    """
    Реализация: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors
    """

    def __init__(
            self,
            model: BaseModel,
            embedding_database: np.ndarray,
            original_array: Iterable[str],

            radius: float = 1.0,
            algorithm: str = 'auto',
            leaf_size: int = 30,
            metric: Union[str, Callable] = 'cosine',
            p: float = 2,
            metric_params: dict = None,
            n_jobs: int = None,
    ):
        super().__init__(
            model=model,
            embedding_database=embedding_database,
            original_array=original_array,
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
            metric: Union[str, Callable] = '_euclidean_distance',
    ):
        if metric == '_euclidean_distance':
            faiss_database = faiss.IndexFlatL2(embedding_database.shape[1])
        elif metric == '_inner_product_distance':
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


class ChromaDBSearch(BaseEmbeddingSearch):
    """"""

    _MAX_BATCH_CHROMA_DB = 5000

    def __init__(
            self,
            model: BaseModel,
            embedding_database: np.ndarray,
            original_array: Iterable[str],
            metric: Union[str, Callable] = '_cosine_distance',
    ):

        if metric == '_euclidean_distance' or metric == 'l2':
            metric = 'l2'
        elif metric == '_cosine_distance' or metric == 'cosine':
            metric = 'cosine'
        elif metric == '_inner_product_distance' or metric == 'ip':
            metric = 'ip'

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
            metric=metric
        )

    def search(self, text: str, k: int) -> pd.DataFrame:
        """"""

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
