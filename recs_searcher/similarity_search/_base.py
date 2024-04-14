"""
Алгоритмы для получения результатов моделей.
"""

from typing import Iterable, List, Literal, Optional, Tuple
import numpy as np

from ..utils import create_date_name
from ..base import BaseSearch, BaseEmbeddingSearch, BaseEmbedding, BaseTransformation

from thefuzz import process
import faiss
import chromadb
from sklearn.neighbors import NearestNeighbors


class TheFuzzSearch(BaseSearch):
    """Класс поиска наиболее похожих слов в БД
    с помощью расстояния Ливенштейна.
    Основано на: https://github.com/seatgeek/thefuzz
    """

    def __init__(
            self,
            original_array: Iterable[str],
            preprocessing: List[BaseTransformation],
            clear_array: Optional[Iterable[str]] = None,
    ):
        super().__init__(
            original_array=original_array,
            preprocessing=preprocessing,
            clear_array=clear_array,
        )

    def _search(
        self,
        clear_text: str,
        k: int,
    ) -> Tuple[List[str], List[float]]:
        results = process.extract(clear_text, self._clear_text, limit=k)

        list_texts = []
        list_similarity = []
        for result in results:
            list_similarity.append(result[1])

            index = np.where(self._clear_text == result[0])
            list_texts.append(self._original_array[index][0])

        return list_texts, list_similarity


class NearestNeighborsSearch(BaseEmbeddingSearch):
    """
    Поиск на основе ближайших соседей.
    Основа на: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors
    """

    def __init__(
            self,
            model: BaseEmbedding,
            embedding_database: np.ndarray,
            original_array: Iterable[str],
            preprocessing: List[BaseTransformation],
            clear_array: Optional[Iterable[str]] = None,

            radius: float = 1.0,
            algorithm: str = 'auto',
            leaf_size: int = 30,
            metric: Literal['cosine', 'l1', 'l2', 'minkowski', 'manhattan', 'cosine'] = 'cosine',
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
            clear_array=clear_array,
        )

        self._knn = None
        self._radius = radius
        self._algorithm = algorithm
        self._leaf_size = leaf_size
        self._p = p
        self._metric_params = metric_params
        self._n_jobs = n_jobs

    def _search(
        self,
        array: np.ndarray,
        k: int,
    ) -> Tuple[List[str], List[float]]:
        if not self._knn or self._knn.n_neighbors != k:
            self._knn = NearestNeighbors(
                n_neighbors=k,
                radius=self._radius,
                algorithm=self._algorithm,
                leaf_size=self._leaf_size,
                metric=self._metric,
                p=self._p,
                metric_params=self._metric_params,
                n_jobs=self._n_jobs,
            ).fit(self._embedding_database)

        list_similarity, list_texts = self._knn.kneighbors(array)
        list_similarity = list_similarity[0]
        list_texts = self._original_array[list_texts[0]]
        return list_texts, list_similarity


class FaissSearch(BaseEmbeddingSearch):
    """
    Основано на: https://github.com/facebookresearch/faiss
    """

    def __init__(
            self,
            model: BaseEmbedding,
            embedding_database: np.ndarray,
            original_array: Iterable[str],
            preprocessing: List[BaseTransformation],
            count_voronoi_cells: Optional[int] = 1,  # > 1 if type_optimization_searcher != None
            type_optimization_searcher: Optional[Literal['IVF', 'IVFPQ']] = None,
            number_centroids: Optional[int] = 8,  # For type_optimization_searcher='IVFPQ'
            bits: Optional[int] = 8,  # For type_optimization_searcher='IVFPQ'
            clear_array: Optional[Iterable[str]] = None,
    ):
        faiss_database = faiss.IndexFlatL2(embedding_database.shape[1])
        if count_voronoi_cells > 1:
            if type_optimization_searcher == 'IVF':
                faiss_database = faiss.IndexIVFFlat(
                    faiss_database,
                    embedding_database.shape[1],
                    count_voronoi_cells
                )
            elif type_optimization_searcher == 'IVFPQ':
                faiss_database = faiss.IndexIVFPQ(
                    faiss_database,
                    embedding_database.shape[1],
                    count_voronoi_cells,
                    number_centroids,
                    bits,
                )
            else:
                raise ValueError('If `count_voronoi_cells` > 1 then `type_optimization` must not be equal to None')
            faiss_database.train(embedding_database)

        faiss_database.add(embedding_database)

        super().__init__(
            model=model,
            embedding_database=faiss_database,
            original_array=original_array,
            preprocessing=preprocessing,
            metric=None,
            clear_array=clear_array,
        )

    def _search(
        self,
        array: np.ndarray,
        k: int,
    ) -> Tuple[List[str], List[float]]:
        distances, indices = self._embedding_database.search(array, k)
        distances, indices = distances[0], indices[0]

        list_texts = []
        list_similarity = []
        for i in range(indices.shape[0]):
            list_texts.append(self._original_array[indices[i]])
            list_similarity.append(distances[i])
        return list_texts, list_similarity


class ChromaDBSearch(BaseEmbeddingSearch):
    """
    Основано на: https://github.com/chroma-core/chroma
    """

    _MAX_BATCH_CHROMA_DB = 5000

    def __init__(
            self,
            model: BaseEmbedding,
            embedding_database: np.ndarray,
            original_array: Iterable[str],
            preprocessing: List[BaseTransformation],
            metric: Literal['l2', 'ip', 'cosine'] = 'cosine',
            clear_array: Optional[Iterable[str]] = None,
    ):
        chroma_client = chromadb.Client()
        name_database = create_date_name('database')
        chroma_database = chroma_client.create_collection(
            name=name_database,
            metadata={"hnsw:space": metric}
        )

        if original_array.shape[0] > self._MAX_BATCH_CHROMA_DB:
            for i in range(0, original_array.shape[0] - self._MAX_BATCH_CHROMA_DB, self._MAX_BATCH_CHROMA_DB):
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
            metric=metric,
            clear_array=clear_array,
        )

    def _search(
        self,
        array: np.ndarray,
        k: int,
    ) -> Tuple[List[str], List[float]]:
        result = self._embedding_database.query(
            query_embeddings=array.tolist(),
            n_results=k,
        )
        list_texts, list_similarity = result['ids'][0], result['distances'][0]
        return list_texts, list_similarity
