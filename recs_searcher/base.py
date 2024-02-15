"""
Базовые классы.
"""


from abc import ABC, abstractmethod
from pathlib import Path
import pickle
from typing import Iterable, List, Optional
import random

import pandas as pd
import numpy as np

import pathlib
import platform
if platform.system() == 'Linux':
    pathlib.WindowsPath = pathlib.PosixPath


class BaseTransformation(ABC):
    """Абстрактный класс для трансформаторов текста."""

    def __init__(self, seed: Optional[int] = None):
        self._seed = seed

    @abstractmethod
    def _transform(self, array: List[str]) -> List[str]:
        """Преобразование, применяемое к каждому текстовому элементу списка array.

        Параметры
        ----------
        array : List[str]
            Список с текстом, который нужно преобразовать.
            Например,
            ['Hello! My nam3 is Harry :)', 'Понятно, а я Рон.'].

        Returns
        -------
        array: List[str]
            Список с применёнными преобразованиями текста.
        """

    def transform(self, array: Iterable[str]) -> List[str]:
        """Применение преобразования к массиву.

        Параметры
        ----------
        array : Iterable[str]
            Массив с текстом, который нужно преобразовать.

        Returns
        -------
        array: List[str]
            Список с применёнными преобразованиями текста.
        """
        array = list(array)

        random.seed(self._seed)
        array = self._transform(array)
        return array


class BaseEmbedding(ABC):
    """Абстрактный класс для эмбеддингов."""

    def load(self, path_folder_load: str, filename: str) -> None:
        """Загрузка vectorizer из файла.

        Параметры
        ----------
        path_folder_load : str
            Путь, где лежит файл.

        filename : str
            Название файла для загрузки.

        Returns
        -------
        None
        """
        if '.pkl' not in filename:
            filename += '.pkl'
        path = Path(path_folder_load) / filename

        with open(path, 'rb') as f:
            self = pickle.load(f)

    def save(self, path_folder_save: str, filename: str) -> None:
        """Сохранение vectorizer в файл.

        Параметры
        ----------
        path_folder_save : str
            Путь сохранения файла.

        filename : str
            Название файла для сохранения.

        Returns
        -------
        None
        """
        path_folder_save = Path(path_folder_save)
        if not path_folder_save.exists():
            path_folder_save.mkdir()

        if '.pkl' not in filename:
            filename += '.pkl'
        path = path_folder_save / filename

        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @abstractmethod
    def fit(self, array: Iterable[str]) -> object:
        """Обучение vectorizer.

        Параметры
        ----------
        array : Iterable[str]
            Массив с текстом, на котором обучается vectorizer.

        Returns
        -------
        self
        """

    @abstractmethod
    def transform(self, array: Iterable[str]) -> np.ndarray:
        """Обучение vectorizer.

        Параметры
        ----------
        array : Iterable[str]
            Массив с текстом, который нужно трансформировать в вектор чисел.

        Returns
        -------
        array: np.ndarray
            Массив с закодированными словами в числа.
        """

    def fit_transform(self, array: Iterable[str]) -> np.ndarray:
        """Обучение vectorizer и трансформация текста.
        Выполняет методы fit() и transform() текущего класса.

        Параметры
        ----------
        array : Iterable[str]
            Массив с текстом, на котором обучаемся и 
            который нужно трансформировать в вектор чисел.

        Returns
        -------
        array: np.ndarray
            Массив с закодированными словами в числа.
        """
        self.fit(array)
        return self.transform(array)


class BaseSearch(ABC):
    """Абстрактный класс для получения результатов
    поиска наиболее схожего текста из БД."""

    def __init__(
            self,
            original_array: Iterable[str],
            preprocessing: List[BaseTransformation],
    ):
        self._original_array = original_array
        self._preprocessing = preprocessing

    @abstractmethod
    def search(self, text: str, k: int) -> pd.DataFrame:
        """Поиск наиболее схожих k-текстов из БД на text пользователя.

        Параметры
        ----------
        text : str
            Пользовательский текст, которому нужно найти наиболее
            схожий текст из БД.

        k : int
            Кол-во выдаваемых результатов.

        Returns
        -------
        df: pd.DataFrame
            Датафрейм с результатами.
            df.columns = ['text', 'similarity']
        """


class BaseEmbeddingSearch(BaseSearch):
    """Абстрактный класс для получения результатов
    поиска наиболее схожего текста из БД на основе эмбеддингов.
    """

    def __init__(
            self,
            model: BaseEmbedding,
            embedding_database,
            original_array: Iterable[str],
            preprocessing: List[BaseTransformation],
            metric: str,
    ):
        super().__init__(
            original_array=original_array,
            preprocessing=preprocessing,
        )
        self._model = model
        self._embedding_database = embedding_database
        self._metric = metric

    @abstractmethod
    def search(self, text: str, k: int) -> pd.DataFrame:
        """Та же логика, что и в классе `BaseSearch`."""
