"""
Базовые классы.
"""


from abc import ABC, abstractmethod
from pathlib import Path
import pickle
from typing import Iterable, List, Optional, Union
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
            preprocessing: List[BaseTransformation] = [],
            clear_array: Optional[Iterable[str]] = None,
    ):
        self._original_array = original_array
        self._preprocessing = preprocessing
        if clear_array is not None:
            self._clear_text = clear_array
        else:
            self._clear_text = self._preprocessing_text(original_array, preprocessing)

    def _preprocessing_text(
        self,
        text: Union[Iterable[str], str],
        preprocessing: List[BaseTransformation],
    ) -> Union[List[str], str]:
        """"""
        if isinstance(text, str):
            for transformator in preprocessing:
                tmp_text = transformator.transform([text])[0]
                if tmp_text:
                    text = tmp_text
        else:
            for transformator in preprocessing:
                text = transformator.transform(text)
        return text

    @abstractmethod
    def search(
        self,
        text: str,
        k: int,
        ascending: bool = False,
    ) -> pd.DataFrame:
        """Поиск наиболее схожих k-текстов из БД на text пользователя.

        Параметры
        ----------
        text : str
            Пользовательский текст, которому нужно найти наиболее
            схожий текст из БД.

        k : int
            Кол-во выдаваемых результатов.

        ascending : bool
            Флаг сортировки полученных результатов.
            False - убывающая, True - возрастающая сортировка.

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
            metric: str,
            preprocessing: List[BaseTransformation] = [],
            clear_array: Optional[Iterable[str]] = None,
    ):
        super().__init__(
            original_array=original_array,
            preprocessing=preprocessing,
            clear_array=clear_array,
        )
        self._model = model
        self._embedding_database = embedding_database
        self._metric = metric

    @abstractmethod
    def search(self, text: str, k: int) -> pd.DataFrame:
        """Та же логика, что и в классе `BaseSearch`."""


class BaseExplain(ABC):
    """Абстрактный класс для интерпретации схожести двух текстовых данных."""

    def __init__(
        self,
        model: BaseEmbedding,
        preprocessing: List[BaseTransformation] = [],
    ):
        self._model = model
        self._preprocessing = preprocessing

    def _preprocessing_text(
        self,
        text: str,
        preprocessing: List[BaseTransformation],
    ) -> str:
        """"""
        for transformator in preprocessing:
            tmp_text = transformator.transform([text])[0]
            if tmp_text:
                text = tmp_text
        return text

    @abstractmethod
    def _explain(
        self,
        clear_compared_text: str,
        clear_original_text: str,
        n_grams: int = 1,
    ) -> pd.DataFrame:
        """
        Returns
        -------
        df: pd.DataFrame
            Датафрейм с результатами.
            df.columns = ['text', 'similarity']
        """

    def explain(
        self,
        compared_text: str,
        original_text: str,
        n_grams: int = 1,
        k: int = 10,
        ascending: bool = True,
    ) -> pd.DataFrame:
        """"""
        if n_grams == 0:
            raise ValueError('The `n_grams` parameter must be > 0')

        clear_compared_text = self._preprocessing_text(compared_text, self._preprocessing)
        clear_original_text = self._preprocessing_text(original_text, self._preprocessing)

        df = self._explain(clear_compared_text, clear_original_text, n_grams=n_grams)
        df = df.sort_values(by=['similarity'], ascending=ascending).reset_index(drop=True)

        k = min(k, df.shape[0])
        df = df.iloc[:k]
        return df
