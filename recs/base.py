"""
Базовые классы.
"""


from abc import ABC, abstractmethod
import os
import pickle
from typing import Union, Callable, Iterable, List
from inspect import getmembers, isfunction

import pandas as pd
import numpy as np

from torch.utils.data import Dataset

from dataset import _dataframes
from utils import _metrics, WrapperTransform


CUR_PATH = os.path.dirname(__file__)
PATH_SAVE_MODEL = CUR_PATH + '\\models\\weights'
if not os.path.exists(PATH_SAVE_MODEL):
    os.mkdir(PATH_SAVE_MODEL)


class BaseTransformation(ABC):
    """Абстрактный класс для трансформаторов текста."""

    def _one_transform(
            self,
            array: List[str],
            transformation: Union[None, Callable[[str], str]],
    ) -> List[str]:
        """Применение одного преобразования к списку текста.

        Параметры
        ----------
        array : List[str]
            Список с текстом, который нужно преобразовать.
            Например,
            ['Hello! My nam3 is Harry :)', 'Понятно, а я Рон.'].

        transformation: function | None
            Функция для преобразования, либо None если не нужно применять
            преобразование.

        Returns
        -------
        array: List[str]
            Список с применёнными преобразованиями текста.
        """
        if transformation:
            array = list(map(transformation, array))
        return array

    def _sparse_input(
            self,
            arg: Union[bool, Callable[[str], str]],
            func: Callable[[str], str],
    ) -> Union[None, Callable[[str], str]]:
        """Преобразуем полученные от пользователя аргументы
        в нужный вил.

        Параметры
        ----------
        arg : bool | function
            Пользовательский аргумент:
                - если True - возвращается функция в переменной func;
                - если False - возвращает None (не применяет преобразование);
                - если Callable - возвращается функция в переменной arg.

        func: function | None
            Функция для преобразования, либо None если не нужно применять
            преобразование.

        Returns
        -------
        return: None | function
            Функция для преобразования, либо ничего.
        """
        if isinstance(arg, bool):
            if arg:
                return func
            else:
                return None
        else:
            return arg

    @abstractmethod
    def _base_transform(self, array: List[str]) -> List[str]:
        """Базовые преобразования, подходящие для любого языка.

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

    @abstractmethod
    def _custom_transform(self, array: List[str]) -> List[str]:
        """Новые преобразования. Создан для переопределения в
        будущих классах.

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
        """Применение всех преобразований к массиву.

        Параметры
        ----------
        array : Iterable[str]
            Массив с текстом, который нужно преобразовать.
            Например,
            ['Hello! My nam3 is Harry :)', 'Понятно, а я Рон.'].

        Returns
        -------
        array: List[str]
            Список с применёнными преобразованиями текста.
        """
        array = list(array)

        array = self._base_transform(array)
        array = self._custom_transform(array)
        return array


class BaseAugmentation(BaseTransformation):
    """Абстрактный класс для аугментации текста."""

    def __init__(self, seed: int = 0):
        self._start_seed = seed
        self._seed = seed

    def _sparse_input(
            self,
            arg: Union[bool, dict],
            func: Callable,
    ) -> Union[None, WrapperTransform]:
        """"""
        self._seed += 1
        if isinstance(arg, bool):
            if arg:
                return WrapperTransform(func, seed=self._seed)
            else:
                return None
        else:
            return WrapperTransform(func, seed=self._seed, **arg)

    def _one_transform(
            self,
            array: List[str],
            transformation: Union[None, WrapperTransform],
    ) -> List[str]:
        """"""
        if transformation:
            lst = []
            for text in array:
                text = transformation.transform(text)
                lst.append(text)
                transformation.up_seed()
            array = lst
            transformation.reset_seed()
        return array

    @abstractmethod
    def _base_transform(self, array: List[str]) -> List[str]:
        pass

    @abstractmethod
    def _custom_transform(self, array: List[str]) -> List[str]:
        pass


class BaseDataset(Dataset):
    """Абстрактный класс для обёртки датасетов.

    Примечение
    -------
    _dict_callable_datasets : dict[str, Callable]
            Словарь, значением является функция загрузка датасета,
            а ключом является название функции для загрузка датасета.

            Например,
            {'load_city_russia': load_city_russia, }.

            Данный словарь пополняется автоматически, собирая
            все функции из `dataset._dataframes`
    """

    _dict_callable_datasets = {
        metric_name: metric_func \
            for metric_name, metric_func in getmembers(_dataframes, isfunction)
            if metric_name[0] != '_'
    }

    def __init__(
            self,
            array: Union[str, Iterable[str]],
    ):
        if isinstance(array, str):
            self._array = self._dict_callable_datasets[array]().target.values
        else:
            self._array = array

    def __len__(self):
        return len(self._array)

    @abstractmethod
    def __getitem__(self, idx):
        """Получение элемента по индексу"""


class BaseModel(ABC):
    """Абстрактный класс для моделей эмбеддингов."""

    def load(self, filename: str) -> object:
        """"""
        filename = PATH_SAVE_MODEL + '\\' + filename

        if '.pkl' not in filename:
            filename += '.pkl'

        with open(filename, 'rb') as f:
            self = pickle.load(f)
        return self

    def save(self, filename: str) -> object:
        """"""
        filename = PATH_SAVE_MODEL + '\\' + filename

        if '.pkl' not in filename:
            filename += '.pkl'

        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        return self

    @abstractmethod
    def fit(self, array: Union[BaseDataset, Iterable[str]]) -> object:
        """"""

    @abstractmethod
    def transform(self, array: Iterable[str]) -> np.ndarray:
        """"""

    def fit_transform(self, array: Iterable[str]) -> np.ndarray:
        self.fit(array)
        return self.transform(array)


class BaseSearch(ABC):
    """Абстрактный класс для получения результатов
    поиска наиболее схожего текста из БД."""

    @abstractmethod
    def search(self, text: str, k: int) -> pd.DataFrame:
        """"""


class BaseEmbeddingSearch(BaseSearch):
    """Абстрактный класс для получения результатов
    поиска наиболее схожего текста из БД на основе эмбеддингов.

    Примечение
    -------
    _dict_callable_metrics : dict[str, Callable]
            Словарь, значением является функция подсчёта метрики,
            а ключом является название функции для подсчёта метрики.

            Например,
            {'cosine_distance': cosine_distance, }.

            Данный словарь пополняется автоматически, собирая
            все функции из utils._metrics
    """
    _dict_callable_metrics = {
        metric_name: metric_func \
            for metric_name, metric_func in getmembers(_metrics, isfunction)
    }

    def __init__(
            self,
            model: BaseModel,
            embedding_database,
            original_array: Iterable[str],
            metric: Union[str, Callable],
    ):
        self._model = model
        self._embedding_database = embedding_database
        self._original_array = original_array
        self._metric = self._dict_callable_metrics.get(metric, metric) \
            if isinstance(metric, str) else metric

    @abstractmethod
    def search(self, text: str, k: int) -> pd.DataFrame:
        """"""
