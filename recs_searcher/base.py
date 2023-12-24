"""
Базовые классы.
"""


from abc import ABC, abstractmethod
from pathlib import Path
import pickle
from typing import Union, Iterable, List, Optional
import random

import pandas as pd
import numpy as np

from torch.utils.data import Dataset


class BaseTransformation(ABC):
    """Абстрактный класс для трансформаторов текста."""

    def __init__(self, seed: Optional[int] = None):
        self._seed = seed

    @abstractmethod
    def _transform(self, array: List[str]) -> List[str]:
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

        random.seed(self._seed)
        array = self._transform(array)
        return array


class BaseDataset(Dataset):
    """Абстрактный класс для обёртки датасетов.
    Нужен для обучения нейронных сетей.
    """

    def __init__(
            self,
            array: Iterable[str],
    ):
        self._array = array

    def __len__(self):
        return len(self._array)

    @abstractmethod
    def __getitem__(self, idx):
        """Получение элемента по индексу"""

    def append(self, text: str) -> Optional[bool]:
        """"""
        if isinstance(self._array, list):
            self._array.append(text)
        elif isinstance(self._array, np.ndarray):
            np.append(self._array, text)
        else:
            return False


class BaseModel(ABC):
    """Абстрактный класс для моделей эмбеддингов."""

    def load(self, path_folder_save: str, filename: str) -> object:
        """"""
        if '.pkl' not in filename:
            filename += '.pkl'
        path = Path(path_folder_save) / filename

        with open(path, 'rb') as f:
            self = pickle.load(f)
        return self

    def save(self, path_folder_save: str, filename: str) -> object:
        """"""
        path_folder_save = Path(path_folder_save)
        if not path_folder_save.exists():
            path_folder_save.mkdir()

        if '.pkl' not in filename:
            filename += '.pkl'
        path = path_folder_save / filename

        with open(path, 'wb') as f:
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

    def __init__(
            self,
            original_array: Iterable[str],
            preprocessing: List[BaseTransformation],
    ):
        self._original_array = original_array
        self._preprocessing = preprocessing

    @abstractmethod
    def search(self, text: str, k: int) -> pd.DataFrame:
        """"""


class BaseEmbeddingSearch(BaseSearch):
    """Абстрактный класс для получения результатов
    поиска наиболее схожего текста из БД на основе эмбеддингов.
    """

    def __init__(
            self,
            model: BaseModel,
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
        """"""
