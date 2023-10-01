"""
Базовые классы.
"""


from abc import ABC, abstractmethod
from typing import Union, Callable, Iterable, List

import pandas as pd
import numpy as np


class BaseTransformation(ABC):
    """Абстрактный класс для трансформаторов текста.

    Примечания
    -----
    Все классы должны иметь указывать все параметры,
    которые можно установить внутри `__init__` (кроме
    `*args` и `**kwargs`).
    """

    def _one_transform(
            self,
            array: List[str],
            transformation: Union[bool, Callable[[str], str]],
            transformation_func: Union[None, Callable[[str], str]] = None,
    ) -> List[str]:
        """Применение преобразований к массиву текста.

        Параметры
        ----------
        array : List[str]
            Массив с текстом для преобразования.
            Например,
            ['Hello! My nam3 is Harry :)', 'Понятно, а я Рон.'].

        transformation: bool or function
            - Если True, то применяется заготовленная
            функция `transformation_func`.
            - Если подан function, то применяется
            функция `transformation`. Можно передавать
            по api.

        transformation_func: function or None, default=None
            - Заготовленная функция разработчиком, которая
            применяется при `transformation`=True.

        Returns
        -------
        array: List[str]
            Массив с применёнными преобразованиями текста.
        """

        if transformation:

            if isinstance(transformation, bool):
                array = list(map(transformation_func, array))
            else:
                array = list(map(transformation, array))

        return array

    @abstractmethod
    def transform(self, array: Iterable[str]) -> List[str]:
        """Применение всех преобразований к массиву.

        Параметры
        ----------
        array : Iterable[str]
            Массив с текстом для преобразования.
            Например,
            ['Hello! My nam3 is Harry :)', 'Понятно, а я Рон.'].

        Returns
        -------
        array: List[str]
            Массив с применёнными преобразованиями текста.
        """


class BaseModel(ABC):
    """Абстрактный класс для моделей эмбеддингов."""

    @abstractmethod
    def load(self, filename: str) -> object:
        """"""

    @abstractmethod
    def save(self, filename: str) -> object:
        """"""

    @abstractmethod
    def fit(self, array: Iterable[str]) -> object:
        """"""

    @abstractmethod
    def transform(self, array: Iterable[str]) -> np.ndarray:
        """"""

    def fit_transform(self, array: Iterable[str]) -> np.ndarray:
        self.fit(array)
        return self.transform(array)
