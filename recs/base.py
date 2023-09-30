"""
Базовые классы.
"""


from abc import ABC, abstractmethod
from typing import Union, Callable, Iterable, List

from utils._clear_text import (
    text_lower,
    remove_punct,
    remove_number,
    remove_whitespace,
    remove_html_tag,
    remove_url,
    remove_emoji,
)


class BaseCleaner(ABC):
    """Абстрактный класс для чистики текста.

    Примечания
    -----
    Все классы должны иметь указывать все параметры,
    которые можно установить внутри `__init__` (кроме
    `*args` и `**kwargs`).
    """

    def __init__(
            self,

            # Очистка для любого языка:
            text_lower: Union[bool, Callable[[str], str]] = True,
            remove_punct: Union[bool, Callable[[str], str]] = True,
            remove_number: Union[bool, Callable[[str], str]] = True,
            remove_whitespace: Union[bool, Callable[[str], str]] = True,
            remove_html_tag: Union[bool, Callable[[str], str]] = True,
            remove_url: Union[bool, Callable[[str], str]] = True,
            remove_emoji: Union[bool, Callable[[str], str]] = True,
    ):
        self._text_lower = text_lower
        self._remove_punct = remove_punct
        self._remove_number = remove_number
        self._remove_whitespace = remove_whitespace
        self._remove_html_tag = remove_html_tag
        self._remove_url = remove_url
        self._remove_emoji = remove_emoji

    def _one_transform(
            self,
            array: List[str],
            transformation: Union[bool, Callable[[str], str]],
            transformation_func: Callable[[str], str] = None,
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

        transformation_func: function, default=None
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

    def _base_transform(self, array: List[str]) -> List[str]:
        """Базовые преобразования, подходящие для любого языка.

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

        transformation_func: function, default=None
            - Заготовленная функция разработчиком, которая
            применяется при `transformation`=True.

        Returns
        -------
        array: List[str]
            Массив с применёнными преобразованиями текста.
        """

        array = self._one_transform(array, self._text_lower, text_lower)
        array = self._one_transform(array, self._remove_punct, remove_punct)
        array = self._one_transform(array, self._remove_number, remove_number)
        array = self._one_transform(array, self._remove_whitespace, remove_whitespace)
        array = self._one_transform(array, self._remove_html_tag, remove_html_tag)
        array = self._one_transform(array, self._remove_url, remove_url)
        array = self._one_transform(array, self._remove_emoji, remove_emoji)
        return array

    @abstractmethod
    def _custom_transform(self, array: List[str]) -> List[str]:
        """Преобразования для будущих подклассов с
        кастомными преобразованиями для разных языков.

        Параметры
        ----------
        array : List[str]
            Массив с текстом для преобразования.
            Например,
            ['Hello! My nam3 is Harry :)', 'Понятно, а я Рон.'].

        Returns
        -------
        array: List[str]
            Массив с применёнными преобразованиями текста.
        """
        pass

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

        array = list(array)

        array = self._base_transform(array)
        array = self._custom_transform(array)
        return array


if __name__ == '__main__':
    from preprocessing._base import (
        GeneralCleaner
    )

    lst = ['hellO! Man.', 'How are y0u??', 'Паренёк, ты ч3го?']

    cleaner = GeneralCleaner()

    a = cleaner.transform(lst)
    print(a)
