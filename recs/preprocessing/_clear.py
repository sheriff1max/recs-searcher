"""
Алгоритмы для чистки текста.
"""


from typing import Iterable, Union, Callable, List

from base import BaseTransformation

from utils._clear_text import (
    text_lower,
    remove_punct,
    remove_number,
    remove_whitespace,
    remove_html_tag,
    remove_url,
    remove_emoji,
)


class BaseCleaner(BaseTransformation):
    """Класс очистики текста для разных языков."""

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
        super().__init__(
            text_lower,
            remove_punct,
            remove_number,
            remove_whitespace,
            remove_html_tag,
            remove_url,
            remove_emoji,
        )
    
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

    def _custom_transform(self, array: List[str]) -> List[str]:
        """Новые преобразования, подходящие для любого языка.

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

        return array

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
