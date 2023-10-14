"""
Алгоритмы для чистки текста.
"""


from typing import Iterable, Union, Callable, List

from base import BaseTransformation

from utils._clear_text import (
    _text_lower,
    _remove_punct,
    _remove_number,
    _remove_whitespace,
    _remove_html_tag,
    _remove_url,
    _remove_emoji,
)


class BaseCleaner(BaseTransformation):
    """Класс очистики текста для разных языков."""

    def __init__(
            self,
            text_lower: Union[bool, Callable[[str], str]] = True,
            remove_punct: Union[bool, Callable[[str], str]] = True,
            remove_number: Union[bool, Callable[[str], str]] = True,
            remove_whitespace: Union[bool, Callable[[str], str]] = True,
            remove_html_tag: Union[bool, Callable[[str], str]] = True,
            remove_url: Union[bool, Callable[[str], str]] = True,
            remove_emoji: Union[bool, Callable[[str], str]] = True,
    ):
        self._text_lower = self._sparse_input(text_lower, _text_lower)
        self._remove_punct = self._sparse_input(remove_punct, _remove_punct)
        self._remove_number = self._sparse_input(remove_number, _remove_number)
        self._remove_whitespace = self._sparse_input(remove_whitespace, _remove_whitespace)
        self._remove_html_tag = self._sparse_input(remove_html_tag, _remove_html_tag)
        self._remove_url = self._sparse_input(remove_url, _remove_url)
        self._remove_emoji = self._sparse_input(remove_emoji, _remove_emoji)

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
        array = self._one_transform(array, self._text_lower)
        array = self._one_transform(array, self._remove_punct)
        array = self._one_transform(array, self._remove_number)
        array = self._one_transform(array, self._remove_whitespace)
        array = self._one_transform(array, self._remove_html_tag)
        array = self._one_transform(array, self._remove_url)
        array = self._one_transform(array, self._remove_emoji)
        return array

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
        return array

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
