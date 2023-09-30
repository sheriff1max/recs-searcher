"""
Базовые классы.
"""


from abc import ABC, abstractmethod
from typing import Union, Callable
import re
import string


class BaseCleaner(ABC):
    """Абстрактный базовый класс очистики текста для разных языков.
    
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

            # Очистка под конкретный язык:
            remove_stop_word: Union[bool, Callable[[str], str]] = True,
            change_form: Union[bool, Callable[[str], str]] = True,
            correction_spelling: Union[bool, Callable[[str], str]] = True,
            replace_abbreviation: Union[bool, Callable[[str], str]] = True
    ):
        self.text_lower = text_lower
        self.remove_punct = remove_punct
        self.remove_number = remove_number
        self.remove_whitespace = remove_whitespace
        self.remove_html_tag = remove_html_tag
        self.remove_url = remove_url
        self.remove_emoji = remove_emoji

        self.remove_stop_word = remove_stop_word
        self.change_form = change_form
        self.correction_spelling = correction_spelling
        self.replace_abbreviation = replace_abbreviation

    def _base_transform(self, text: str) -> str:
        """Базовые преобразования."""
        

    @abstractmethod
    def transform(self, text: str) -> str:
        pass


if __name__ == '__main__':
    a = BaseCleaner()
    print(a)
