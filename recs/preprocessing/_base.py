"""
Чистка данных для разных языков.
"""


from typing import Union, Callable, List

from base import BaseCleaner


class GeneralCleaner(BaseCleaner):
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
    
    def _custom_transform(self, array: List[str]) -> List[str]:
        """Т.к. данный класс применим для любого языка,
        уникальных преобразований у него нет."""
        return array


class EngCleaner(BaseCleaner):
    """Класс очистики текста для английского языка."""

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
            remove_stop_word: Union[bool, Callable[[str], str]] = False,
            change_form: Union[bool, Callable[[str], str]] = False,
            correction_spelling: Union[bool, Callable[[str], str]] = False,
            replace_abbreviation: Union[bool, Callable[[str], str]] = False
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

        self._remove_stop_word = remove_stop_word
        self._change_form = change_form
        self._correction_spelling = correction_spelling
        self._replace_abbreviation = replace_abbreviation
    
    def _custom_transform(self, array: List[str]):
        """"""
        return array
