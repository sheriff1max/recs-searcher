"""
Класс для создания ошибок в тексте.
Нужно для тестирования и обучения.
"""

from typing import Iterable, Union, Callable, List, Dict, Optional

from base import BaseTransformation

from augmentation._misspelling import (
    _add_syms,
    _change_syms,
    _delete_syms,
    _multiply_syms,
    _swap_syms,
)
from augmentation._structure import (
    _delete_words,
    _get_abbreviation,
    _shuffle_words,
)


class MisspellingAugmentation(BaseTransformation):
    """Класс создания ошибок в словах предложения.
    
    Примечание
    ----------
    Каждый параметр представляет из себя либо bool, либо dict:
    - bool:
        - True: используется преобразование с базовыми параметрами.
        - False: не используется преобразование.
    - dict: используется преобразование с данными параметрами, определёнными
    в словаре.
    """

    def __init__(
            self,
            add_syms: Union[bool, dict] = True,
            change_syms: Union[bool, dict] = True,
            delete_syms: Union[bool, dict] = True,
            multiply_syms: Union[bool, dict] = True,
            swap_syms: Union[bool, dict] = True,
            seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        self._add_syms = self._sparse_input(add_syms, _add_syms)
        self._change_syms = self._sparse_input(change_syms, _change_syms)
        self._delete_syms = self._sparse_input(delete_syms, _delete_syms)
        self._multiply_syms = self._sparse_input(multiply_syms, _multiply_syms)
        self._swap_syms = self._sparse_input(swap_syms, _swap_syms)

    def _base_transform(self, array: List[str]) -> List[str]:
        """"""
        array = self._one_transform(array, self._add_syms)
        array = self._one_transform(array, self._change_syms)
        array = self._one_transform(array, self._delete_syms)
        array = self._one_transform(array, self._multiply_syms)
        array = self._one_transform(array, self._swap_syms)
        return array

    def _custom_transform(self, array: List[str]) -> List[str]:
        """"""
        return array


class StructureAugmentation(BaseTransformation):
    """Класс создания ошибок в словах предложения.
    
    Примечание
    ----------
    Каждый параметр представляет из себя либо bool, либо dict:
    - bool:
        - True: используется преобразование с базовыми параметрами.
        - False: не используется преобразование.
    - dict: используется преобразование с данными параметрами, определёнными
    в словаре.
    """

    def __init__(
            self,
            delete_words: Union[bool, dict] = False,
            get_abbreviation: Union[bool, dict] = False,
            shuffle_words: Union[bool, dict] = True,
            seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        self._delete_words = self._sparse_input(delete_words, _delete_words)
        self._get_abbreviation = self._sparse_input(get_abbreviation, _get_abbreviation)
        self._shuffle_words = self._sparse_input(shuffle_words, _shuffle_words)

    def _base_transform(self, array: List[str]) -> List[str]:
        """"""
        array = self._one_transform(array, self._delete_words)
        array = self._one_transform(array, self._get_abbreviation)
        array = self._one_transform(array, self._shuffle_words)
        return array

    def _custom_transform(self, array: List[str]) -> List[str]:
        """"""
        return array
