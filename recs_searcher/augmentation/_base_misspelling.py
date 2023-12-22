"""
Алгоритмы для создания ошибок в словах.
"""


import random
import string
from typing import List, Literal, Optional
from ..base import BaseTransformation


ALPHABET = {
    'russian': ''.join(list(map(chr, range(ord('А'), ord('я')+1)))),
    'english': string.ascii_letters,
}


class ChangeSyms(BaseTransformation):
    """"""

    def __init__(
            self,
            p: float = 0.05,
            language: Literal['english', 'russian'] = 'english',
            change_only_alpha: bool = True,
            seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        self._p = p
        self._language = language
        self._change_only_alpha = change_only_alpha

    def _transform(self, array: List[str]) -> List[str]:
        """"""
        transformed_array = []
        for text in array:
            changed_text = ''
            for sym in text:

                if sym.isalpha() or not self._change_only_alpha:
                    if random.randrange(100) < self._p * 100:
                        sym = random.choice(ALPHABET[self._language])
                changed_text += sym
            transformed_array.append(changed_text)
        return transformed_array


class DeleteSyms(BaseTransformation):
    """"""

    def __init__(
            self,
            p: float = 0.05,
            delete_only_alpha: bool = True,
            seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        self._p = p
        self._delete_only_alpha = delete_only_alpha

    def _transform(self, array: List[str]) -> List[str]:
        """"""
        transformed_array = []
        for text in array:

            changed_text = ''
            for sym in text:
                if sym.isalpha() or not self._delete_only_alpha:
                    if random.randrange(100) < self._p * 100:
                        continue
                changed_text += sym
            transformed_array.append(changed_text)
        return transformed_array


class AddSyms(BaseTransformation):
    """"""

    def __init__(
            self,
            p: float = 0.05,
            language: Literal['english', 'russian'] = 'english',
            seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        self._p = p
        self._language = language

    def _transform(self, array: List[str]) -> List[str]:
        """"""
        transformed_array = []
        for text in array:

            changed_text = ''
            for sym in text:
                if random.randrange(100) < self._p * 100:
                    sym += random.choice(ALPHABET[self._language])
                changed_text += sym
            transformed_array.append(changed_text)
        return transformed_array


class MultiplySyms(BaseTransformation):
    """"""

    def __init__(
            self,
            p: float = 0.05,
            count_multiply: int = 2,
            multiply_only_alpha: bool = True,
            seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        self._p = p
        self._count_multiply = count_multiply
        self._multiply_only_alpha = multiply_only_alpha

    def _transform(self, array: List[str]) -> List[str]:
        """"""
        transformed_array = []
        for text in array:

            changed_text = ''
            for sym in text:
                if sym.isalpha() or not self._multiply_only_alpha:
                    if random.randrange(100) < self._p * 100:
                        sym = sym * self._count_multiply
                changed_text += sym
            transformed_array.append(changed_text)
        return transformed_array


class SwapSyms(BaseTransformation):
    """"""

    def __init__(
            self,
            p: float = 0.05,
            seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        self._p = p

    def _transform(self, array: List[str]) -> List[str]:
        """"""
        transformed_array = []
        for text in array:

            text = list(text)
            for i in range(0, len(text)-1):
                if random.randrange(100) < self._p * 100:
                    text[i], text[i+1] = text[i+1], text[i]
            changed_text = ''.join(text)
            transformed_array.append(changed_text)
        return transformed_array
