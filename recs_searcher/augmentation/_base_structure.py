"""
Алгоритмы по изменению структуры текста.
"""


from typing import Optional, List
import random
from ..base import BaseTransformation


class Abbreviation(BaseTransformation):
    """"""

    def __init__(
            self,
            sep: str = ' ',
            seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        self._sep = sep

    def _transform(self, array: List[str]) -> List[str]:
        """Берутся первые символы слов и превратить
        в одно слово. Нужно для сокращений НАИМЕНОВАНИЙ.
        Например, `Harry Potter` -> `HP`. """
        transformed_array = []
        for text in array:
            text = text.split(self._sep)
            changed_text = [word[0] for word in text if word]
            changed_text = ''.join(changed_text).upper()
            transformed_array.append(changed_text)
        return transformed_array


class DeleteWords(BaseTransformation):
    """"""

    def __init__(
            self,
            p: float = 0.1,
            delete_only_alpha: bool = True,
            sep: str = ' ',
            seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        self._p = p
        self._delete_only_alpha = delete_only_alpha
        self._sep = sep

    def _transform(self, array: List[str]) -> List[str]:
        """"""
        transformed_array = []
        for text in array:
            
            original_text = text.split(self._sep)
            changed_text = []
            for word in original_text:
                if word.isalpha() or not self._delete_only_alpha:
                    if random.randrange(100) < self._p * 100 and len(original_text) != 1:
                        continue
                changed_text.append(word)

            changed_text = ' '.join(changed_text)
            transformed_array.append(changed_text)
        return transformed_array


class ShuffleWords(BaseTransformation):
    """"""

    def __init__(
            self,
            sep: str = ' ',
            seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        self._sep = sep

    def _transform(self, array: List[str]) -> List[str]:
        """"""
        transformed_array = []
        for text in array:

            changed_text = text.split(self._sep)
            changed_text = random.sample(changed_text, len(changed_text))
            changed_text = ' '.join(changed_text)

            transformed_array.append(changed_text)
        return transformed_array
