"""
Алгоритмы для нормализации текста.
"""

from typing import Literal, Optional, List, Union, Callable
from ..base import BaseTransformation
import spacy


class LemmatizeSpacy(BaseTransformation):
    """"""

    def __init__(self, spacy_model_name: str):
        super().__init__()

        try:
            self._spacy_model = spacy.load(spacy_model_name)
        except OSError:
            spacy.cli.download(spacy_model_name)
            self._spacy_model = spacy.load(spacy_model_name)

    def _transform(self, array: List[str]) -> List[str]:
        """"""
        transformed_array = []
        for text in array:
            doc = self._spacy_model(text)
            transformed_text = ' '.join([token.lemma_ if token.lemma_ else token for token in doc])
            transformed_array.append(transformed_text)
        return transformed_array


class RemoveStopwordsSpacy(BaseTransformation):
    """"""

    def __init__(self, spacy_model_name: str):
        super().__init__()

        try:
            self._spacy_model = spacy.load(spacy_model_name)
        except OSError:
            spacy.cli.download(spacy_model_name)
            self._spacy_model = spacy.load(spacy_model_name)

    def _transform(self, array: List[str]) -> List[str]:
        """"""
        transformed_array = []
        for text in array:
            clear_tokens = []
            doc = self._spacy_model(text)
            for token in doc:
                if not token.is_stop:
                    clear_tokens.append(token.text)

            # Не возвращаю пустую строку.
            if not clear_tokens:
                transformed_array.append(text)
            else:
                transformed_text = ' '.join(clear_tokens)
                transformed_array.append(transformed_text)
        return transformed_array
