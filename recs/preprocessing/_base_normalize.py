"""
Алгоритмы для нормализации текста.
"""

from typing import Literal, Optional, List, Union, Callable

from base import BaseTransformation
from preprocessing._normalize_text import (
    _create_dict_stopwords,
)

from nltk.tokenize import word_tokenize
import spacy

from words2numsrus import NumberExtractor
import wordtodigits


class BaseNormalizer(BaseTransformation):
    """"""

    def __init__(
            self,
            language: Literal['russian', 'english'],
            spacy_model_name: Optional[str]=None,

            remove_stopwords: Union[bool, Callable[[str], str]] = True,
            number_extract: Union[bool, Callable[[str], str]] = True,
            lemmatize: Union[bool, Callable[[str], str]] = True,
    ):
        if language == 'russian':
            _number_extractor = NumberExtractor().replace_groups
        elif language == 'english':
            _number_extractor = wordtodigits.convert
        self._number_extractor = self._sparse_input(number_extract, _number_extractor)

        self._remove_stopwords = self._sparse_input(remove_stopwords, self.remove_stopwords)
        if self._remove_stopwords == self.remove_stopwords:
            self._dict_stopwords = _create_dict_stopwords(language)

        self._lemmatize = self._sparse_input(lemmatize, self.lemmatize)
        if self._lemmatize == self.lemmatize:
            if not spacy_model_name:
                if language == 'russian':
                    spacy_model_name = 'ru_core_news_md'
                elif language == 'english':
                    spacy_model_name = 'en_core_web_md'

            try:
                self.spacy_model = spacy.load(spacy_model_name)
            except OSError:
                print('Write next python command:')
                print(f'python -m spacy download {spacy_model_name}')
                raise OSError()

    def remove_stopwords(self, text: str) -> str:
        """"""
        dict_stopwords = self._dict_stopwords

        clear_tokens = []
        for token in word_tokenize(text):
            len_token = len(token)
            if token and not (len_token in dict_stopwords and token in dict_stopwords[len_token]):
                clear_tokens.append(token)

        # Не возвращаю пустую строку.
        if not clear_tokens:
            return text
        return ' '.join(clear_tokens)

    def lemmatize(self, text: str) -> str:
        """"""
        doc = self.spacy_model(text)
        return ' '.join([token.lemma_ if token.lemma_ else token for token in doc])

    def _base_transform(self, array: List[str]) -> List[str]:
        """Базовые преобразования по умолчанию.

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
        array = self._one_transform(array, self._remove_stopwords)
        array = self._one_transform(array, self._number_extractor)
        array = self._one_transform(array, self._lemmatize)
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
