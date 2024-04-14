"""
Алгоритмы для чистки текста.
"""


from typing import List
import re
import string
from ..base import BaseTransformation
import spacy


class TextLower(BaseTransformation):
    """Алгоритм привод текст к нижнему регистру."""

    def __init__(self):
        super().__init__()

    def _transform(self, array: List[str]) -> List[str]:
        transformed_array = []
        for text in array:
            text = text.lower()
            transformed_array.append(text)
        return transformed_array


class RemovePunct(BaseTransformation):
    """Алгоритм удаляет все пунктуационные знаки из текста."""

    def __init__(self):
        super().__init__()
        self._whitespaces = ''.join([' ' for _ in range(len(string.punctuation))])

    def _transform(self, array: List[str]) -> List[str]:
        transformed_array = []
        for text in array:
            text = text.translate(str.maketrans(string.punctuation, self._whitespaces, ''))
            transformed_array.append(text)
        return transformed_array


class RemoveNumber(BaseTransformation):
    """Алгоритм удаляет все числа из текста."""

    def __init__(self):
        super().__init__()

    def _transform(self, array: List[str]) -> List[str]:
        transformed_array = []
        for text in array:
            text = re.sub(r'\d+', "", text)
            transformed_array.append(text)
        return transformed_array


class RemoveWhitespace(BaseTransformation):
    """Алгоритм удаляет все лишние пробелы в тексте."""

    def __init__(self):
        super().__init__()

    def _transform(self, array: List[str]) -> List[str]:
        transformed_array = []
        for text in array:
            text = text.replace('  ', ' ').strip()
            transformed_array.append(text)
        return transformed_array


class RemoveHTML(BaseTransformation):
    """Алгоритм удаляет всю HTML-разметку из текста."""

    def __init__(self):
        super().__init__()
        self._pattern = re.compile('<.*?>')

    def _transform(self, array: List[str]) -> List[str]:
        transformed_array = []
        for text in array:
            text = re.sub(self._pattern, '', text) 
            transformed_array.append(text)
        return transformed_array


class RemoveURL(BaseTransformation):
    """Алгоритм удаляет все ссылки из текста."""

    def __init__(self):
        super().__init__()
        self._pattern = re.compile(r'https?://\S+|www\.\S+')

    def _transform(self, array: List[str]) -> List[str]:
        transformed_array = []
        for text in array:
            text = re.sub(self._pattern, '', text) 
            transformed_array.append(text)
        return transformed_array


class RemoveEmoji(BaseTransformation):
    """Алгоритм удаляет все эмодзи из текста."""

    def __init__(self):
        super().__init__()
        self._pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # смайлики.
            u"\U0001F300-\U0001F5FF"  # символы и пиктограммы.
            u"\U0001F680-\U0001F6FF"  # транспорт и символы на карте.
            u"\U0001F1E0-\U0001F1FF"  # флаги (iOS).
                                "]+", flags=re.UNICODE)

    def _transform(self, array: List[str]) -> List[str]:
        transformed_array = []
        for text in array:
            text = re.sub(self._pattern, '', text)
            transformed_array.append(text)
        return transformed_array


class SpacyClear(BaseTransformation):
    """Сборный алгоритм предобработки текстовых данных,
    основанный на библиотеке `Spacy`."""

    def __init__(self,
        spacy_model_name: str,
        remove_punct: bool = True,
        remove_url: bool = True,
        remove_email: bool = True,
        remove_digit: bool = True,
        remove_quote: bool = True,
        remove_num: bool = True,
        remove_space: bool = True,
    ):
        super().__init__()
        self._remove_punct = remove_punct
        self._remove_url = remove_url
        self._remove_email = remove_email
        self._remove_digit = remove_digit
        self._remove_quote = remove_quote
        self._remove_num = remove_num
        self._remove_space = remove_space

        try:
            self._spacy_model = spacy.load(spacy_model_name)
        except OSError:
            print(f'Downloading {spacy_model_name}...')
            spacy.cli.download(spacy_model_name)
            self._spacy_model = spacy.load(spacy_model_name)

    def _transform(self, array: List[str]) -> List[str]:
        transformed_array = []
        for text in array:
            cleared_text = []

            doc = self._spacy_model(text)
            for token in doc:
                if self._remove_punct and token.is_punct:
                    continue
                # Удаление ссылки.
                elif self._remove_url and token.like_url:
                    continue
                # Удаление почтового адреса.
                elif self._remove_email and token.like_email:
                    continue
                # Удаление чисел.
                elif self._remove_digit and token.is_digit and self._remove_num and token.like_num:
                    continue
                # Удаление ковычки.
                elif self._remove_quote and token.is_quote:
                    continue
                # Удаление пробелов и `\n`.
                elif self._remove_space and token.is_space:
                    continue
                cleared_text.append(token.text)

            cleared_text = ' '.join(cleared_text)
        return transformed_array
