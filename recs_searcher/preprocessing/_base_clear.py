"""
Алгоритмы для чистки текста.
"""


from typing import List
import re
import string
from ..base import BaseTransformation


class TextLower(BaseTransformation):
    """"""

    def __init__(self):
        super().__init__()

    def _transform(self, array: List[str]) -> List[str]:
        """"""
        transformed_array = []
        for text in array:
            text = text.lower()
            transformed_array.append(text)
        return transformed_array


class RemovePunct(BaseTransformation):
    """"""

    def __init__(self):
        super().__init__()
        self._whitespaces = ''.join([' ' for _ in range(len(string.punctuation))])

    def _transform(self, array: List[str]) -> List[str]:
        """"""
        transformed_array = []
        for text in array:
            text = text.translate(str.maketrans(string.punctuation, self._whitespaces, ''))
            transformed_array.append(text)
        return transformed_array


class RemoveNumber(BaseTransformation):
    """"""

    def __init__(self):
        super().__init__()

    def _transform(self, array: List[str]) -> List[str]:
        """"""
        transformed_array = []
        for text in array:
            text = re.sub(r'\d+', "", text)
            transformed_array.append(text)
        return transformed_array


class RemoveWhitespace(BaseTransformation):
    """"""

    def __init__(self):
        super().__init__()

    def _transform(self, array: List[str]) -> List[str]:
        """"""
        transformed_array = []
        for text in array:
            text = text.replace('  ', ' ').strip()
            transformed_array.append(text)
        return transformed_array


class RemoveHTML(BaseTransformation):
    """"""

    def __init__(self):
        super().__init__()
        self._pattern = re.compile('<.*?>')

    def _transform(self, array: List[str]) -> List[str]:
        """"""
        transformed_array = []
        for text in array:
            text = re.sub(self._pattern, '', text) 
            transformed_array.append(text)
        return transformed_array


class RemoveURL(BaseTransformation):
    """"""

    def __init__(self):
        super().__init__()
        self._pattern = re.compile(r'https?://\S+|www\.\S+')

    def _transform(self, array: List[str]) -> List[str]:
        """"""
        transformed_array = []
        for text in array:
            text = re.sub(self._pattern, '', text) 
            transformed_array.append(text)
        return transformed_array


class RemoveEmoji(BaseTransformation):
    """"""

    def __init__(self):
        super().__init__()
        self._pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # смайлики.
            u"\U0001F300-\U0001F5FF"  # символы и пиктограммы.
            u"\U0001F680-\U0001F6FF"  # транспорт и символы на карте.
            u"\U0001F1E0-\U0001F1FF"  # флаги (iOS).
                                "]+", flags=re.UNICODE)

    def _transform(self, array: List[str]) -> List[str]:
        """"""
        transformed_array = []
        for text in array:
            text = re.sub(self._pattern, '', text)
            transformed_array.append(text)
        return transformed_array
