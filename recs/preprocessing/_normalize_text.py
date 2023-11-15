"""
Алгоритмы для нормализации текста.
"""


from typing import Literal

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


def _create_dict_stopwords(language: Literal['russian', 'english']) -> dict:
    """"""
    stopwords_lst = stopwords.words(language)

    if language == 'russian':
        stopwords_lst.remove('два')
        stopwords_lst.remove('три')

    stopwords_dict = {}
    for word in stopwords_lst:
        if len(word) in stopwords_dict:
            stopwords_dict[len(word)].add(word)
        else:
            stopwords_dict[len(word)] = {word}
    return stopwords_dict
