"""
Модели для создания эмбеддингов
"""


from typing import Iterable, List
import pickle
import os

from base import BaseModel

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


CUR_PATH = os.path.dirname(__file__)
PATH_SAVE_MODEL = CUR_PATH + '\\weights'
if not os.path.exists(PATH_SAVE_MODEL):
    os.mkdir(PATH_SAVE_MODEL)


class TfidfWrapperModel(BaseModel):
    """"""

    def __init__(self, model: TfidfVectorizer):
        self._model = model

    def load(self, filename: str) -> object:

        filename = PATH_SAVE_MODEL + '\\' + filename

        if '.pkl' not in filename:
            filename += '.pkl'

        with open(filename, 'rb') as f:
            self._model = pickle.load(f)
        return self

    def save(self, filename: str) -> object:

        filename = PATH_SAVE_MODEL + '\\' + filename

        if '.pkl' not in filename:
            filename += '.pkl'

        with open(filename, 'wb') as f:
            pickle.dump(self._model, f)
        return self

    def fit(self, array: Iterable[str]) -> object:
        self._model.fit(array)
        return self

    def transform(self, array: Iterable[str]) -> np.ndarray:
        array = self._model.transform(array).toarray()
        return array


# class FastTextWrapperModel(BaseModel):
#     """"""

#     def __init__(self):
#         pass

#     def load_model(self, filename: str) -> object:
#         return super().load_model(filename)

#     def save_model(self, filename: str) -> object:
#         return super().save_model(filename)

#     def fit(self, array: Iterable[str]) -> object:
#         return super().fit()

#     def transform(self, array: Iterable[str]) -> List[str]:
#         return super().transform()


# class BertModel(BaseModel):
#     """"""

#     def __init__(self):
#         pass

#     def load_model(self, filename: str) -> object:
#         return super().load_model(filename)

#     def save_model(self, filename: str) -> object:
#         return super().save_model(filename)

#     def fit(self, array: Iterable[str]) -> object:
#         return super().fit()

#     def transform(self, array: Iterable[str]) -> List[str]:
#         return super().transform()
