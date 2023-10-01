"""
Модели для создания эмбеддингов
"""


from typing import Iterable, List
import pickle

from base import BaseModel

from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfWrapperModel(BaseModel):
    """"""

    def __init__(self, model: TfidfVectorizer):
        self._model = model

    def load_model(self, filename: str) -> object:

        if '.pkl' not in filename:
            filename += '.pkl'

        with open(filename, 'wb') as f:
            self._model = pickle.load(f)
        return self

    def save_model(self, filename: str) -> object:

        if '.pkl' not in filename:
            filename += '.pkl'

        with open(filename, 'wb') as f:
            pickle.dump(self._model, f)
        return self

    def fit(self, array: Iterable[str]) -> object:
        self._model.fit(array)
        return self

    def transform(self, array: Iterable[str]) -> List[str]:
        return self._model.transform(array)


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
