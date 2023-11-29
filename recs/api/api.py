""" api.py
Взаимодействие с системой."""

from typing import List, Iterable, Dict, Type, Union, Optional

from base import(
    BaseDataset,
    BaseTransformation,
    BaseModel,
    BaseSearch,
    BaseEmbeddingSearch,
)
from models import Validate

import numpy as np
import pandas as pd
import os
import pickle

CUR_PATH = os.path.dirname(__file__)
PATH_SAVE_PIPELINE = os.path.join(CUR_PATH, 'pipelines')
if not os.path.exists(PATH_SAVE_PIPELINE):
    os.mkdir(PATH_SAVE_PIPELINE)


class Pipeline:
    """Класс полного цикла """

    def __init__(
            self,
            dataset: Iterable[str],
            preprocessing: List[BaseTransformation],
            searcher: Type[BaseSearch],
            model: BaseModel = None,
            verbose: bool = True,
            **searcher_args,
    ):
        self._original_dataset = np.array(dataset)
        self._preprocessing = preprocessing
        self.__verbose('Data preparation for training has begun...', verbose)
        self._clear_dataset = self.__clear_dataset(self._original_dataset, self._preprocessing)

        self._model = model
        self._embedding_database = None
        if isinstance(self._model, BaseModel):
            self.__verbose('The training of the model has begun...', verbose)
            self._embedding_database = self.__fit_transform(self._model, self._clear_dataset)

        # Класс поисковика для будущего дообучения.
        self.__type_searcher = searcher
        self.__searcher_args = searcher_args

        self._searcher = self.__create_searcher(
            searcher=self.__type_searcher,
            model=self._model,
            embedding_database=self._embedding_database,
            original_array=self._original_dataset,
            preprocessing=self._preprocessing,
            searcher_args=searcher_args,
        )
        self.__verbose('Pipeline ready!', verbose)

        # Здесь хранится метрика последнего раза валидации.
        self.score_metrics = None

    def __verbose(self, message: str, verbose: bool) -> None:
        """"""
        if verbose:
            print(message)

    def __clear_dataset(
            self,
            dataset: Iterable[str],
            preprocessing: List[BaseTransformation],
    ) -> np.ndarray:
        """"""
        for transformation in preprocessing:
            dataset = transformation.transform(dataset)
        return np.array(dataset)

    def __fit_transform(
            self,
            model: BaseModel,
            dataset: Iterable[str],
    ) -> np.ndarray:
        """"""
        embedding_database = model.fit_transform(dataset)
        return embedding_database

    def __create_searcher(
            self,
            searcher: Type[BaseSearch],
            model: BaseModel,
            embedding_database: np.ndarray,
            original_array: Iterable[str],
            preprocessing: List[BaseTransformation],
            searcher_args: dict,
    ) -> BaseSearch:
        """"""
        if isinstance(self._model, BaseModel):
            return searcher(
                model=model,
                embedding_database=embedding_database,
                original_array=original_array,
                preprocessing=preprocessing,
                **searcher_args,
            )
        else:
            return searcher(
                original_array=original_array,
                preprocessing=preprocessing,
            )

    def load(self, filename: str) -> object:
        """"""
        return load_pipeline(filename)

    def save(self, filename: str) -> object:
        """"""
        filename = PATH_SAVE_PIPELINE + '\\' + filename
        if '.pkl' not in filename:
            filename += '.pkl'

        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        return self

    def validate(
            self,
            augmentation_transforms: List[BaseTransformation],
            accuracy_top: List[int] = [1, 5, 10]
    ) -> Dict[int, float]:
        """"""
        score_metrics = Validate(
            self._searcher,
            augmentation_transforms,
            accuracy_top,
        )
        self.score_metrics = score_metrics
        return score_metrics

    def search(self, text: str, k: int) -> pd.DataFrame:
        """"""
        return self._searcher.search(text, k)

    def fine_tuning(
            self,
            dataset: Iterable[str],
    ) -> object:
        """Дообучение пайплайна на новых данных."""
        dataset = np.array(dataset)
        dataset = np.append(self._original_dataset, dataset)
        return Pipeline(
            dataset=dataset,
            preprocessing=self._preprocessing,
            model=self._model,
            searcher=self.__type_searcher,
            verbose=True,
            **self.__searcher_args
        )


def load_pipeline(filename: str) -> Pipeline:
    """"""
    filename = PATH_SAVE_PIPELINE + '\\' + filename
    if '.pkl' not in filename:
        filename += '.pkl'

    with open(filename, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline
