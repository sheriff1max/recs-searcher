""" api.py
Взаимодействие с системой."""

from typing import List, Iterable, Dict, Type, Optional

from ..base import(
    BaseTransformation,
    BaseEmbedding,
    BaseSearch,
)
from ..similarity_search import Validate

import numpy as np
import pandas as pd
from pathlib import Path
import pickle

import pathlib
import platform
if platform.system() == 'Linux':
    pathlib.WindowsPath = pathlib.PosixPath


class Pipeline:
    """Класс полного цикла """

    def __init__(
        self,
        dataset: Iterable[str],
        searcher: Type[BaseSearch],
        preprocessing: Optional[List[BaseTransformation]] = None,
        model: Optional[BaseEmbedding] = None,
        verbose: bool = True,
        **searcher_args,
    ):
        self._original_dataset = np.array(dataset)
        self._preprocessing = preprocessing

        self.__verbose('Data preparation for training has begun...', verbose)
        self._clear_dataset = self.__clear_dataset(self._original_dataset, self._preprocessing)

        self._model = model
        self._embedding_database = None
        # Если True - обучаем модель для создания эмбеддингов.
        if isinstance(model, BaseEmbedding):
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
            clear_array=self._clear_dataset,
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
        model: BaseEmbedding,
        dataset: Iterable[str],
    ) -> np.ndarray:
        """"""
        embedding_database = model.fit_transform(dataset)
        return embedding_database

    def __create_searcher(
        self,
        searcher: Type[BaseSearch],
        model: Optional[BaseEmbedding],
        embedding_database: np.ndarray,
        original_array: Iterable[str],
        preprocessing: List[BaseTransformation],
        clear_array: Iterable[str],
        searcher_args: dict,
    ) -> BaseSearch:
        """"""
        if isinstance(model, BaseEmbedding):
            return searcher(
                model=model,
                embedding_database=embedding_database,
                original_array=original_array,
                preprocessing=preprocessing,
                clear_array=clear_array,
                **searcher_args,
            )
        else:
            return searcher(
                original_array=self._original_dataset,
                preprocessing=preprocessing,
                clear_array=clear_array,
            )

    def get_model(self) -> Optional[BaseEmbedding]:
        return self._model

    def get_preprocessing(self) -> Optional[List[BaseTransformation]]:
        return self._preprocessing

    def load(self, path_to_filename: str) -> object:
        """"""
        self = load_pipeline(path_to_filename)
        return self

    def save(self, path_folder_save: str, filename: str) -> object:
        """"""
        path_folder_save = Path(path_folder_save)
        if not path_folder_save.exists():
            path_folder_save.mkdir()

        if '.pkl' not in filename:
            filename += '.pkl'
        path = path_folder_save / filename

        with open(path, 'wb') as f:
            pickle.dump(self, f)
        return self

    def validate(
        self,
        augmentation_transforms: List[BaseTransformation],
        accuracy_top: List[int] = [1, 5, 10],
        ascending: bool = True,
    ) -> Dict[int, float]:
        """"""
        score_metrics = Validate(
            self._searcher,
            augmentation_transforms,
            accuracy_top,
            ascending=ascending,
        )
        self.score_metrics = score_metrics
        return score_metrics

    def search(self, text: str, k: int, ascending: bool = True) -> pd.DataFrame:
        """"""
        return self._searcher.search(text, k, ascending=ascending)

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
            **self.__searcher_args,
        )
    
    def change_searcher(
        self,
        searcher: Type[BaseSearch],
        **searcher_args,
    ) -> None:
        """"""
        self.__type_searcher = searcher
        self.__searcher_args = searcher_args

        self._searcher = self.__create_searcher(
            searcher=self.__type_searcher,
            model=self._model,
            embedding_database=self._embedding_database,
            original_array=self._original_dataset,
            preprocessing=self._preprocessing,
            clear_array=self._clear_dataset,
            searcher_args=searcher_args,
        )


def load_pipeline(path_to_filename: str) -> Pipeline:
    """"""
    if '.pkl' not in path_to_filename:
        path_to_filename += '.pkl'

    path_to_filename = Path(path_to_filename)
    with open(path_to_filename, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline
