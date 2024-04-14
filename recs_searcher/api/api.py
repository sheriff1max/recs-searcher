""" api.py
Взаимодействие с системой."""

from typing import List, Iterable, Dict, Type, Optional, Union, Tuple

from ..base import(
    BaseTransformation,
    BaseEmbedding,
    BaseSearch,
    BaseExplain,
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
    """API для взаимодействия с алгоритмами.
    С помощью данного класса можно:
        1. обучить выбранную модель на пользовательском датасете;
        2. сохранять и загружать собранный и обученный pipeline для
            последующего использования;
        3. производить вычисления для пользовательского ввода;
        4. проверить точность работы обученного pipeline;
        5. получить интерпретацию полученных результатов.

    Параметры
        ----------
        dataset : Iterable[str]
            Датасет, на котором будут обучаться алгоритмы.
        searcher : Type[BaseSearch]
            Класс, на основе которого будут искаться схожие текста.
        preprocessing : Optional[List[BaseTransformation]]
            Список предобработки текстовых данных.
        model : Optional[BaseEmbedding]
            Модель для создания эмбеддингов.
        explainer : Optional[Type[BaseExplain]]
            Алгоритм для интерпретации результатов и объяснения
            схожести двух текстов.
        verbose : Optional[bool]
            Вывод этапов Pipeline.
        searcher_args : Параметры для настройки `searcher` типа `BaseSearch`.
    """

    def __init__(
        self,
        dataset: Iterable[str],
        searcher: Type[BaseSearch],
        preprocessing: Optional[List[BaseTransformation]] = None,
        model: Optional[BaseEmbedding] = None,
        explainer: Optional[Type[BaseExplain]] = None,
        verbose: Optional[bool] = True,
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

        self.__type_explainer = explainer

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
        """
        Вывод инофрмации об этапе обучения pipeline.

        Параметры
        ----------
        message : str
            Выводимое сообщение.
        verbose : bool
            Выводить сообщение или нет.

        Returns
        -------
        None
        """
        if verbose:
            print(message)

    def __clear_dataset(
        self,
        dataset: Iterable[str],
        preprocessing: List[BaseTransformation],
    ) -> np.ndarray:
        """
        Предобработка текста.

        Параметры
        ----------
        dataset : Iterable[str]
            Необработанный датасет текстов.
        preprocessing : List[BaseTransformation]
            Список алгоритмов для предобработка текста.

        Returns
        -------
        text: np.ndarray
            Массив обработанных текстов.
        """
        for transformation in preprocessing:
            dataset = transformation.transform(dataset)
        return np.array(dataset)

    def __fit_transform(
        self,
        model: BaseEmbedding,
        dataset: Iterable[str],
    ) -> np.ndarray:
        """
        Обучение на входном датасете и преобразование его в эмбеддинги.

        Параметры
        ----------
        model : BaseEmbedding
            Модель для обучения.
        dataset : Iterable[str]
            Датасет текстов.

        Returns
        -------
        text: np.ndarray
            Датасет в виде эмбеддингов.
        """
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
        """
        Инициализация поискового алгоритма.

        Параметры
        ----------
        searcher : Type[BaseSearch]
            Тип поискового алгоритма.
        model : BaseEmbedding
            Модель для обучения.
        embedding_database : np.ndarray
            Датасет в виде эмбеддингов.
        original_array : Iterable[str]
            Исходный датасет.
        preprocessing : List[BaseTransformation]
            Список алгоритмов для предобработки текстов.
        clear_array : Iterable[str]
            Предобработанный текст.
        searcher_args : dict
            Словарь аргументов для поискового алгоритма.

        Returns
        -------
        searcher: BaseSearch
            Инициализированный объект поискового алгоритма.
        """
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
        """
        Загрузка pipeline из файла.

        Параметры
        ----------
        path_to_filename : str
            Путь до файла.

        Returns
        -------
        self: Pipeline
            Загруженный pipeline.
        """
        self = load_pipeline(path_to_filename)
        return self

    def save(self, path_folder_save: str, filename: str) -> object:
        """
        Сохранение pipeline в файл pickle.

        Параметры
        ----------
        path_folder_save : str
            Путь до папки, куда сохранить файл.
        filename : str
            Название файла.

        Returns
        -------
        self: Pipeline
            Текущий pipeline.
        """
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
        accuracy_top: Optional[List[int]] = [1, 5, 10],
        ascending: Optional[bool] = True,
    ) -> Dict[int, float]:
        """
        Получение метрик точности обученного pipeline.

        Параметры
        ----------
        augmentation_transforms : List[BaseTransformation]
            Список алгоритмов аугментации для создания ошибок в тексте.
        accuracy_top : Optional[List[int]]
            Список для оценивания N@Accuracy.
        ascending : Optional[bool]
            Флаг сортировки полученных результатов.
            False - убывающая, True - возрастающая сортировка.

        Returns
        -------
        score_metrics: Dict[int, float]
            Посчитанные метрики.
        """
        score_metrics = Validate(
            self._searcher,
            augmentation_transforms,
            accuracy_top,
            ascending=ascending,
        )
        self.score_metrics = score_metrics
        return score_metrics

    def search(self, text: str, k: int, ascending: bool = True) -> pd.DataFrame:
        """Поиск наиболее схожих k-текстов из БД на text пользователя.

        Параметры
        ----------
        text : str
            Пользовательский текст, которому нужно найти наиболее
            схожий текст из БД.
        k : int
            Кол-во выдаваемых результатов.
        ascending : bool
            Флаг сортировки полученных результатов.
            False - убывающая, True - возрастающая сортировка.

        Returns
        -------
        df: pd.DataFrame
            Датафрейм с результатами.
            df.columns = ['text', 'similarity']
        """
        return self._searcher.search(text, k, ascending=ascending)

    def explain(
        self,
        compared_text: str,
        original_text: str,
        n_grams: Union[Tuple[int, int], int] = 1,
        k: int = 10,
        ascending: bool = True,
        **explainer_args
    ) -> pd.DataFrame:
        """Поиск наиболее схожих k-текстов из БД на text пользователя.

        Параметры
        ----------
        text : str
            Пользовательский текст, которому нужно найти наиболее
            схожий текст из БД.
        k : int
            Кол-во выдаваемых результатов.
        ascending : bool
            Флаг сортировки полученных результатов.
            False - убывающая, True - возрастающая сортировка.

        Returns
        -------
        df: pd.DataFrame
            Датафрейм с результатами.
            df.columns = ['text', 'similarity']
        """
        if self.__type_explainer is None:
            raise ValueError(f'The `explainer` parameter was not declared during initialization.')

        explainer = self.__type_explainer(
            model=self.get_model(),
            preprocessing=self.get_preprocessing(),
            **explainer_args,
        )
        return explainer.explain(
            compared_text=compared_text,
            original_text=original_text,
            n_grams=n_grams,
            k=k,
            ascending=ascending,
        )

    def fine_tuning(self, dataset: Iterable[str]) -> object:
        """
        Дообучение пайплайна на новых данных.

        Параметры
        ----------
        dataset : Iterable[str]
            Датасет текстов.

        Returns
        -------
        pipeline: Pipeline
            Новый объект Pipeline.
        """
        dataset = np.array(dataset)
        dataset = np.append(self._original_dataset, dataset)
        self = Pipeline(
            dataset=dataset,
            preprocessing=self._preprocessing,
            model=self._model,
            searcher=self.__type_searcher,
            verbose=True,
            **self.__searcher_args,
        )
        return self

    def change_searcher(
        self,
        searcher: Type[BaseSearch],
        **searcher_args,
    ) -> None:
        """
        Изменение поискового алгоритма.

        Параметры
        ----------
        searcher : Type[BaseSearch]
            Тип поискового алгоритма.
        searcher_args : dict
            Словарь аргументов для поискового алгоритма.

        Returns
        -------
        None
        """
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
    """
    Загрузка pipeline из файла.

    Параметры
    ----------
    path_to_filename : str
        Путь до файла.

    Returns
    -------
    self: Pipeline
        Загруженный pipeline.
    """
    if '.pkl' not in path_to_filename:
        path_to_filename += '.pkl'

    path_to_filename = Path(path_to_filename)
    with open(path_to_filename, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline
