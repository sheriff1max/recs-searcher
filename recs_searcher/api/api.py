""" api.py
Взаимодействие с системой."""

from typing import List, Iterable, Dict, Type, Optional, Union, Tuple, Literal

from ..base import(
    BaseTransformation,
    BaseEmbedding,
    BaseSearch,
    BaseExplain,
)
from ..similarity_search import *
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

        # Название поисковика.
        self.__type_searcher = searcher.__name__  # TODO: костыль. Фиксит краш при сохранении ChromaDBSearcher
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
        searcher: str,
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
        searcher : str
            Название поискового алгоритма.
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
        searcher = eval(searcher)
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

    def load(self, path_to_filename: str) -> 'Pipeline':
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

        # Удаляем объект поискового алгоритма для уменьшения объёма файла.
        self._searcher = None

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
        n_grams: Optional[Union[Tuple[int, int], int]] = 1,
        analyzer: Optional[Literal['word', 'char']] = 'word',
        sep: Optional[str] = ' ',
        k: Optional[int] = 10,
        ascending: Optional[bool] = True,
        **explainer_args
    ) -> Tuple[pd.DataFrame, List[Tuple[int, int]]]:
        """
        Поиск наиболее схожих N-грамм из `compared_text` в `original_text`.

        Параметры
        ----------
        compared_text : str
            Пользовательский текст, в котором нужно найти n-граммы,
            похожие на original_text.
        original_text : str
            Текст, с которым сравнивается compared_text.
        n_grams : Optional[Union[Tuple[int, int], int]]
            Длины N-грамм, которые будут оцениваться.
            Может приниматься либо одно число, либо список чисел.
        analyzer: Optional[Literal['word', 'char']]
            Считать схожесть текстов на основе N-грамм слов или символов.
        sep: Optional[str]
            Разделитель слов.
        k : Optional[int]
            Кол-во выдаваемых результатов.
        ascending : Optional[bool]
            Флаг сортировки полученных результатов.
            False - убывающая, True - возрастающая сортировка.

        Returns
        -------
        df: pd.DataFrame
            Датафрейм с результатами.
            df.columns = ['text', 'similarity']
        indeces_n_grams: List[Tuple[int, int]]
            Список кортежей индексов старта и конца самых важных N-грамм из `df`.
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
            analyzer=analyzer,
            sep=sep,
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
        searcher: Optional[Type[BaseSearch]] = None,
        **searcher_args,
    ) -> None:
        """
        Изменение или пересоздание поискового алгоритма.

        Параметры
        ----------
        searcher : Optional[Type[BaseSearch]]
            Тип поискового алгоритма.
        searcher_args : Optional[dict]
            Словарь аргументов для поискового алгоритма.

        Returns
        -------
        None
        """
        if searcher:
            self.__type_searcher = searcher.__name__
        if searcher_args:
            self.__searcher_args = searcher_args

        self._searcher = self.__create_searcher(
            searcher=self.__type_searcher,
            model=self._model,
            embedding_database=self._embedding_database,
            original_array=self._original_dataset,
            preprocessing=self._preprocessing,
            clear_array=self._clear_dataset,
            searcher_args=self.__searcher_args,
        )

    def change_explainer(
        self,
        explainer: Type[BaseExplain],
    ) -> None:
        """
        Изменение алгоритма для интерпретации результатов.

        Параметры
        ----------
        explainer : Type[BaseExplain]
            Тип алгоритма для интерпретации результатов.

        Returns
        -------
        None
        """
        self.__type_explainer = explainer


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
    pipeline.change_searcher()
    return pipeline
