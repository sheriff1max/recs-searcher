"""
Базовые классы модуля.
"""


from abc import ABC, abstractmethod
from pathlib import Path
import pickle
from typing import Iterable, List, Literal, Optional, Union, Tuple
import random

import pandas as pd
import numpy as np

import pathlib
import platform
if platform.system() == 'Linux':
    pathlib.WindowsPath = pathlib.PosixPath


class BaseTransformation(ABC):
    """Абстрактный класс для трансформаторов текста."""

    def __init__(self, seed: Optional[int] = None):
        self._seed = seed

    @abstractmethod
    def _transform(self, array: List[str]) -> List[str]:
        """Преобразование, применяемое к каждому текстовому элементу списка array.

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

    def transform(self, array: Iterable[str]) -> List[str]:
        """Применение преобразования к массиву.

        Параметры
        ----------
        array : Iterable[str]
            Массив с текстом, который нужно преобразовать.

        Returns
        -------
        array: List[str]
            Список с применёнными преобразованиями текста.
        """
        array = list(array)

        random.seed(self._seed)
        np.random.seed(self._seed)
        array = self._transform(array)
        return array


class BaseEmbedding(ABC):
    """Абстрактный класс для эмбеддингов."""

    def load(self, path_folder_load: str, filename: str) -> None:
        """Загрузка vectorizer из файла.

        Параметры
        ----------
        path_folder_load : str
            Путь, где лежит файл.
        filename : str
            Название файла для загрузки.

        Returns
        -------
        None
        """
        if '.pkl' not in filename:
            filename += '.pkl'
        path = Path(path_folder_load) / filename

        with open(path, 'rb') as f:
            self = pickle.load(f)

    def save(self, path_folder_save: str, filename: str) -> None:
        """Сохранение vectorizer в файл.

        Параметры
        ----------
        path_folder_save : str
            Путь сохранения файла.
        filename : str
            Название файла для сохранения.

        Returns
        -------
        None
        """
        path_folder_save = Path(path_folder_save)
        if not path_folder_save.exists():
            path_folder_save.mkdir()

        if '.pkl' not in filename:
            filename += '.pkl'
        path = path_folder_save / filename

        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @abstractmethod
    def fit(self, array: Iterable[str]) -> object:
        """Обучение vectorizer.

        Параметры
        ----------
        array : Iterable[str]
            Массив с текстом, на котором обучается vectorizer.

        Returns
        -------
        self
        """

    @abstractmethod
    def transform(self, array: Iterable[str]) -> np.ndarray:
        """Обучение vectorizer.

        Параметры
        ----------
        array : Iterable[str]
            Массив с текстом, который нужно трансформировать в вектор чисел.

        Returns
        -------
        array: np.ndarray
            Массив с закодированными словами в числа.
        """

    def fit_transform(self, array: Iterable[str]) -> np.ndarray:
        """Обучение vectorizer и трансформация текста.
        Выполняет методы fit() и transform() текущего класса.

        Параметры
        ----------
        array : Iterable[str]
            Массив с текстом, на котором обучаемся и 
            который нужно трансформировать в вектор чисел.

        Returns
        -------
        array: np.ndarray
            Массив с закодированными словами в числа.
        """
        self.fit(array)
        return self.transform(array)


class BaseSearch(ABC):
    """Абстрактный класс для получения результатов
    поиска наиболее схожего текста из БД."""

    def __init__(
            self,
            original_array: Iterable[str],
            preprocessing: List[BaseTransformation] = [],
            clear_array: Optional[Iterable[str]] = None,
    ):
        self._original_array = original_array
        self._preprocessing = preprocessing
        if clear_array is not None:
            self._clear_text = clear_array
        else:
            self._clear_text = self._preprocessing_text(original_array, preprocessing)

    def _preprocessing_text(
        self,
        text: Union[Iterable[str], str],
        preprocessing: List[BaseTransformation],
    ) -> Union[List[str], str]:
        """
        Предобработка текста.

        Параметры
        ----------
        text : Union[Iterable[str], str]
            Необработанный текст.
        preprocessing : List[BaseTransformation]
            Список алгоритмов для предобработка текста.

        Returns
        -------
        text: Union[List[str], str]
            Обработанный текст.
        """
        text = _preprocessing_text(text, preprocessing)
        return text

    @abstractmethod
    def _search(
        self,
        clear_text: str,
        k: int,
    ) -> Tuple[List[str], List[float]]:
        """Поиск наиболее схожих текстов из БД на clear_text пользователя.

        Параметры
        ----------
        clear_text : str
            Пользовательский текст, которому нужно найти наиболее схожие тексты из БД.
        k : int
            Кол-во выдаваемых результатов.

        Returns
        -------
        list_texts, list_similarity: Tuple[List[str], List[float]]
            Вычисленные результаты.
            В списке `list_texts` хранится текст, который похож на текст пользователя.
            В списке `list_similarity` хранится схожесть пользовательского текста на текст
                из бд в виде числа.
            Все данные соотносятся по индексу в этих двух списках.
        """

    def search(
        self,
        text: str,
        k: int,
        ascending: bool = False,
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
        text = self._preprocessing_text(text, self._preprocessing)

        list_texts, list_similarity = self._search(text, k)
        df = pd.DataFrame({'text': list_texts, 'similarity': list_similarity})
        df = df.sort_values(by=['similarity'], ascending=ascending)
        return df


class BaseEmbeddingSearch(BaseSearch):
    """Абстрактный класс для получения результатов
    поиска наиболее схожего текста из БД на основе эмбеддингов.
    """

    def __init__(
            self,
            model: BaseEmbedding,
            embedding_database,
            original_array: Iterable[str],
            metric: str,
            preprocessing: List[BaseTransformation] = [],
            clear_array: Optional[Iterable[str]] = None,
    ):
        super().__init__(
            original_array=original_array,
            preprocessing=preprocessing,
            clear_array=clear_array,
        )
        self._model = model
        self._embedding_database = embedding_database
        self._metric = metric

    @abstractmethod
    def _search(
        self,
        array: np.ndarray,
        k: int,
    ) -> Tuple[List[str], List[float]]:
        """Поиск наиболее схожих k-векторов из БД на array пользователя.

        Параметры
        ----------
        array : np.ndarray
            Пользовательский текст в виде эмбеддинга, которому нужно найти наиболее
            схожие вектора из БД.
        k : int
            Кол-во выдаваемых результатов.

        Returns
        -------
        list_texts, list_similarity: Tuple[List[str], List[float]]
            Вычисленные результаты.
            В списке `list_texts` хранится текст, который похож на текст пользователя.
            В списке `list_similarity` хранится схожесть пользовательского текста на текст
                из бд в виде числа.
            Все данные соотносятся по индексу в этих двух списках.
        """
    
    def search(
        self,
        text: str,
        k: int,
        ascending: bool = False,
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
        text = self._preprocessing_text(text, self._preprocessing)

        text = [text]
        array = self._model.transform(text)

        list_texts, list_similarity = self._search(array, k)
        df = pd.DataFrame({'text': list_texts, 'similarity': list_similarity})
        df = df.sort_values(by=['similarity'], ascending=ascending)
        return df


class BaseExplain(ABC):
    """Абстрактный класс для интерпретации схожести двух текстовых данных."""

    def __init__(
        self,
        model: BaseEmbedding,
        preprocessing: List[BaseTransformation] = [],
    ):
        self._model = model
        self._preprocessing = preprocessing

    def _preprocessing_text(
        self,
        text: Union[Iterable[str], str],
        preprocessing: List[BaseTransformation],
    ) -> Union[List[str], str]:
        """
        Предобработка текста.

        Параметры
        ----------
        text : Union[Iterable[str], str]
            Необработанный текст.
        preprocessing : List[BaseTransformation]
            Список алгоритмов для предобработка текста.

        Returns
        -------
        text: Union[List[str], str]
            Обработанный текст.
        """
        text = _preprocessing_text(text, preprocessing)
        return text

    def _split_by_words(
        self,
        clear_text: str,
        n_grams: Optional[int] = 1,
        sep: Optional[str] = ' ',
    ) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Получение N-грамм слов.

        Параметры
        ----------
        clear_text : str
            Предобработанный текст.
        n_grams : Optional[int]
            Длина N-грамм.
        sep: Optional[str]
            Разделитель слов.

        Returns
        -------
        n_grams_list: List[str]
            Список N-грамм слов.
        indeces_n_grams_list: List[Tuple[int, int]]
            Список кортежей индексов начала и конца N-граммы из `n_grams_list`.
        """
        tokens = clear_text.split(sep)

        start_idx = 0
        indeces_n_grams_list = []
        n_grams_list = []
        for i in range(len(tokens) - n_grams + 1):
            n_tokens_list = tokens[i:i + n_grams]
            n_gram = ' '.join(n_tokens_list)
            n_grams_list.append(n_gram)

            indeces = (start_idx, start_idx + len(n_gram))
            indeces_n_grams_list.append(indeces)
            start_idx += len(n_tokens_list[0]) + len(sep)
        return n_grams_list, indeces_n_grams_list

    def _split_by_chars(
        self,
        clear_text: str,
        n_grams: Optional[int] = 1,
        sep: Optional[str] = ' ',
    ) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Получение N-грамм символов.

        Параметры
        ----------
        clear_text : str
            Предобработанный текст.
        n_grams : Optional[int]
            Длина N-грамм.
        sep: Optional[str]
            Разделитель слов.

        Returns
        -------
        n_grams_list: List[str]
            Список N-грамм символов.
        indeces_n_grams_list: List[Tuple[int, int]]
            Список кортежей индексов начала и конца N-граммы из `n_grams_list`.
        """
        n_grams_words, _ = self._split_by_words(clear_text, n_grams=1, sep=sep)

        start_idx = 0
        indeces_n_grams_list = []
        n_grams_list = []
        for word in n_grams_words:
            counter = 0
            for i in range(len(word) - n_grams + 1):
                n_gram = word[i:i + n_grams]
                n_grams_list.append(n_gram)

                indeces = (start_idx, start_idx + len(n_gram))
                indeces_n_grams_list.append(indeces)
                start_idx += 1
                counter += 1
            start_idx += len(sep) + (len(word) - counter)
        return n_grams_list, indeces_n_grams_list

    @abstractmethod
    def _explain(
        self,
        clear_compared_text: str,
        clear_original_text: str,
        n_grams: int = 1,
        analyzer: Literal['word', 'char'] = 'word',
        sep: Optional[str] = ' ',
    ) -> Tuple[List[str], List[float], List[Tuple[int, int]]]:
        """
        Поиск наиболее схожих N-грамм из clear_compared_text в clear_original_text.

        Параметры
        ----------
        clear_compared_text : str
            Пользовательский текст, в котором нужно найти n-граммы,
            похожие на clear_original_text.
        clear_original_text : str
            Текст, с которым сравнивается clear_compared_text.
        n_grams : int
            Длина N-грамм.
        analyzer: Literal['word', 'char']
            Считать схожесть текстов на основе N-грамм слов или символов.
        sep: Optional[str]
            Разделитель слов.

        Returns
        -------
        text_list: List[str]
            Список N-грамм слов или символов.
        similarity_list: List[float]
            Список близости N-граммы к `clear_original_text`.
        indeces_n_grams_list: List[Tuple[int, int]]
            Список кортежей индексов старта и конца N-граммы из `text_list`.
        """

    def explain(
        self,
        compared_text: str,
        original_text: str,
        n_grams: Optional[Union[Tuple[int, int], int]] = 1,
        analyzer: Optional[Literal['word', 'char']] = 'word',
        sep: Optional[str] = ' ',
        k: Optional[int] = 10,
        ascending: Optional[bool] = True,
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
        clear_compared_text = self._preprocessing_text(compared_text, self._preprocessing)
        clear_original_text = self._preprocessing_text(original_text, self._preprocessing)

        if isinstance(n_grams, int):
            if n_grams <= 0:
                raise ValueError('The `n_grams` parameter must be > 0')

            text_list, similarity_list, indeces_n_grams_list = self._explain(
                clear_compared_text,
                clear_original_text,
                n_grams=n_grams,
                analyzer=analyzer,
                sep=sep,
            )
        else:
            if n_grams[0] <= 0:
                raise ValueError('The min value `n_grams` parameter must be > 0')

            text_list = []
            similarity_list = []
            indeces_n_grams_list = []
            for i in range(n_grams[0], n_grams[1]+1):
                tmp_text_list, tmp_similarity_list, tmp_indeces_n_grams_list = self._explain(
                    clear_compared_text,
                    clear_original_text,
                    n_grams=i,
                    analyzer=analyzer,
                    sep=sep,
                )
                text_list.extend(tmp_text_list)
                similarity_list.extend(tmp_similarity_list)
                indeces_n_grams_list.extend(tmp_indeces_n_grams_list)

        df = pd.DataFrame({
            'text': text_list,
            'similarity': similarity_list,
            'indeces_n_grams': indeces_n_grams_list,
        })
        df = df.sort_values(by=['similarity'], ascending=ascending).reset_index(drop=True)

        k = min(k, df.shape[0])
        df = df.iloc[:k]

        indeces_n_grams = df['indeces_n_grams'].values.tolist()
        df.drop(['indeces_n_grams'], axis=1, inplace=True)
        return df, indeces_n_grams


"""
Общие функции для Base-классов.
"""


def _preprocessing_text(
    text: Union[Iterable[str], str],
    preprocessing: List[BaseTransformation],
) -> Union[List[str], str]:
    """
    Предобработка текста.

    Параметры
    ----------
    text : Union[Iterable[str], str]
        Необработанный текст.
    preprocessing : List[BaseTransformation]
        Список алгоритмов для предобработка текста.

    Returns
    -------
    text: Union[List[str], str]
        Обработанный текст.
    """
    if isinstance(text, str):
        for transformator in preprocessing:
            tmp_text = transformator.transform([text])[0]
            if tmp_text:
                text = tmp_text
    else:
        for transformator in preprocessing:
            text = transformator.transform(text)
    return text
