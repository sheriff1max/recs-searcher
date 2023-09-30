"""
Загрузка датасета в формате pandas.DataFrame
"""


import pandas as pd
import os


CUR_PATH = os.path.dirname(__file__)


def load_csv_data(filename: str) -> pd.DataFrame:
    """Загрузка csv-файла.

    Параметры
    ----------
    filename : str
        Путь до csv-файл.
        Все csv-файлы лежат в /recs/recs/datasets/data
        Например, 'city_Russia.csv'.

    Returns
    -------
    df: pd.DataFrame
        Считанный csv-файл.
    """
    df = pd.read_csv(CUR_PATH + '\\data\\' + filename)
    return df


def load_city_russia():
    """Загрузка датасета с городами России.

    Датасет содержит только уникальные значения городов.
    Предназначен для системы, в которой нужно исправлять
    пользовательский ввод городов России.

    =================   ==============
    Кол-во строк            1083
    Кол-во столбцов           1
    =================   ==============

    Returns
    -------
    df: pd.DataFrame
        Считанные данные о городах России.
    """

    df = load_csv_data('Russia_cities.csv')
    return df


def load_video_games():
    """Загрузка датасета с названиями видео-игр.

    Датасет содержит только уникальные значения видео-игр.
    Предназначен для системы, в которой нужно исправлять
    пользовательский ввод видео-игр.

    =================   ==============
    Кол-во строк            11564
    Кол-во столбцов           1
    =================   ==============

    Returns
    -------
    df: pd.DataFrame
        Считанные данные о видео-играх.
    """

    df = load_csv_data('video_games.csv')
    return df


def load_pattern():
    """Загрузка датасета с названиями pattern.

    Датасет содержит только уникальные значения pattern
    Предназначен для системы, в которой нужно исправлять
    пользовательский ввод pattern

    =================   ==============
    Кол-во строк            pattern
    Кол-во столбцов         pattern
    =================   ==============

    Returns
    -------
    df: pd.DataFrame
        Считанные данные о pattern
    """

    df = load_csv_data('pattern.csv')
    return df


if __name__ == '__main__':
    pass
    # tmp = load_city_russia()
    # print(tmp)
