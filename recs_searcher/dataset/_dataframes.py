"""
Загрузка датасета в формате pandas.DataFrame
"""


import pandas as pd
from pathlib import Path

import pathlib
import platform
if platform.system() == 'Linux':
    pathlib.WindowsPath = pathlib.PosixPath


CUR_PATH = Path(__file__).parents[0]


def load_city_russia() -> pd.DataFrame:
    """Загрузка датасета с городами России.
    Датасет содержит только уникальные значения.

    =================   ==============
    Кол-во строк            1083
    Кол-во столбцов           1
    =================   ==============

    Returns
    -------
    df: pd.DataFrame
        Считанные данные.
    """

    df = _load_csv_data('city_russia.csv')
    return df


def load_video_games() -> pd.DataFrame:
    """Загрузка датасета с названиями видео-игр.
    Датасет содержит только уникальные значения.

    =================   ==============
    Кол-во строк            11564
    Кол-во столбцов           1
    =================   ==============

    Returns
    -------
    df: pd.DataFrame
        Считанные данные.
    """

    df = _load_csv_data('video_games.csv')
    return df


def load_exoplanes() -> pd.DataFrame:
    """Загрузка датасета с названиями планет.
    Датасет содержит только уникальные значения.

    =================   ==============
    Кол-во строк            5507
    Кол-во столбцов         1
    =================   ==============

    Returns
    -------
    df: pd.DataFrame
        Считанные данные.
    """

    df = _load_csv_data('exoplanets.csv')
    return df


def load_company_russia() -> pd.DataFrame:
    """Загрузка датасета с названиями ООО из России.
    Датасет содержит только уникальные значения.

    =================   ==============
    Кол-во строк            5246
    Кол-во столбцов         1
    =================   ==============

    Returns
    -------
    df: pd.DataFrame
        Считанные данные.
    """

    df = _load_csv_data('company_russia.csv')
    return df


def load_address_krasnoyarsk() -> pd.DataFrame:
    """Загрузка датасета с адресами Красноярска.
    Датасет содержит только уникальные значения.

    =================   ==============
    Кол-во строк            72886
    Кол-во столбцов         1
    =================   ==============

    Returns
    -------
    df: pd.DataFrame
        Считанные данные.
    """

    df = _load_csv_data('address_krasnoyarsk.csv')
    return df


def load_medical_organizations() -> pd.DataFrame:
    """Загрузка датасета с названиями медицинских организаций из России.
    Датасет содержит только уникальные значения.


    =================   ==============
    Кол-во строк            193
    Кол-во столбцов         1
    =================   ==============

    Returns
    -------
    df: pd.DataFrame
        Считанные данные.
    """

    df = _load_csv_data('medical_organizations.csv')
    return df


def load_medical_supplies() -> pd.DataFrame:
    """Загрузка датасета с названиями медицинских препаратов.
    Датасет содержит только уникальные значения.


    =================   ==============
    Кол-во строк            1211
    Кол-во столбцов         1
    =================   ==============

    Returns
    -------
    df: pd.DataFrame
        Считанные данные.
    """

    df = _load_csv_data('medical_supplies.csv')
    return df


def load_mobile_phones() -> pd.DataFrame:
    """Загрузка датасета с названиями смартфонов.
    Датасет содержит только уникальные значения.


    =================   ==============
    Кол-во строк            224
    Кол-во столбцов         1
    =================   ==============

    Returns
    -------
    df: pd.DataFrame
        Считанные данные.
    """

    df = _load_csv_data('mobile_phones.csv')
    return df


def load_place_address_russia() -> pd.DataFrame:
    """Загрузка датасета с местами выдачи паспортов.
    Датасет содержит только уникальные значения.


    =================   ==============
    Кол-во строк            12588
    Кол-во столбцов         1
    =================   ==============

    Returns
    -------
    df: pd.DataFrame
        Считанные данные.
    """

    df = _load_csv_data('place_address_russia.csv')
    return df


# def load_pattern() -> pd.DataFrame:
#     """Загрузка датасета с названиями pattern.
#     Датасет содержит только уникальные значения.


#     =================   ==============
#     Кол-во строк            pattern
#     Кол-во столбцов         pattern
#     =================   ==============

#     Returns
#     -------
#     df: pd.DataFrame
#         Считанные данные.
#     """

#     df = _load_csv_data('pattern.csv')
#     return df


def _load_csv_data(filename: str) -> pd.DataFrame:
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
    path = CUR_PATH / Path('data') / Path(filename)
    df = pd.read_csv(path)
    return df
