"""
Подготовка csv файла к единому реестровому формату.
"""


import pandas as pd
from pathlib import Path
import csv


CUR_PATH = Path(__file__).parents[0]


def prepare_csv(
        filename: str,
        col_name: str,
        sep=',',
        remove_quoting=False
) -> None:
    """Приводит csv-файл в формат для внедрения в систему.

    Новые csv-файлы будут иметь один столбец - `target`.
    В столбце `target` будут храниться уникальные значения.

    Целевой столбец должен иметь наименования сущностей,
    которые в будущем будут являться стандартом системы
    по исправлению/приведению пользовательского ввода к
    единому формату.

    Параметры
    ----------
    filename : str
        Путь до csv-файл.
        Например, 'test.csv' или 'test'.

    col_name: str
        Название столбца в csv-файле с наименованными сущностями.

    sep: str, default=','
        Раздилитель в csv-файле.
    
    remove_quoting: bool, default=False
        Удаляет ковычки цитирования в итоговом csv-файле.

    Returns
    -------
    None
    """

    if '.csv' in filename:
        filename = filename[:-4]

    path_old_file = CUR_PATH / 'data' / f'{filename}.csv'
    df = pd.read_csv(path_old_file, sep=sep)

    # Целевой столбцев переименуем в `target`.
    df = df.rename(columns={
        col_name: 'target',
    })

    # Уберём дубликаты.
    df = pd.DataFrame({'target': df.target.unique()})

    # Удаление цитирующих ковычек.
    path_new_file = CUR_PATH / 'data' / f'{filename}_new.csv'
    if remove_quoting:
        df.to_csv(path_new_file, index=False,
                quoting=csv.QUOTE_NONE, quotechar='', escapechar='\\')
    else:
        df.to_csv(path_new_file, index=False)

    print(f'{path_new_file} создан.')


if __name__ == '__main__':
    # _prepare_csv.py -f filename.csv -c col_name

    import argparse

    parser = argparse.ArgumentParser(description='test')

    # filename
    parser.add_argument(
        '-f',
        '--filename',
        type=str,
        required=True,
        help='Название csv-файла',
    )
    # col_name
    parser.add_argument(
        '-c',
        '--col_name',
        type=str,
        required=True,
        help='Название столбца с именованными сущностями',
    )
    # sep
    parser.add_argument(
        '-s',
        '--sep',
        type=str,
        default=',',
        help='Раздилитель в csv-файле',
    )
    # remove_quoting
    parser.add_argument(
        '-r',
        '--remove_quoting',
        type=bool,
        default=False,
        help='Удаление цитирующих ковычек',
    )

    args = parser.parse_args()

    # Выхов функции.
    prepare_csv(args.filename, args.col_name, args.sep, args.remove_quoting)
