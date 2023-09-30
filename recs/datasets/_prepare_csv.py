"""
Подготовка csv файла к единому реестровому формату.
"""


import pandas as pd
import os
import csv


CUR_PATH = os.path.dirname(__file__)


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

    df = pd.read_csv(f'{CUR_PATH}\\data\\{filename}.csv', sep=sep)

    # Целевой столбцев переименуем в `target`.
    df = df.rename(columns={
        col_name: 'target',
    })

    # Уберём дубликаты.
    df = pd.DataFrame({'target': df.target.unique()})

    # Удаление цитирующих ковычек.
    if remove_quoting:
        df.to_csv(f'{CUR_PATH}\\{filename}_new.csv', index=False,
                quoting=csv.QUOTE_NONE, quotechar='', escapechar='\\')
    else:
        df.to_csv(f'{CUR_PATH}\\data\\{filename}_new.csv', index=False)

    print(f'В `{CUR_PATH}` создан новый файл: `{filename}_new.csv`')


if __name__ == '__main__':

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
