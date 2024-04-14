"""
Алгоритмы для логирования.
"""


import datetime


def create_date_name(name: str) -> str:
    """Генерация текста на основе даты.

        Параметры:
        ----------
        name : str
            Строка, к которой добавить сгенерированный текст.

        Returns
        -------
        name: str
            Сгенерированный текст.
    """
    now_datetime = []
    for symbol in str(datetime.datetime.now()):
        now_datetime.append(
            symbol if symbol not in (" ", ".", ":") else "-"
        )
    name = f"{name}_{''.join(now_datetime)}"
    return name
