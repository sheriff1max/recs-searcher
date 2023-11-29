"""
Алгоритмы для логирования.
"""


import datetime


def _create_date_name(name: str) -> str:
    """"""
    now_datetime = []
    for symbol in str(datetime.datetime.now()):
        now_datetime.append(
            symbol if symbol not in (" ", ".", ":") else "-"
        )
    name = f"{name}_{''.join(now_datetime)}"
    return name
