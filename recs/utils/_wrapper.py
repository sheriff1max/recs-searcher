"""
Инструменты для обёртки других функций.
"""


from typing import Callable, Union
import random


class WrapperTransform:
    """"""

    def __init__(
            self,
            func: Callable,
            seed: int = 0,
            **data,
    ):
        self._func = func
        self._start_seed = seed
        self._seed = seed
        self._data = data

    def transform(self, text: str):
        """"""
        return _wrapper_func_seed(self._func, self._seed, text=text, **self._data)

    def up_seed(self):
        self._seed += 1

    def reset_seed(self):
        self._seed = self._start_seed


def _wrapper_func_seed(
        func: Callable,
        seed: int = 0,
        **data,
) -> str:
    """"""
    random.seed(seed)
    return func(**data)
