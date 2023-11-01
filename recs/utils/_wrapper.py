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
            seed: Union[None, int] = None,
            **data,
    ):
        self._func = func
        self._seed = seed
        self._data = data

    def transform(self, text: str):
        """"""
        return _wrapper_func_seed(self._func, self._seed, text=text, **self._data)


def _wrapper_func_seed(
        func: Callable,
        seed: Union[None, int] = None,
        **data,
) -> str:
    """"""
    random.seed(seed)
    return func(**data)
