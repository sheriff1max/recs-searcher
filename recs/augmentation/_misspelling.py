"""
Алгоритмы для создания ошибок в словах.
"""


import random
import string
from typing import Union, Callable


ALPHABET = {
    'ru': ''.join(list(map(chr, range(ord('А'), ord('я')+1)))),
    'eng': string.ascii_letters,
}
ALPHABET['any'] = ''.join([alpha for alpha in ALPHABET.values()])


def wrapper_func_seed(
        *data,
        func: Callable,
        seed=Union[None, int]
) -> str:
    """"""
    random.seed(seed)
    return func(*data)


def _change_syms(
        text: str,
        p: float = 0.1,
        language: str = 'any',
        change_only_alpha: bool = True,
) -> str:
    """"""
    changed_text = ''
    for sym in text:

        if sym.isalpha() or not change_only_alpha:
            if random.random() < p:
                sym = random.choice(ALPHABET[language])

        changed_text += sym
    return changed_text


def _delete_syms(
        text: str,
        p: float = 0.1,
        delete_only_alpha: bool = True,
) -> str:
    """"""
    changed_text = ''
    for sym in text:

        if sym.isalpha() or not delete_only_alpha:
            if random.random() < p:
                continue

        changed_text += sym
    return changed_text


def _add_syms(
        text: str,
        p: float = 0.1,
        language: str = 'any',
) -> str:
    """"""
    changed_text = ''
    for sym in text:

        if random.random() < p:
            sym += random.choice(ALPHABET[language])

        changed_text += sym
    return changed_text


def _multiply_syms(
        text: str,
        p: float = 0.1,
        count_multiply: int = 2,
        multiply_only_alpha: bool = True
) -> str:
    """"""
    changed_text = ''
    for sym in text:

        if sym.isalpha() or not multiply_only_alpha:
            if random.random() < p:
                sym = sym * count_multiply

        changed_text += sym
    return changed_text


def _swap_syms(
        text: str,
        p: float = 0.1,
) -> str:
    """"""
    text = list(text)
    for i in range(0, len(text)-1):

        if random.random() < p:
            text[i], text[i+1] = text[i+1], text[i]
    return ''.join(text)


if __name__ == '__main__':
    print(_change_syms('hello man'))
    print(_delete_syms('hello man'))
    print(_add_syms('hello man'))
    print(_multiply_syms('hello man'))
    print(_swap_syms('hello man'))
