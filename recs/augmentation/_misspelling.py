"""
Алгоритмы для создания ошибок в словах.
"""


import random
import string
from typing import Union, Callable, Literal


ALPHABET = {
    'russian': ''.join(list(map(chr, range(ord('А'), ord('я')+1)))),
    'english': string.ascii_letters,
}
ALPHABET['any'] = ''.join([alpha for alpha in ALPHABET.values()])


def _change_syms(
        text: str,
        p: float = 0.05,
        language: Literal['russian', 'english', 'any'] = 'any',
        change_only_alpha: bool = True,
) -> str:
    """"""
    changed_text = ''
    for sym in text:

        if sym.isalpha() or not change_only_alpha:
            if random.randrange(100) < p * 100:
                sym = random.choice(ALPHABET[language])

        changed_text += sym
    return changed_text


def _delete_syms(
        text: str,
        p: float = 0.05,
        delete_only_alpha: bool = True,
) -> str:
    """"""
    changed_text = ''
    for sym in text:

        if sym.isalpha() or not delete_only_alpha:
            if random.randrange(100) < p * 100:
                continue

        changed_text += sym
    return changed_text


def _add_syms(
        text: str,
        p: float = 0.05,
        language: Literal['russian', 'english', 'any'] = 'any',
) -> str:
    """"""
    changed_text = ''
    for sym in text:

        if random.randrange(100) < p * 100:
            sym += random.choice(ALPHABET[language])

        changed_text += sym
    return changed_text


def _multiply_syms(
        text: str,
        p: float = 0.05,
        count_multiply: int = 2,
        multiply_only_alpha: bool = True
) -> str:
    """"""
    changed_text = ''
    for sym in text:

        if sym.isalpha() or not multiply_only_alpha:
            if random.randrange(100) < p * 100:
                sym = sym * count_multiply

        changed_text += sym
    return changed_text


def _swap_syms(
        text: str,
        p: float = 0.05,
) -> str:
    """"""
    text = list(text)
    for i in range(0, len(text)-1):

        if random.randrange(100) < p * 100:
            text[i], text[i+1] = text[i+1], text[i]
    return ''.join(text)


if __name__ == '__main__':
    print(_change_syms('hello man'))
    print(_delete_syms('hello man'))
    print(_add_syms('hello man'))
    print(_multiply_syms('hello man'))
    print(_swap_syms('hello man'))

    # print(wrapper_func_seed(_change_syms, 43, text='hello man'))

    # a = WrapperFunc(_change_syms, 43, language='ru')
    # print(a.transform('hello man'))
