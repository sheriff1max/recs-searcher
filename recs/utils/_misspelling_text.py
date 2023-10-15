"""
Алгоритмы для создания ошибок в тексте.
Нужно для тестирования и обучения.
"""


import random
import string
from typing import Union, Callable


ALPHABET = {
    'ru': ''.join(list(map(chr, range(ord('А'), ord('я')+1)))),
    'eng': string.ascii_letters,
}
ALPHABET['any'] = ''.join([alph for alph in ALPHABET.values()])


def _wrapper_func_seed(
        *data,
        func: Callable,
        seed=Union[None, int]
) -> str:
    """"""
    random.seed(seed)
    return func(*data)


def _change_random_sym(
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


def _remove_random_sym(
        text: str,
        p: float = 0.1,
        remove_only_alpha: bool = True,
) -> str:
    """"""

    changed_text = ''
    for sym in text:

        if sym.isalpha() or not remove_only_alpha:
            if random.random() < p:
                continue

        changed_text += sym

    return changed_text


def _get_abbreviation(
        text: str,
        sep: str = ' '
) -> str:
    """Берутся первые символы слов и превратить
    в одно слово. Нужно для сокращений НАИМЕНОВАНИЙ.
    Например, `Harry Potter` -> `HP`. """

    text = text.split(sep)
    changed_text = [word[0] for word in text]
    changed_text = ''.join(changed_text)
    return changed_text


def _remove_random_word(
        text: str,
        p: float = 0.2,
        remove_only_alpha: bool = True,
        sep: str = ' '
) -> str:
    """"""

    changed_text = []
    for word in text.split(sep):

        if word.isalpha() or not remove_only_alpha:
            if random.random() < p:
                continue
        changed_text.append(word)
    
    changed_text = ' '.join(changed_text)
    return changed_text


def _shuffle_words(
        text: str,
        sep: str = ' '
) -> str:
    """"""

    changed_text = text.split(sep)
    changed_text = random.sample(changed_text, len(changed_text))
    changed_text = ' '.join(changed_text)
    return changed_text


if __name__ == '__main__':
    print(_shuffle_words('hello man'))
    print(_change_random_sym('hello man'))
    print(_get_abbreviation('hello man'))
    print(_remove_random_sym('hello man'))
    print(_remove_random_word('hello man'))
