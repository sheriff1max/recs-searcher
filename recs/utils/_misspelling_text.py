"""
Алгоритмы для создания ошибок в тексте.
Нужно для тестирования и обучения.
"""


import random
import string
from typing import Union


ALPHABET = {
    'ru': ''.join(list(map(chr, range(ord('А'), ord('я')+1)))),
    'eng': string.ascii_letters,
}
ALPHABET['any'] = ''.join([alph for alph in ALPHABET.values()])


def _change_random_sym(
        text: str,
        p: float,
        language: str = 'any',
        change_any_sym: bool = True
) -> str:
    """"""

    misspelling_text = ''
    for sym in text:

        if sym.isalpha() or change_any_sym:
            if random.random() < p:
                sym = random.choice(ALPHABET[language])

        misspelling_text += sym

    return misspelling_text


def _remove_random_sym(text: str, p: float) -> str:
    """"""

    misspelling_text = ''
    for sym in text:
        if not sym.isdigit() and sym != ' ':
            if random.random() < p:
                sym = random.choice(ALPHABET.get(language, ALPHABET['any']))

        misspelling_text += sym

    return misspelling_text


def _get_abbreviation(text: str,) -> str:
    """Берутся первые символы слов и превратить
    в одно слово. Нужно для сокращений НАИМЕНОВАНИЙ.
    Например, `Harry Potter` -> `HP`. """

    text = text.split(' ')
    text = [word[0] for word in text]
    text = ''.join(text)
    return text

def _remove_random_word(text: str, p: float) -> str:
    """"""


def _shuffle_words(text: str, p: float) -> str:
    """"""
