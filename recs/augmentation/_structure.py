"""
Алгоритмы по изменению структуры текста.
"""


import random


def _get_abbreviation(
        text: str,
        sep: str = ' '
) -> str:
    """Берутся первые символы слов и превратить
    в одно слово. Нужно для сокращений НАИМЕНОВАНИЙ.
    Например, `Harry Potter` -> `HP`. """
    text = text.split(sep)
    changed_text = [word[0] for word in text if word]
    changed_text = ''.join(changed_text)
    return changed_text


def _delete_words(
        text: str,
        p: float = 0.1,
        delete_only_alpha: bool = True,
        sep: str = ' '
) -> str:
    """"""
    original_text = text.split(sep)
    changed_text = []
    for word in original_text:

        if word.isalpha() or not delete_only_alpha:
            if random.randrange(100) < p * 100 and len(original_text) != 1:
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