"""
`recs.utils` модуль включает различные утилиты.
"""


from ._clear_text import (
    _text_lower,
    _remove_punct,
    _remove_number,
    _remove_whitespace,
    _remove_html_tag,
    _remove_url,
    _remove_emoji,
)

from ._metrics import (
    _cosine_distance
)

from ._train_datasets import (
    SentenceTransformerDataset
)

from ._wrapper import (
    wrapper_func_seed,
    WrapperTransform,
)

__all__ = [
    '_text_lower',
    '_remove_punct',
    '_remove_number',
    '_remove_whitespace',
    '_remove_html_tag',
    '_remove_url',
    '_remove_emoji',

    '_cosine_distance',

    'SentenceTransformerDataset',

    'wrapper_func_seed',
    'WrapperTransform',
]
