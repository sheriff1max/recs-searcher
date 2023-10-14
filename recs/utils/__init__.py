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

__all__ = [
    '_text_lower',
    '_remove_punct',
    '_remove_number',
    '_remove_whitespace',
    '_remove_html_tag',
    '_remove_url',
    '_remove_emoji',

    '_cosine_distance',
]
