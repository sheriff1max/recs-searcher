"""
`recs.utils` модуль включает различные утилиты.
"""


from ._clear_text import (
    text_lower,
    remove_punct,
    remove_number,
    remove_whitespace,
    remove_html_tag,
    remove_url,
    remove_emoji,
)

__all__ = [
    'text_lower',
    'remove_punct',
    'remove_number',
    'remove_whitespace',
    'remove_html_tag',
    'remove_url',
    'remove_emoji',
]
