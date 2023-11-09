"""
`recs.utils` модуль включает различные инструменты.
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
    _cosine_distance,
    _euclidean_distance,
    _inner_product_distance,
)

from ._wrapper import (
    WrapperTransform,
)

from ._log import (
    _create_date_name,
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
    '_euclidean_distance',
    '_inner_product_distance',

    'SentenceTransformerDataset',

    'WrapperTransform',

    '_create_date_name',
]
