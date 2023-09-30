"""
Алгоритмы для очистки текста.
"""


import re
import string


def text_lower(text: str) -> str:
    """"""
    return text.lower()


def remove_punct(text: str) -> str:
    """"""
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_number(text: str) -> str:
    """"""
    return re.sub(r'\d+', "", text)


def remove_whitespace(text: str) -> str:
    """"""
    return text.strip()


def remove_html_tag(text: str) -> str:
    """"""
    html_pattern = re.compile('<.*?>')
    return re.sub(html_pattern, '', text) 


def remove_url(text: str) -> str:
    """"""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return re.sub(url_pattern, '', text) 


def remove_emoji(text: str) -> str:
    """"""
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # смайлики.
        u"\U0001F300-\U0001F5FF"  # символы и пиктограммы.
        u"\U0001F680-\U0001F6FF"  # транспорт и символы на карте.
        u"\U0001F1E0-\U0001F1FF"  # флаги (iOS).
                            "]+", flags=re.UNICODE)
    return re.sub(emoji_pattern, '', text)
