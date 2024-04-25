import pytest
from recs_searcher import (
    preprocessing,  # предобработка текста
)


def test_preprocessing_text_lower():
    text = 'Hello, Vladimir Putin. How are you?   :) '
    transformer = preprocessing.TextLower()
    transformed_text = transformer.transform([text])[0]
    assert transformed_text == 'hello, vladimir putin. how are you?   :) '  


def test_preprocessing_remove_punct():
    text = 'Hello, Vladimir Putin. How are you?   :) '
    transformer = preprocessing.RemovePunct()
    transformed_text = transformer.transform([text])[0]
    assert transformed_text == 'Hello  Vladimir Putin  How are you       '  


def test_preprocessing_remove_whitespaces():
    text = 'Hello, Vladimir Putin. How are you?   :) '
    transformer = preprocessing.RemoveWhitespace()
    transformed_text = transformer.transform([text])[0]
    assert transformed_text == 'Hello, Vladimir Putin. How are you?  :)' 
