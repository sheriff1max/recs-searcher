import pytest
from recs_searcher import (
    embeddings,  # преобразование текста в эмбеддинги
)


def test_embeddings_count_vectorizer():
    dataset = [
        'Красноярск',
        'Москва',
        'Санкт-Петербург',
        'Самара',
        'Томск',
        'Грозный',
        'Сочи',
    ]
    count_vectorizer = embeddings.CountVectorizerWrapperEmbedding(
        analyzer='char',
        ngram_range=(1, 2),
    )
    dataset_embeddings = count_vectorizer.fit_transform(dataset)
    assert dataset_embeddings.shape[0] == 7
    assert dataset_embeddings.shape[1] == 68

    count_vectorizer = embeddings.CountVectorizerWrapperEmbedding(
        analyzer='word',
        ngram_range=(1, 1),
    )
    dataset_embeddings = count_vectorizer.fit_transform(dataset)
    assert dataset_embeddings.shape[0] == 7
    assert dataset_embeddings.shape[1] == 8
