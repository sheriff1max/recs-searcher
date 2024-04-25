import pytest
from recs_searcher import (
    dataset,  # учебные датасеты
    preprocessing,  # предобработка текста
    embeddings,  # преобразование текста в эмбеддинги
    similarity_search,  # быстрые поисковики в пространстве эмбеддингов
    augmentation,  # аугментация текста для валидации пайплайнов
    explain,  # интерпретация сходства двух текстов
    api,  # Пайплайн
)


def test_pipeline():
    SEED = 1

    dataset_phones = dataset.load_mobile_phones()
    preprocessing_list = [
        preprocessing.TextLower(),
        preprocessing.RemovePunct(),
        preprocessing.RemoveWhitespace(),
    ]
    model_count_vectorizer_char = embeddings.CountVectorizerWrapperEmbedding(
        analyzer='char',
        ngram_range=(1, 2),
    )
    explainer = explain.DistanceExplain
    searcher_faiss = similarity_search.FaissSearch
    validate_augmentation_transforms = [
        augmentation.CharAugmentation(
            action='insert',
            unit_prob=1.0,
            min_aug=1,
            max_aug=2,
            mult_num=2,
            seed=SEED,
        ),
        augmentation.CharAugmentation(
            action='delete',
            unit_prob=1.0,
            min_aug=1,
            max_aug=2,
            mult_num=2,
            seed=SEED,
        ),
    ]
    accuracy_top = [1, 5, 10]

    pipeline = api.Pipeline(
        dataset=dataset_phones.target.values,
        preprocessing=preprocessing_list,
        model=model_count_vectorizer_char,
        explainer=explainer,
        searcher=searcher_faiss,
        verbose=True,
    )
    assert pipeline.get_model() is not None
    assert pipeline.get_preprocessing() is not None

    df = pipeline.search('phone', 5, ascending=True)
    assert df.shape[0] == 5
    assert df.shape[1] == 2

    metrics = pipeline.validate(
        validate_augmentation_transforms,
        accuracy_top,
        ascending=True
    )
    assert isinstance(metrics, dict)
    assert len(metrics) == len(accuracy_top)

    df, indeces_n_grams = pipeline.explain(
        compared_text='Donald Trump bought an Apple iPhone 13 Pro Max and called his colleague Vladimir Putin',
        original_text='Apple iPhone 13 Pro Max',
        n_grams=(1, 5),
        analyzer='word',
        sep=' ',
        k=10,
        ascending=True,
    )
    assert df.shape[0] == 10
    assert df.shape[1] == 2
    assert df.iloc[0, 0] == 'apple iphone 13 pro max'
    assert len(indeces_n_grams) == 10
    assert isinstance(indeces_n_grams[0], tuple)
