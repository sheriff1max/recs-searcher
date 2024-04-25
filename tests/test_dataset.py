import pytest
from recs_searcher import (
    dataset,  # учебные датасеты
)


def test_load_dataset():
    dataset_phones = dataset.load_mobile_phones()
    assert dataset_phones.shape[0] == 223  
    assert dataset_phones.shape[1] == 1  
