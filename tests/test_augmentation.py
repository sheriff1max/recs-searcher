import pytest
from recs_searcher import (
    augmentation,  # аугментация текста для валидации пайплайнов
)


def test_augmentation_delete():
    text = 'Нижний Новгород'
    transformer = augmentation.CharAugmentation(
        action='delete',
        unit_prob=1.0,
        min_aug=1,
        max_aug=2,
        mult_num=2,
        seed=1,
    )
    transformed_text = transformer.transform([text])[0]
    assert transformed_text == 'Ниний Ногород'  
