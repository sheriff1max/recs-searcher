"""
Алгоритмы для валидации моделей.
"""


from typing import Iterable, Union, List
from base import BaseSearch, BaseTransformation, BaseDataset

from tqdm import tqdm


class Validate:
    """Класс валидации моделей."""

    def __init__(
            self,
            searcher: BaseSearch,
            augmentation_transforms: List[BaseTransformation],
            accuracy_top: List[int] = [1, 5, 10],
    ):
        self._searcher = searcher
        self._original_array = searcher._original_array

        self._augmentation_array = searcher._original_array
        for augmentation_transform in augmentation_transforms:
            self._augmentation_array = augmentation_transform.transform(self._augmentation_array)

        self._accuracy_top = accuracy_top

        self._validate()

    def _validate(self) -> None:
        """"""

        for i in self._accuracy_top:

            true = 0
            for k in tqdm(range(len(self._original_array))):
                augmentation_text = self._augmentation_array[k]
                original_text = self._original_array[k]

                top_i_df = self._searcher.search(augmentation_text, i)

                if original_text in top_i_df.name.values:
                    true += 1
                    # print(augmentation_text, top_i_df.name.values)

            print(f'Top {i}Acc = {true / len(self._original_array)}')
