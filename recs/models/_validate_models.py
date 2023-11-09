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

        max_k = max(self._accuracy_top)
        dict_true_for_k = {k: 0 for k in self._accuracy_top}

        for i in tqdm(range(len(self._original_array))):
            augmentation_text = self._augmentation_array[i]
            original_text = self._original_array[i]

            top_i_df = self._searcher.search(augmentation_text, max_k)

            for k in dict_true_for_k.keys():
                if original_text in top_i_df.name.values[:k]:
                    dict_true_for_k[k] += 1

        for k in dict_true_for_k.keys():
            print(f'Top {k}Acc = {dict_true_for_k[k] / len(self._original_array)}')
