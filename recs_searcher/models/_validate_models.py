"""
Алгоритмы для валидации моделей.
"""


from typing import Iterable, Union, List, Dict
from ..base import BaseSearch, BaseTransformation, BaseDataset

from tqdm import tqdm


class Validate:
    """Класс валидации моделей."""

    def __new__(
            cls,
            searcher: BaseSearch,
            augmentation_transforms: List[BaseTransformation],
            accuracy_top: List[int] = [1, 5, 10],
    ) -> Dict[int, float]:
        """"""
        original_array = searcher._original_array

        augmentation_array = searcher._original_array
        for augmentation_transform in augmentation_transforms:
            augmentation_array = augmentation_transform.transform(augmentation_array)

        max_k = max(accuracy_top)
        dict_true_for_k = {k: 0 for k in accuracy_top}

        for i in tqdm(range(len(original_array))):
            augmentation_text = augmentation_array[i]
            original_text = original_array[i]

            top_i_df = searcher.search(augmentation_text, max_k)

            for k in dict_true_for_k.keys():
                if original_text in top_i_df.name.values[:k]:
                    dict_true_for_k[k] += 1

        metrics = {}
        for k in dict_true_for_k.keys():
            accuracy_k = dict_true_for_k[k] / len(original_array)
            metrics[k] = accuracy_k
            print(f'Top {k}Acc = {accuracy_k}')
        return metrics
