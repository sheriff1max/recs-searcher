"""
Алгоритмы для валидации моделей.
"""


from typing import List, Dict
from ..base import BaseSearch, BaseTransformation
from tqdm import tqdm


class Validate:
    """Класс валидации пайплайна."""

    def __new__(
            cls,
            searcher: BaseSearch,
            augmentation_transforms: List[BaseTransformation],
            accuracy_top: List[int] = [1, 5, 10],
            ascending: bool = True,
    ) -> Dict[int, float]:
        """
        Получение метрик точности обученной модели.

        Параметры
        ----------
        searcher : BaseSearch
            Алгоритм на основе которого будут искаться схожие текста.
        augmentation_transforms : List[BaseTransformation]
            Список алгоритмов аугментации для создания ошибок в тексте.
        accuracy_top : Optional[List[int]]
            Список для оценивания N@Accuracy.
        ascending : Optional[bool]
            Флаг сортировки полученных результатов.
            False - убывающая, True - возрастающая сортировка.

        Returns
        -------
        score_metrics: Dict[int, float]
            Посчитанные метрики.
        """
        original_array = searcher._original_array

        augmentation_array = searcher._original_array
        for augmentation_transform in augmentation_transforms:
            augmentation_array = augmentation_transform.transform(augmentation_array)

        max_k = max(accuracy_top)
        dict_true_for_k = {k: 0 for k in accuracy_top}

        for i in tqdm(range(len(original_array))):
            augmentation_text = augmentation_array[i]
            original_text = original_array[i]

            top_i_df = searcher.search(augmentation_text, max_k, ascending=ascending)

            for k in dict_true_for_k.keys():
                if original_text in top_i_df.text.values[:k]:
                    dict_true_for_k[k] += 1

        score_metrics = {}
        for k in dict_true_for_k.keys():
            accuracy_k = dict_true_for_k[k] / len(original_array)
            score_metrics[k] = accuracy_k
            print(f'Top {k}Acc = {accuracy_k}')
        return score_metrics
