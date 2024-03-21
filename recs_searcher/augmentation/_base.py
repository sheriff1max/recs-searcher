import random
from abc import abstractmethod
from typing import List

from ..base import BaseTransformation


class BaseAugmentation(BaseTransformation):
    def __init__(
        self,
        min_aug: int = 1,
        max_aug: int = 5,
        seed: int = None,
    ) -> None:
        super().__init__(seed=seed)
        self.min_aug = min_aug
        self.max_aug = max_aug

    def __augs_count(
        self,
        size: int,
        rate: float,
    ) -> int:
        """Подсчитывает количество аугментаций и выполняет обрезание
        по максимальному или минимальному числу.
        
        Параметры
        ----------
            size: int
                Количество символов или слов в тексте.
            rate: float
                Процент символов или слов, к которым будет применено увеличение.

        Returns
        -------
            int: количество аугментаций.
        """
        cnt = 0
        if size > 1:
            cnt = int(rate * size)
        return cnt

    def __get_random_idx(
        self,
        inputs: List[str],
        aug_count: int,
    ) -> List[int]:
        """Рандомно выбирает индексы для аугментации.

        Параметры
        ----------
            inputs: List[str]
                Список символов или слов.
            aug_count: int
                Количество аугментаций.

        Returns:
        ----------
            List[int]: список индексов.
        """
        token_idxes = [i for i in range(len(inputs))]
        aug_idxs = random.sample(token_idxes, aug_count)
        return aug_idxs

    def _aug_indexing(
        self,
        inputs: List[str],
        rate: float,
        clip: bool = False
    ) -> List[int]:
        """
        Args:
            inputs: List[str]
                Список символов или слов.
            rate: float
                Процент символов или слов, к которым будет применено увеличение.
            clip: bool
                Учитывает максимальное и минимальное значения. По умолчанию False.

        Returns:
            List[int]: List of indices.
        """
        aug_count = self.__augs_count(len(inputs), rate)
        if clip:
            aug_count = max(aug_count, self.min_aug)
            aug_count = min(aug_count, self.max_aug)

        aug_idxs = self.__get_random_idx(inputs, aug_count)
        return aug_idxs

    @abstractmethod
    def _transform(self, array: List[str]) -> List[str]:
        pass
