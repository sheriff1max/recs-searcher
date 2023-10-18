"""
Кастомные датасеты для обучения, унаследованные от PyTorch.
"""


import string
import random
from typing import Iterable, List, Callable, Union, Type, Dict

import numpy as np
from torch.utils.data import Dataset

from sentence_transformers import SentenceTransformer, models, InputExample, losses

from base import BaseAugmentation


class SentenceTransformerDataset(Dataset):
    """"""

    def __init__(
            self,
            array: Iterable[str],
            augmentation_transform: Union[None, List[BaseAugmentation]] = None
    ):
        self._array = [[text, text] for text in array]
        self._augmentation_transform = augmentation_transform

    def __len__(self):
        return len(self.array)

    def __getitem__(self, idx):

        lst = self.array[idx]

        original_text = lst[0]
        augmenation_text = lst[1]
        if self._augmentation_transform:
            for augmentation_func in self._augmentation_transform:
                augmenation_text = augmentation_func.transform(augmenation_text)

        return InputExample(texts=[original_text, augmenation_text])
