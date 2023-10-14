"""
Кастомные датасеты, унаследованные от PyTorch.
"""


import string
import random
from typing import Iterable, List, Callable, Union, Type, Dict

import numpy as np
from torch.utils.data import Dataset

from sentence_transformers import SentenceTransformer, models, InputExample, losses

from base import BaseTransformation


class SentenceTransformerDataset(Dataset):
    """"""

    def __init__(
            self,
            array: Iterable[str],
            augmentation_misspelling: Union[BaseTransformation, \
                                            Callable[[str], str]] = lambda x: x
    ):
        self._array = [[text, text] for text in array]
        self._augmentation_misspelling = augmentation_misspelling

    def __len__(self):
        return len(self.array)

    def __getitem__(self, idx):

        lst = self.array[idx]

        original_text = lst[0]
        similar_text = self._augmentation_misspelling(lst[1])

        return InputExample(texts=[original_text, similar_text])
