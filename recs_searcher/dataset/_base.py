"""
Обёртки датасетов для различных моделей.
"""


from typing import Iterable, Union, List, Optional

from ..base import BaseTransformation, BaseDataset
from sentence_transformers import InputExample


class SentenceTransformerDataset(BaseDataset):
    """"""

    def __init__(
            self,
            array: Iterable[str],
            augmentation_transform: Optional[List[BaseTransformation]] = None,
    ):
        super().__init__(array=array)
        self._augmentation_transform = augmentation_transform

    def __len__(self):
        return len(self._array)

    def __getitem__(self, idx):

        text = self._array[idx]

        original_text = text
        augmenation_text = text
        if self._augmentation_transform:
            for augmentation_func in self._augmentation_transform:
                augmenation_text = augmentation_func.transform([augmenation_text])[0]
        return InputExample(texts=[original_text, augmenation_text])


class StandartDataset(BaseDataset):
    """"""

    def __init__(
            self,
            array: Iterable[str],            
    ):
        super().__init__(array=array)

    def __len__(self):
        return len(self._array)

    def __getitem__(self, idx):
        text = self._array[idx]
        return text
