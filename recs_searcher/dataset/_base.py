"""
Обёртки датасетов для различных моделей.
"""


from typing import Iterable, List, Optional
from ..base import BaseTransformation
from sentence_transformers import InputExample
from torch.utils.data import Dataset


# TODO: обновить для use-case поиска схожих текстов.
class SentenceTransformerDataset(Dataset):
    """Обёртка дадасета для эмбеддингов из Sentence-Transformers."""

    def __init__(
        self,
        array: Iterable[str],
        augmentation_transform: Optional[List[BaseTransformation]] = None,
    ):
        self._array = array
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
