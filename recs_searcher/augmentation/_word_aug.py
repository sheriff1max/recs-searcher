import re
from typing import List, Optional, Literal
import numpy as np

from ._base import BaseAugmentation
from ._actions import WORD_ACTIONS


class WordAugmentation(BaseAugmentation):
    """Augmentation at the level of words."""

    def __init__(
        self,
        unit_prob: float = 0.3,
        min_aug: int = 1,
        max_aug: int = 5,
        action: Optional[Literal["delete", "swap", "split"]] = None,
        seed: int = None,
    ) -> None:
        super().__init__(
            min_aug=min_aug,
            max_aug=max_aug,
            seed=seed,
        )
        self.unit_prob = unit_prob
        if action is None:
            self.action = np.random.choice(WORD_ACTIONS)
        else:
            self.action = action

    @property
    def actions_list(self) -> List[str]:
        """
        Returns:
            List[str]: A list of possible methods.
        """

        return WORD_ACTIONS

    def __split(self, word: str) -> str:
        """Divides a word character-by-character.

        Args:
            word (str): A word with the correct spelling.

        Returns:
            str: Word with spaces.
        """
        word = " ".join(list(word))
        return word

    def __delete(self) -> str:
        """Deletes a random word.

        Returns:
            str: Empty string.
        """
        return ""

    def _transform(self, array: List[str]) -> List[str]:
        transformed_array = []

        for text in array:
            aug_sent_arr = text.split()
            aug_idxs = self._aug_indexing(aug_sent_arr, self.unit_prob, clip=True)
            for idx in aug_idxs:
                if self.action == "delete":
                    aug_sent_arr[idx] = self.__delete()
                elif self.action == "swap":
                    swap_idx = np.random.randint(0, len(aug_sent_arr) - 1)
                    aug_sent_arr[swap_idx], aug_sent_arr[idx] = (
                        aug_sent_arr[idx],
                        aug_sent_arr[swap_idx],
                    )
                elif self.action == "split":
                    aug_sent_arr[idx] = self.__split(aug_sent_arr[idx])
                else:
                    raise NameError(
                        """These type of augmentation is not available, please check EDAAug.actions_list() to see
                    available augmentations"""
                    )

            text = re.sub(" +", " ", " ".join(aug_sent_arr).strip())
            transformed_array.append(text)

        return transformed_array
