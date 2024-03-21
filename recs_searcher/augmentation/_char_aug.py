from typing import List, Union, Optional, Literal
import numpy as np

from ._base import BaseAugmentation
from ._actions import CHAR_ACTIONS


class CharAugmentation(BaseAugmentation):
    """Augmentation at the character level."""

    def __init__(
        self,
        unit_prob: float = 0.3,
        min_aug: int = 1,
        max_aug: int = 5,
        mult_num: int = 5,
        action: Optional[Literal["delete", "multiply", "swap", "insert"]] = None,
        seed: Union[int, None] = None,
    ) -> None:
        super().__init__(
            min_aug=min_aug,
            max_aug=max_aug,
            seed=seed,
        )
        self.mult_num = mult_num
        self.unit_prob = unit_prob
        if action is None:
            action = np.random.choice(CHAR_ACTIONS)
        else:
            self.action = action

    @property
    def actions_list(self) -> List[str]:
        """
        Returns:
        -------
            List[str]: A list of possible methods.
        """
        return CHAR_ACTIONS

    def __delete(self) -> str:
        """Deletes a random character.

        Returns:
            str: Empty string.
        """
        return ""

    def __insert(self, char: str, vocab: List[str]) -> str:
        """Inserts a random character.

        Args:
            char (str): A symbol from the word.
            vocab (List[str]): ...

        Returns:
            str: A symbol + new symbol.
        """
        return char + np.random.choice(vocab)

    def __multiply(self, char: str) -> str:
        """Repeats a randomly selected character.

        Args:
            char (str): A symbol from the word.

        Returns:
            str: A symbol from the word matmul n times.
        """
        if char in [" ", ",", ".", "?", "!", "-"]:
            return char
        else:
            n = np.random.randint(1, self.mult_num)
            return char * n

    def _transform(self, array: List[str]) -> List[str]:
        transformed_array = []
        for text in array:
            typo_text_arr = list(text)
            aug_idxs = self._aug_indexing(typo_text_arr, self.unit_prob, clip=True)

            for idx in aug_idxs:
                if self.action == "delete":
                    typo_text_arr[idx] = self.__delete()
                elif self.action == "insert":
                    vocab = list(set(text))
                    typo_text_arr[idx] = self.__insert(typo_text_arr[idx], vocab)
                elif self.action == "multiply":
                    typo_text_arr[idx] = self.__multiply(typo_text_arr[idx])
                elif self.action == "swap":
                    sw = max(0, idx - 1)
                    typo_text_arr[sw], typo_text_arr[idx] = (
                        typo_text_arr[idx],
                        typo_text_arr[sw],
                    )
                else:
                    raise NameError(
                        """These type of augmentation is not available, please try TypoAug.actions_list() to see
                    available augmentations"""
                    )
            text = ''.join(typo_text_arr)
            transformed_array.append(text)

        return transformed_array
