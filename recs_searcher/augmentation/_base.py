"""
Алгоритмы для Pipeline валидации.
"""


from typing import List, Literal, Optional
from ..base import BaseTransformation
from augmentex import WordAug, CharAug


class AugmentexWordWrapperAugmentation(BaseTransformation):
    """
    Аугментация на уровне слов.
    Основано на : https://github.com/ai-forever/augmentex

    Параметры
        ----------
        actions: str
            Действия на словом.
            `replace` - заменить случайное слово его неправильным аналогом;
            `delete` - удалить случайное слово;
            `swap` - поменять местами два случайных слова;
            `stopword` - добавить случайные слова из стоп-листа;
            `split` - добавить пробелы между буквами к слову;
            `reverse` - изменить регистр первой буквы случайного слова;
            `text2emoji` - заменить слово соответствующим эмодзи;
            `ngram` - заменить ngram в слове ошибочными.

        min_aug: Optional[int]
            Минимальное количество аугментаций.

        max_aug: Optional[int]
            Максимальное количество аугментаций.

        unit_prob: Optional[float]
            Процент от фразы, к которой будут применена аугментация.
    """

    def __init__(
        self,
        action: Literal['replace', 'delete', 'swap', 'stopword', 'split', 'reverse', 'text2emoji', 'ngram'],
        min_aug: Optional[int] = 1,
        max_aug: Optional[int] = 5,
        unit_prob: Optional[float] = 0.3,
        lang: Optional[Literal['rus', 'end']] = 'rus',
        platform: Optional[Literal['pc', 'mobile']] = 'pc',
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        self._action = action
        self._augmentation = WordAug(
            unit_prob=unit_prob,
            min_aug=min_aug,
            max_aug=max_aug,
            lang=lang,
            platform=platform,
            random_seed=seed,
        )

    def _transform(self, array: List[str]) -> List[str]:
        transformed_array = []
        for text in array:
            text = self._augmentation.augment(text=text, action=self._action)
            transformed_array.append(text)
        return transformed_array


class AugmentexCharWrapperAugmentation(BaseTransformation):
    """
    Аугментация на уровне букв.
    Основано на : https://github.com/ai-forever/augmentex

    Параметры
        ----------
        action : str
            Действия на символом.
            `shift` - поменять местами верхний/нижний регистр в строке;
            `orfo` - заменить правильные символы их распространенными неправильными аналогами;
            `typo` - заменить правильные символы, как если бы они были набраны с ошибкой на клавиатуре;
            `delete` - удалить случайный символ;
            `insert` - вставить случайный символ;
            `multiply` - умножение случайного символа;
            `swap` - поменять местами два соседних символа.

        min_aug: Optional[int]
            Минимальное количество аугментаций.

        max_aug: Optional[int]
            Максимальное количество аугментаций.

        mult_num: Optional[int]
            Максимальное количество повторений символов (для action='multiply').

        unit_prob: Optional[float]
            Процент от фразы, к которой будут применена аугментация.
    """

    def __init__(
        self,
        action: Literal['shift', 'orfo', 'typo', 'delete', 'insert', 'multiply', 'swap'],
        min_aug: Optional[int] = 1,
        max_aug: Optional[int] = 5,
        mult_num: Optional[int] = 2,
        unit_prob: Optional[float] = 0.3,
        lang: Optional[Literal['rus', 'end']] = 'rus',
        platform: Optional[Literal['pc', 'mobile']] = 'pc',
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        self._action = action
        self._augmentation = CharAug(
            unit_prob=unit_prob,
            min_aug=min_aug,
            max_aug=max_aug,
            mult_num=mult_num,
            lang=lang,
            platform=platform,
            random_seed=seed,
        )

    def _transform(self, array: List[str]) -> List[str]:
        transformed_array = []
        for text in array:
            text = self._augmentation.augment(text=text, action=self._action)
            transformed_array.append(text)
        return transformed_array
