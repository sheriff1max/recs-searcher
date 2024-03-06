"""
Модели для создания эмбеддингов.
"""


from typing import Iterable, List, Callable, Union, Type, Dict, Optional

from ..base import BaseEmbedding, BaseTransformation
from ..dataset import SentenceTransformerDataset
from ..utils import concat_embeddings

from sklearn.feature_extraction.text import CountVectorizer

from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import SentenceEvaluator

from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim import AdamW
from torch import nn

import numpy as np


class CountVectorizerWrapperEmbedding(BaseEmbedding):
    """Мешок слов/символов N-грам."""

    def __init__(
            self,
            analyzer="word",  # 'word' | 'char' | 'char_wb'
            ngram_range=(1, 1),

            max_df=1.0,
            min_df=1,
            input="content",
            encoding="utf-8",
            decode_error="strict",
            strip_accents=None,
            lowercase=False,
            preprocessor=None,
            tokenizer=None,
            stop_words=None,
            token_pattern=r"(?u)\b\w\w+\b",
            max_features=None,
            vocabulary=None,
            binary=False,
            dtype=np.float64,
    ):
        self._model = CountVectorizer(
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            analyzer=analyzer,
            stop_words=stop_words,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            binary=binary,
            dtype=dtype,
        )

    def fit(self, array: Iterable[str]) -> object:
        self._model.fit(array)
        return self

    def transform(self, array: Iterable[str]) -> np.ndarray:
        array = self._model.transform(array).toarray()
        return array


class SentenceTransformerWrapperEmbedding(BaseEmbedding):
    """
    Эмбеддинги на основе трансформеров из Sentence-transformers.

    Примечание
    ----------
    Предобученные модели брать с этих источников:
        1. https://sbert.net/docs/pretrained_models.html
        2. https://huggingface.co/models?library=sentence-transformers
    """

    def __init__(
            self,
            name_model: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            train_loss: Type[nn.Module] = losses.MultipleNegativesRankingLoss,
            augmentation_transform: Union[None, List[BaseTransformation]] = None,

            shuffle=True,
            batch_size=32,

            evaluator: SentenceEvaluator = None,
            epochs: int = 4,
            steps_per_epoch = None,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = AdamW,
            optimizer_params : Dict[str, object]= {'lr': 2e-2},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            show_progress_bar: bool = True,
            checkpoint_path: str = None,
            checkpoint_save_steps: int = 500,
            checkpoint_save_total_limit: int = 0,
    ):
        self._model = SentenceTransformer(name_model)
        self._train_loss = train_loss(model=self._model)

        self._augmentation_transform = augmentation_transform
        self._shuffle = shuffle
        self._batch_size = batch_size
        self._evaluator = evaluator
        self._epochs = epochs
        self._steps_per_epoch = steps_per_epoch
        self._scheduler = scheduler
        self._warmup_steps = warmup_steps
        self._optimizer_class = optimizer_class
        self._optimizer_params = optimizer_params
        self._weight_decay = weight_decay
        self._evaluation_steps = evaluation_steps
        self._output_path = output_path
        self._save_best_model = save_best_model
        self._max_grad_norm = max_grad_norm
        self._use_amp = use_amp
        self._callback = callback
        self._show_progress_bar = show_progress_bar
        self._checkpoint_path = checkpoint_path
        self._checkpoint_save_steps = checkpoint_save_steps
        self._checkpoint_save_total_limit = checkpoint_save_total_limit

    def fit(self, array: Iterable[str]) -> object:
        train_dataset = SentenceTransformerDataset(array, self._augmentation_transform)

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=self._shuffle,
            batch_size=self._batch_size
        )

        self._model.fit(
            train_objectives=[(train_dataloader, self._train_loss)],

            evaluator=self._evaluator,
            epochs=self._epochs,
            steps_per_epoch=self._steps_per_epoch,
            scheduler=self._scheduler,
            warmup_steps=self._warmup_steps,
            optimizer_class=self._optimizer_class,
            optimizer_params=self._optimizer_params,
            weight_decay=self._weight_decay,
            evaluation_steps=self._evaluation_steps,
            output_path=self._output_path,
            save_best_model=self._save_best_model,
            max_grad_norm=self._max_grad_norm,
            use_amp=self._use_amp,
            callback=self._callback,
            show_progress_bar=self._show_progress_bar,
            checkpoint_path=self._checkpoint_path,
            checkpoint_save_steps=self._checkpoint_save_steps,
            checkpoint_save_total_limit=self._checkpoint_save_total_limit
        )
        return self

    def transform(self, array: Iterable[str]) -> np.ndarray:
        array = self._model.encode(array)
        return np.array(array)


class EnsembleWrapperEmbedding(BaseEmbedding):
    """Ансамбль из vectorizer, объединяющий их результаты."""

    def __init__(
            self,
            models: List[BaseEmbedding],
            weights: Optional[List[float]] = None,
    ):
        self._models = models
        self._weights = weights

    def fit(self, array: Iterable[str]) -> object:
        for model in self._models:
            model.fit(array)
        return self

    def transform(self, array: Iterable[str]) -> np.ndarray:
        embeddings = []
        for model in self._models:
            embedding = model.transform(array)
            embeddings.append(embedding)

        array = concat_embeddings(embeddings, self._weights)
        return array
