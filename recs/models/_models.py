"""
Модели для создания эмбеддингов
"""


from typing import Iterable, List, Callable, Union, Type, Dict
import pickle
import os

from base import BaseModel, BaseTransformation, BaseDataset

from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models.fasttext import FastText
from gensim.models.fasttext_inner import MAX_WORDS_IN_BATCH

from sentence_transformers import SentenceTransformer, models, InputExample, losses
from sentence_transformers.evaluation import SentenceEvaluator

from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim import AdamW
from torch import nn

import numpy as np


class TfidfWrapperModel(BaseModel):
    """"""

    def __init__(
            self,
            *,
            input="content",
            encoding="utf-8",
            decode_error="strict",
            strip_accents=None,
            lowercase=True,
            preprocessor=None,
            tokenizer=None,
            analyzer="word",
            stop_words=None,
            token_pattern=r"(?u)\b\w\w+\b",
            ngram_range=(1, 1),
            max_df=1.0,
            min_df=1,
            max_features=None,
            vocabulary=None,
            binary=False,
            dtype=np.float64,
            norm="l2",
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False,
    ):
        self._model = TfidfVectorizer(
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
            norm=norm,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf,
        )

    def fit(self, array: Iterable[str]) -> object:
        self._model.fit(array)
        return self

    def transform(self, array: Iterable[str]) -> np.ndarray:
        array = self._model.transform(array).toarray()
        return array


class FastTextWrapperModel(BaseModel):
    """"""

    def __init__(
            self,
            sg=0,
            hs=0,
            vector_size=100,
            alpha=0.025,
            window=5,
            min_count=5,
            max_vocab_size=None,
            word_ngrams=1,
            sample=1e-3,
            seed=1,
            workers=3,
            min_alpha=0.0001,
            negative=5,
            ns_exponent=0.75,
            cbow_mean=1,
            hashfxn=hash,
            epochs=5,
            null_word=0,
            min_n=3,
            max_n=6,
            sorted_vocab=1,
            bucket=2000000,
            trim_rule=None,
            batch_words=MAX_WORDS_IN_BATCH,
            callbacks=(),
            max_final_vocab=None,
            shrink_windows=True,
    ):
        self._model = None

        self._sg=sg
        self._hs=hs
        self._vector_size=vector_size
        self._alpha=alpha
        self._window=window
        self._min_count=min_count
        self._max_vocab_size=max_vocab_size
        self._word_ngrams=word_ngrams
        self._sample=sample
        self._seed=seed
        self._workers=workers
        self._min_alpha=min_alpha
        self._negative=negative
        self._ns_exponent=ns_exponent
        self._cbow_mean=cbow_mean
        self._hashfxn=hashfxn
        self._epochs=epochs
        self._null_word=null_word
        self._min_n=min_n
        self._max_n=max_n
        self._sorted_vocab=sorted_vocab
        self._bucket=bucket
        self._trim_rule=trim_rule
        self._batch_words=batch_words
        self._callbacks=callbacks
        self._max_final_vocab=max_final_vocab
        self._shrink_windows=shrink_windows

    def fit(self, array: Iterable[str]) -> object:
        
        # TODO:
        array_tokenized = [text.split() for text in array]

        self._model = FastText(
            array_tokenized,

            sg=self._sg,
            hs=self._hs,
            vector_size=self._vector_size,
            alpha=self._alpha,
            window=self._window,
            min_count=self._min_count,
            max_vocab_size=self._max_vocab_size,
            word_ngrams=self._word_ngrams,
            sample=self._sample,
            seed=self._seed,
            workers=self._workers,
            min_alpha=self._min_alpha,
            negative=self._negative,
            ns_exponent=self._ns_exponent,
            cbow_mean=self._cbow_mean,
            hashfxn=self._hashfxn,
            epochs=self._epochs,
            null_word=self._null_word,
            min_n=self._min_n,
            max_n=self._max_n,
            sorted_vocab=self._sorted_vocab,
            bucket=self._bucket,
            trim_rule=self._trim_rule,
            batch_words=self._batch_words,
            callbacks=self._callbacks,
            max_final_vocab=self._max_final_vocab,
            shrink_windows=self._shrink_windows,            
        )

        return self

    def transform(self, array: Iterable[str]) -> np.ndarray:

        # TODO: заменить на норм токенизацию
        array = [text.split() for text in array]

        array_list = []

        for text_tokens in array:
            vector = self._model.wv.get_sentence_vector(text_tokens)
            array_list.append(vector)

        return np.array(array_list)


class SentenceTransformerWrapperModel(BaseModel):
    """

    Примечание
    ----------
    Предобученные модели брать с этих источников:
        1. https://sbert.net/docs/pretrained_models.html
        2. https://huggingface.co/models?library=sentence-transformers
    """

    def __init__(
            self,
            name_model: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            train_loss: Type[nn.Module] = losses.MultipleNegativesRankingLoss
    ):
        self._model = SentenceTransformer(name_model)
        self._train_loss = train_loss(model=self._model)

    def fit(
            self,
            array: Iterable[str],
            augmentation_transform: Union[None, List[BaseTransformation]] = None,

            shuffle=True,
            batch_size=32,

            evaluator: SentenceEvaluator = None,
            epochs: int = 1,
            steps_per_epoch = None,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = AdamW,
            optimizer_params : Dict[str, object]= {'lr': 2e-5},
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
            checkpoint_save_total_limit: int = 0
    ) -> object:
        """"""

        train_dataset = BaseDataset(array, augmentation_transform)
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=shuffle,
            batch_size=batch_size
        )

        self._model.fit(
            train_objectives=[(train_dataloader, self._train_loss)],

            evaluator=evaluator,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            scheduler=scheduler,
            warmup_steps=warmup_steps,
            optimizer_class=optimizer_class,
            optimizer_params=optimizer_params,
            weight_decay=weight_decay,
            evaluation_steps=evaluation_steps,
            output_path=output_path,
            save_best_model=save_best_model,
            max_grad_norm=max_grad_norm,
            use_amp=use_amp,
            callback=callback,
            show_progress_bar=show_progress_bar,
            checkpoint_path=checkpoint_path,
            checkpoint_save_steps=checkpoint_save_steps,
            checkpoint_save_total_limit=checkpoint_save_total_limit
        )
        return self

    def transform(self, array: Iterable[str]) -> np.ndarray:

        array = self._model.encode(array)
        return np.array(array)

    def load(self, filename: str) -> object:
        """"""

    def save(self, filename: str) -> object:
        """"""
