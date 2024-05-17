from typing import Any, Literal
from functools import cached_property

import pandas as pd
from langchain_community.vectorstores import FAISS
from datasets import Dataset, DatasetDict
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

from .base import BaseConfig, BaseDedup
from ..helpers.utils import hash_uuid
from ..helpers.embeddings import Embeddings


class SemanticDedupConfig(BaseConfig):
    column: str = "messages"
    threshold: float = 0.2
    cache_embeddings: bool = False
    embeddings_model: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    device: Literal['mps', 'cuda', 'npu', 'hpu', 'cpu'] | None = None
    multi_process: bool = False
    show_progress: bool = True


class SemanticDedup(BaseDedup):
    def __init__(self, dataset: Dataset, config = SemanticDedupConfig()):
        super().__init__(dataset, config)
        self.config: SemanticDedupConfig
        self.embeddings = Embeddings(
            model=HuggingFaceEmbeddings(
                model_name=self.config.embeddings_model, 
                model_kwargs={'device': self.config.device},
                multi_process=self.config.multi_process,
                show_progress=self.config.show_progress,
            ),
            use_cache=self.config.cache_embeddings
        )

    @cached_property
    def can_be_deduped(self):
        if not isinstance(self.dataset, Dataset) or self.dataset.shape[0] == 0:
            return False
        if self.config.column not in self.dataset.column_names:
            return False
        _col = self.dataset[self.config.column]
        if not all(isinstance(x, str) for x in _col):
            return False
        if len(_col) == len(set(_col)):
            return False
        return True

    def _dedup(self) -> DatasetDict:
        if not self.can_be_deduped:
            return self.dataset

        texts = self.dataset[self.config.column]
        embeddings = self.embeddings.embed_documents(texts)
        text_embeddings: list[tuple[str, list[float]]] = list(zip(
            texts, embeddings
        ))
        ids = [hash_uuid(x).hex for x in texts]
        vdb = FAISS.from_embeddings(
            text_embeddings, 
            self.embeddings.model, 
            metadatas=[{"id": _id} for _id in ids],
            ids=ids,
        )

        embeddings_col = "_embeddings"
        score_col = "score"
        match_col = "match"

        dataset: Dataset = self.dataset.add_column(embeddings_col, embeddings)
        del texts, embeddings, text_embeddings, ids

        def add_score_and_match(row: dict[str, Any]) -> dict[str, str | float]:
            doc, score = vdb.similarity_search_with_score_by_vector(row[embeddings_col], k=2)[-1]
            return {match_col: doc.page_content, score_col: score}

        df: pd.DataFrame = dataset.map(add_score_and_match).to_pandas()

        del vdb, dataset

        # normalize the scores between 0 and 1
        scores = df[score_col]
        df[score_col] = (scores - scores.min()) / (scores.max() - scores.min())

        threshold = self.config.threshold

        deduped_df = (
            pd.concat(
                [
                    df.query(f"{score_col} < {threshold}").drop_duplicates(score_col),
                    df.query(f"{score_col} >= {threshold}"),
                ]
            )
            .drop(columns=[embeddings_col, match_col, score_col])
            .sort_index()
        )

        dd = DatasetDict(
            deduplicated=Dataset.from_pandas(deduped_df),
            duplicates=Dataset.from_pandas(
                self.dataset.to_pandas().query(
                    f"{self.config.column} not in @deduped_df.{self.config.column}"
                )
            ),
        )

        return dd
