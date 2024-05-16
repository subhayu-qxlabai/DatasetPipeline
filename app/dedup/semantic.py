from typing import Literal
from functools import cached_property

from datasets import Dataset
from langchain.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

from .base import BaseConfig, BaseDedup
from ..helpers.utils import hash_uuid
from ..helpers.embeddings import Embeddings


class SemanticDedupConfig(BaseConfig):
    threshold: float = 0.8
    dedup_column: str = "messages"
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
        if self.config.dedup_column not in self.dataset.column_names:
            return False
        _col = self.dataset[self.config.dedup_column]
        if not all(isinstance(x, str) for x in _col):
            return False
        if len(_col) == len(set(_col)):
            return False
        return True
    
    def _dedup(self):
        if not self.can_be_deduped:
            return self.dataset
        embeddings = self.embeddings.embed_documents(texts)
        texts = self.dataset[self.config.dedup_column]
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
        
        try:
            dataset = self.dataset.remove_columns("_id")
        except:
            dataset = self.dataset
        
        dataset: Dataset = (
            dataset
            .add_column("_embeddings", embeddings)
            .add_column("_id", ids)
        )
        
        # TODO: Add deduplication logic here
        
        return dataset
        
