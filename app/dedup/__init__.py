from functools import partial
from dataclasses import dataclass

from datasets import Dataset, DatasetDict

from .base import BaseDedup, BaseConfig, BaseModel
from .semantic import SemanticDedup, SemanticDedupConfig


class DedupConfig(BaseModel):
    semantic: SemanticDedupConfig | None = SemanticDedupConfig()


@dataclass
class Dedup:
    dataset: Dataset
    config: DedupConfig | None = DedupConfig()
    
    def __post_init__(self):
        self.semantic_dedup: type[SemanticDedup] = partial(SemanticDedup, config=self.config.semantic)
    
    @property
    def _base_chain(self) -> SemanticDedupConfig:
        return (
            self.semantic_dedup(self.dataset)
        )
    
    def dedup(self) -> DatasetDict:
        """
        Deduplicate the dataset.

        Returns:
            Dataset: The deduplied dataset.
        """
        if self.config is None:
            return self.dataset
        return self.semantic_dedup(self.dataset).dedup()
    