"""
This module provides the main entry point for the deduplication functionality of the DatasetPipeline.

It contains the main classes and functions for deduplication, including:

- `BaseDedup` and `BaseDedupConfig`: Base classes for deduplication functionality.
- `BaseModel`: Base class for deduplication models.
- `SemanticDedup` and `SemanticDedupConfig`: Classes and configuration for semantic deduplication.
- `Dedup` and `DedupConfig`: Class for deduplicating datasets and configuration.

To use this module, import the `Dedup` class and create a `DedupConfig` object. For example:
"""
from functools import partial
from dataclasses import dataclass

from datasets import Dataset, DatasetDict
from pydantic import Field

from .base import BaseDedup, BaseDedupConfig, BaseModel
from .semantic import SemanticDedup, SemanticDedupConfig


class DedupConfig(BaseModel):
    semantic: SemanticDedupConfig | None = Field(default=SemanticDedupConfig(), description="Configuration for semantic deduplication.")


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
    