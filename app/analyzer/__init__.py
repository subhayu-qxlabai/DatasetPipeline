"""
This module contains the `__init__.py` file for the `analyzer` package.

It imports and exports the necessary classes and functions for analyzing datasets.

Submodules:
- `base`: Contains the base classes for analyzing datasets.
- `quality`: Contains the classes for analyzing the quality of text data.
- `output`: Contains the classes for analyzing the output of text data.

Classes:
- `BaseAnalyzer`: The base class for analyzing datasets.
- `QualityAnalyzer`: The class for analyzing the quality of text data.
- `OutputAnalyzer`: The class for analyzing the output of text data.
"""
from functools import partial
from dataclasses import dataclass

from datasets import Dataset
from pydantic import Field

from .base import BaseAnalyzer, BaseAnalyzerConfig, BaseModel
from .quality import QualityAnalyzer, QualityAnalyzerConfig, Message, Messages, TEXT_QUALITY_EXAMPLE_MESSAGES
from .output import OutputAnalyzer, OutputAnalyzerConfig


class AnalyzerConfig(BaseModel):
    quality: QualityAnalyzerConfig | None = Field(default=QualityAnalyzerConfig(), description="Configuration for qualitative analysis.")
    

@dataclass
class Analyzer:
    dataset: Dataset
    config: AnalyzerConfig | None = AnalyzerConfig()
    
    def __post_init__(self):
        self.quality_analyzer: type[QualityAnalyzer] = partial(QualityAnalyzer, config=self.config.quality)
    
    @property
    def _base_chain(self) -> QualityAnalyzer:
        return (
            self.quality_analyzer(self.dataset)
        )
    
    def analyze(self) -> Dataset:
        """
        Standardizes the dataset by applying a series of data transformations.

        Returns:
            Dataset: The analyzed dataset.
        """
        if self.config is None:
            return self.dataset
        return (self._base_chain | OutputAnalyzer).analyze()
    