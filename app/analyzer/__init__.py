from functools import partial
from dataclasses import dataclass

from datasets import Dataset
from pydantic import Field

from .base import BaseAnalyzer, BaseConfig, BaseModel
from .quality import QualityAnalyzer, QualityConfig, Message, Messages, TEXT_QUALITY_EXAMPLE_MESSAGES
from .output import OutputAnalyzer


class AnalyzerConfig(BaseModel):
    quality: QualityConfig | None = Field(default=QualityConfig(), description="Configuration for qualitative analysis.")
    

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
    