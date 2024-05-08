from datasets import Dataset

from .base import BaseAnalyzer, BaseConfig


class OutputAnalyzer(BaseAnalyzer):
    def __init__(self, dataset: Dataset, config: BaseConfig = BaseConfig()):
        super().__init__(dataset, config)

    def _analyze(self) -> Dataset:
        return self.dataset
