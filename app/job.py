from pathlib import Path
from typing import ClassVar
from datasets import Dataset

from .models import BaseModel
from .loader import Loader, LoaderConfig
from .format import Format, FormatConfig
from .analyzer import Analyzer, AnalyzerConfig
from .saver import Saver, SaverConfig
from .helpers.utils import run_parallel_exec


class JobConfig(BaseModel):
    loader: LoaderConfig
    format: FormatConfig | None = FormatConfig()
    analyzer: AnalyzerConfig | None = AnalyzerConfig()
    saver: SaverConfig | None = SaverConfig()


class Job(BaseModel):
    config: JobConfig
    
    completed: ClassVar[bool] = False
    
    def load(self):
        loader = Loader(self.config.loader)
        return loader.load()

    def format(self, dataset: Dataset, textualize: bool = False):
        if self.config.format is None:
            return dataset
        format = Format(dataset, self.config.format)
        return format.format(textualize=textualize)

    def analyze(self, dataset: Dataset):
        if self.config.analyzer is None:
            return dataset
        analyzer = Analyzer(dataset, self.config.analyzer)
        return analyzer.analyze()
    
    def save(self, name_and_dataset: tuple[str, Dataset]):
        if self.config.saver is None:
            return
        name, dataset = name_and_dataset
        self.config.saver.local.filename = name
        saver = Saver(dataset, self.config.saver)
        return saver.save()
    
    def _format_and_analyze(self, name_and_dataset: tuple[str, Dataset], textualize: bool = False):
        name, dataset = name_and_dataset
        dataset = self.format(dataset, textualize)
        dataset = self.analyze(dataset)
        return (name, dataset)

    def run(self) -> list[Path | Dataset]:
        data: dict[str, Dataset] = self.load()
        data: list[tuple[tuple[str, Dataset], ...]] = run_parallel_exec(
            self._format_and_analyze, list(data.items()), True
        )
        data: list[tuple[str, Dataset]] = [(name, dst) for _, (name, dst) in data]
        if self.config.saver is not None:
            data = run_parallel_exec(self.save, data)
        self.completed = True
        return data

    def __call__(self) -> list[Path | Dataset]:
        return self.run()
    