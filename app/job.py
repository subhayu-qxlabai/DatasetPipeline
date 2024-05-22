from pathlib import Path
from itertools import chain
from datasets import Dataset, DatasetDict

from .models import BaseModel
from .loader import Loader, LoaderConfig
from .format import Format, FormatConfig
from .dedup import Dedup, DedupConfig
from .analyzer import Analyzer, AnalyzerConfig
from .saver import Saver, SaverConfig
from .helpers import run_parallel_exec
from .helpers.logger import LOGGER


class JobConfig(BaseModel):
    load: LoaderConfig
    format: FormatConfig | None = FormatConfig()
    deduplicate: DedupConfig | None = DedupConfig()
    analyze: AnalyzerConfig | None = AnalyzerConfig()
    save: SaverConfig | None = SaverConfig()


class Job(BaseModel):
    config: JobConfig

    def load(self):
        loader = Loader(self.config.load)
        return loader.load()

    def format(self, dataset: Dataset, textualize: bool = False):
        if self.config.format is None:
            return dataset
        format = Format(dataset, self.config.format)
        return format.format(textualize=textualize)

    def analyze(self, dataset: Dataset):
        if self.config.analyze is None:
            return dataset
        analyzer = Analyzer(dataset, self.config.analyze)
        return analyzer.analyze()

    def dedup(self, dataset: Dataset):
        if self.config.deduplicate is None:
            return dataset
        dedup = Dedup(dataset, self.config.deduplicate)
        return dedup.dedup()

    def save(self, name_and_dataset: tuple[str, Dataset]):
        if self.config.save is None:
            return
        name, dataset = name_and_dataset
        self.config.save.local.filename = name
        saver = Saver(dataset, self.config.save)
        return saver.save()

    def _format_dedup_and_analyze(
        self, name_and_dataset: tuple[str, Dataset], textualize: bool = False
    ) -> list[tuple[str, Dataset]]:
        name, dataset = name_and_dataset
        dataset = self.format(dataset, textualize)
        dataset = self.dedup(dataset)
        if isinstance(dataset, DatasetDict):
            name_dataset_map: list[tuple[str, Dataset]] = [
                (f"{name}-{split}", dst) for split, dst in dataset.items()
            ]
        else:
            name_dataset_map: list[tuple[str, Dataset]] = [(name, dataset)]
        name_dataset_map = [
            (name, (self.analyze(d) if "duplicates" not in name else d))
            for name, d in name_dataset_map
        ]
        return name_dataset_map

    def run(self) -> list[Path | Dataset]:
        data: dict[str, Dataset] = self.load()
        data: list[
            tuple[tuple[str, Dataset], list[tuple[str, Dataset]]] | Exception
        ] = run_parallel_exec(self._format_dedup_and_analyze, list(data.items()), True)
        errors = [
            (name, response)
            for (name, _), response in data
            if isinstance(response, Exception)
        ]
        if errors:
            for name, error in errors:
                LOGGER.error(f"Error during processing {name!r}: {error}")

        data: list[list[tuple[str, Dataset]]] = [
            _data for (_, _), _data in data if not isinstance(_data, Exception)
        ]
        data: list[tuple[str, Dataset]] = list(chain(*data))
        if self.config.save is not None:
            data = run_parallel_exec(self.save, data)
        return data

    def __call__(self) -> list[Path | Dataset]:
        return self.run()
