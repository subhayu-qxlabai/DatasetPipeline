"""
This module provides a `Job` class for performing data processing jobs.

The `Job` class takes a `JobConfig` object as input, which specifies the configuration for the job, including the data loading, formatting, deduplication, analysis, and saving steps.

The `Job` class has the following methods:

- `load()`: Loads the data using a `Loader` object based on the configuration in `JobConfig`.
- `format()`: Formats the data using a `Format` object based on the configuration in `JobConfig`.
- `analyze()`: Analyzes the data using an `Analyzer` object based on the configuration in `JobConfig`.
- `dedup()`: Deduplicates the data using a `Dedup` object based on the configuration in `JobConfig`.
- `save()`: Saves the data using a `Saver` object based on the configuration in `JobConfig`.
- `_format_dedup_and_analyze()`: Applies the format, deduplication, and analysis steps to a dataset and returns the result.
- `run()`: Runs the job by loading the data, applying the format, deduplication, and analysis steps, and saving the results if specified in the configuration.
- `__call__()`: Calls the `run()` method to execute the job.

The module also imports various modules and classes from other parts of the codebase, including `BaseModel` from `.models`, `Loader`, `LoaderConfig`, `Format`, `FormatConfig`, `Dedup`, `DedupConfig`, `Analyzer`, `AnalyzerConfig`, `Saver`, `SaverConfig`, `run_parallel_exec`, and `LOGGER` from `.helpers`.

Example usage:

```python
from .job import Job, JobConfig

# Create a JobConfig object with the desired configuration
config = JobConfig(
    load=LoaderConfig(...),
    format=FormatConfig(...),
    deduplicate=DedupConfig(...),
    analyze=AnalyzerConfig(...),
    save=SaverConfig(...)
)

# Create a Job object with the config
job = Job(config)

# Run the job
results = job()
```
"""

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
    """
    Represents a job in the pipeline.

    Attributes:
        config (JobConfig): The configuration for the job.

    Methods:
        load(self) -> Dataset: Loads the dataset using the specified loader.
        format(self, dataset: Dataset, textualize: bool = False) -> Dataset: Formats the dataset using the specified format.
        analyze(self, dataset: Dataset) -> Dataset: Analyzes the dataset using the specified analyzer.
        dedup(self, dataset: Dataset) -> Dataset: Deduplicates the dataset using the specified deduplicator.
        save(self, name_and_dataset: tuple[str, Dataset]) -> None: Saves the dataset using the specified saver.
        _format_dedup_and_analyze(self, name_and_dataset: tuple[str, Dataset], textualize: bool = False) -> Dataset: Formats, deduplicates, and analyzes the dataset.
        run(self) -> list[Path | Dataset]: Runs the job and returns the list of paths or datasets generated.
        __call__(self) -> list[Path | Dataset]: Calls the `run` method.
    """
    config: JobConfig

    # ... rest of the code ...
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
