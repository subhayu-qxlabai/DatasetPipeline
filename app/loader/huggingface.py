"""
Module for loading datasets from the Hugging Face dataset repository.

The module defines a configuration class `HFLoaderConfig` and a loader class `HFLoader`.

The `HFLoaderConfig` class is a subclass of `BaseConfig` and defines the configuration options for loading datasets from Hugging Face. It has the following attributes:
- `path`: Repository path to the dataset. Required.
- `name`: Name of the dataset configuration. Defaults to `None`.
- `token`: Hugging Face API token. Defaults to `None`.
- `take_rows`: Number of rows to take from the dataset. Defaults to `None`.
- `split`: Split to take from the dataset. Defaults to `None`.
- `cache_dir`: Directory to cache the dataset. Defaults to `None`.

The `HFLoader` class is a subclass of `BaseLoader` and is responsible for loading datasets from Hugging Face. It takes an instance of `HFLoaderConfig` as input in its constructor. The `load_or_download` method downloads the dataset from the specified repository path using the `load_dataset` function from the `datasets` library. The `_load` method further processes the loaded dataset by creating a dictionary mapping of dataset paths and splits, and returns a `DatasetDict` object containing the processed datasets.

Usage example:
```python
from huggingface_loader import HFLoaderConfig, HFLoader

# Create an instance of HFLoaderConfig with the desired configuration options
config = HFLoaderConfig(
    path="path/to/dataset",
    name="dataset_name",
    token="your_huggingface_api_token",
    take_rows=100,
    split="train",
    cache_dir="path/to/cache_directory"
)

# Create an instance of HFLoader with the config
loader = HFLoader(config)

# Load the datasets
datasets = loader.load()
```
"""

from pydantic import Field
from datasets import Dataset, DatasetDict, load_dataset

from .base import BaseLoader, BaseConfig
from ..helpers import LOGGER


class HFLoaderConfig(BaseConfig):
    path: str = Field(description="Repository path to the dataset. Required.")
    name: str | None = Field(default=None, description="Name of the dataset configuration. Defaults to `null`.")
    token: str | None = Field(default=None, description="Hugging Face API token. Defaults to `null`.")
    take_rows: int | None = Field(default=None, description="Number of rows to take from the dataset. Defaults to `null`.")
    split: str | None = Field(default=None, description="Split to take from the dataset. Defaults to `null`.")
    cache_dir: str | None = Field(default=None, description="Directory to cache the dataset. Defaults to `null`.")
    

class HFLoader(BaseLoader):
    def __init__(self, config: HFLoaderConfig):
        super().__init__(config)
        self.config: HFLoaderConfig

    def load_or_download(self) -> Dataset | DatasetDict:
        LOGGER.info(f"Downloading dataset from {self.config.path}")
        return load_dataset(
            path=self.config.path,
            name=self.config.name,
            token=self.config.token,
            split=self.config.split,
            cache_dir=self.config.cache_dir
        )


    def _load(self) -> DatasetDict:
        dsts = self.load_or_download()
        path_datasets_map = [
            (path, dst if isinstance(dst, DatasetDict) else {0: dst})
            for path, dst in [(self.config.path, dsts)]
        ]
        path_datasets_map = {
            (f"{path}-{split}" if isinstance(split, str) else path): (
                dst.take(self.config.take_rows) if self.config.take_rows else dst
            )
            for path, ddict in path_datasets_map
            for split, dst in ddict.items()
        }
        return DatasetDict(path_datasets_map)
