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


    def load(self) -> DatasetDict:
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
