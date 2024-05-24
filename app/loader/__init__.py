"""
This module contains the `__init__.py` file for the `loader` package.

It imports and exports the necessary classes and functions for loading datasets.

Submodules:
- `base`: Contains the base classes for creating loader classes.
- `huggingface`: Contains the classes for loading datasets from HuggingFace Hub.
- `local_file`: Contains the classes for loading datasets from the local file system.

Classes:
- `LoaderConfig`: The configuration for loading datasets.
- `Loader`: The class for loading datasets.
"""
from dataclasses import dataclass

from datasets import Dataset, DatasetDict
from pydantic import Field, model_validator

from .base import BaseLoader, BaseConfig, BaseModel
from .huggingface import HFLoader, HFLoaderConfig
from .local_file import LocalFileLoader, LocalFileLoaderConfig


class LoaderConfig(BaseModel):
    huggingface: HFLoaderConfig | None = Field(default=None, description="Configurations for loading datasets from HuggingFace Hub.")
    local: LocalFileLoaderConfig | None = Field(default=None, description="Configurations for loading datasets from local file system.")
    
    @model_validator(mode='after')
    def verify_square(self):
        if self.huggingface is None and self.local is None:
            raise ValueError('At least one of `huggingface` or `local` loaders must be specified.')
        return self
    

@dataclass
class Loader:
    config: LoaderConfig
    
    def __post_init__(self):
        self.loaders: list[BaseLoader] = [
            *[HFLoader(self.config.huggingface)],
            *[LocalFileLoader(self.config.local)],
        ]

    def load(self) -> DatasetDict:
        path_dataset_map = {}
        for loader in self.loaders:
            dst = loader.load()
            if dst is None:
                continue
            path_dataset_map.update(
                dst if isinstance(dst, DatasetDict) else DatasetDict({0: dst})
            )
        
        return DatasetDict(path_dataset_map)
    