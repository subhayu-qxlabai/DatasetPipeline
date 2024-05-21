from dataclasses import dataclass

from datasets import Dataset, DatasetDict
from pydantic import Field

from .base import BaseLoader, BaseConfig, BaseModel
from .huggingface import HFLoader, HFLoaderConfig
from .local_file import LocalFileLoader, LocalFileLoaderConfig


class LoaderConfig(BaseModel):
    huggingface: list[HFLoaderConfig] = Field(default_factory=list, description="Configurations for loading datasets from HuggingFace Hub.")
    local: list[LocalFileLoaderConfig] = Field(default_factory=list, description="Configurations for loading datasets from local file system.")
    

@dataclass
class Loader:
    config: LoaderConfig
    
    def __post_init__(self):
        self.loaders: list[BaseLoader] = [
            *[HFLoader(loader) for loader in self.config.huggingface],
            *[LocalFileLoader(loader) for loader in self.config.local],
        ]

    def load(self) -> DatasetDict:
        path_dataset_map = {}
        for loader in self.loaders:
            dst = loader.load()
            path_dataset_map.update(dst if isinstance(dst, DatasetDict) else DatasetDict({0: dst}))
        
        return DatasetDict(path_dataset_map)
    