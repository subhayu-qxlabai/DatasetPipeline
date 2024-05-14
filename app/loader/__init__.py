from dataclasses import dataclass

from datasets import Dataset, DatasetDict

from .base import BaseLoader, BaseConfig, BaseModel
from .huggingface import HFLoader, HFLoaderConfig


class LoaderConfig(BaseModel):
    huggingface: list[HFLoaderConfig]
    

@dataclass
class Loader:
    config: LoaderConfig
    
    def __post_init__(self):
        self.loaders = [
            *[HFLoader(loader) for loader in self.config.huggingface]
        ]

    def load(self) -> DatasetDict:
        path_dataset_map = {}
        for loader in self.loaders:
            dst = loader.load()
            path_dataset_map.update(dst if isinstance(dst, DatasetDict) else DatasetDict({0: dst}))
        
        return DatasetDict(path_dataset_map)
    