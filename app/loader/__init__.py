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

    def load(self) -> dict[str, Dataset]:
        path_datasets_map = [
            (loader.config.path, loader.load())
            for loader in self.loaders
        ]
        path_datasets_map = [
            (path, dst if isinstance(dst, DatasetDict) else {0: dst})
            for path, dst in path_datasets_map
        ]
        return {
            (f"{path}-{split}" if isinstance(split, str) else path): dst
            for path, ddict in path_datasets_map
            for split, dst in ddict.items()
        }
    