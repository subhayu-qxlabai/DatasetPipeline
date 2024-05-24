"""
This module contains the `Saver` class and its configuration class `SaverConfig`. 

The `Saver` class is responsible for saving a `Dataset` object to different storage formats. It currently supports saving to a local directory.

The `SaverConfig` class is a pydantic model that defines the configuration for the `Saver` class. It has a single field `local`, which is an instance of `LocalSaverConfig`.

"""

from dataclasses import dataclass

from datasets import Dataset
from pydantic import Field

from .base import BaseSaver, BaseConfig, BaseModel
from .local import LocalSaver, LocalSaverConfig, LocalDirSaverConfig, FileType


class SaverConfig(BaseModel):
    local: LocalSaverConfig | None = Field(default=LocalSaverConfig(), description="Configuration for saving the dataset locally.")


@dataclass
class Saver:
    dataset: Dataset
    config: SaverConfig
    
    def __post_init__(self):
        self.savers = [
            LocalSaver(self.dataset, self.config.local) if self.config.local is not None else None,
        ]

    def save(self):
        return [saver.save() for saver in self.savers if saver is not None]
    