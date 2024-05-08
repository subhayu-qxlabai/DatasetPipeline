from dataclasses import dataclass

from datasets import Dataset

from .base import BaseSaver, BaseConfig, BaseModel
from .local import LocalSaver, LocalSaverConfig, LocalDirSaverConfig, FileType


class SaverConfig(BaseModel):
    local: LocalSaverConfig | None = LocalSaverConfig()


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
    