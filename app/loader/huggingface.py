from itertools import chain
from pathlib import Path

from pydantic import computed_field, model_validator
from datasets import Dataset, DatasetDict, load_dataset

from .base import BaseLoader, BaseConfig


class HFLoaderConfig(BaseConfig):
    token: str | None = None
    save: bool = True
    merge: bool = True
    split: str | None = "train"
    directory: Path | str = "dataset"
    
    @model_validator(mode="after")
    def validate_save_dir(self):
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        return self
    
    @computed_field
    @property
    def save_path(self) -> Path:
        return self.directory / self.path


class HFLoader(BaseLoader):
    def __init__(self, config: HFLoaderConfig):
        super().__init__(config)
        self.config: HFLoaderConfig
        
    def load_or_download(self):
        if self.config.save_path.exists():
            try:
                return Dataset.load_from_disk(self.config.save_path.as_posix())
            except Exception:
                pass
        return load_dataset(
            path=self.config.path,
            token=self.config.token
        )
    
    def save(self, dataset: Dataset):
        if self.config.save:
            dataset.save_to_disk(self.config.save_path.as_posix())

    def load(self) -> Dataset | DatasetDict:
        dsts = self.load_or_download()
        self.save(dsts)
        
        if self.config.merge:
            if isinstance(dsts, DatasetDict):
                dsts = Dataset.from_list(list(chain(*[d.to_list() for d in dsts.values()])))
            return dsts
        
        if isinstance(dsts, DatasetDict) and self.config.split in dsts:
            return dsts.get(self.config.split)

        return dsts
    