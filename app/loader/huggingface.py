from pathlib import Path

from pydantic import computed_field, model_validator
from datasets import Dataset, DatasetDict, load_dataset

from .base import BaseLoader, BaseConfig
from ..helpers import LOGGER


class HFLoaderConfig(BaseConfig):
    path: str
    name: str | None = None
    token: str | None = None
    take_rows: int | None = None
    split: str | None = "train"
    directory: Path | str | None = "dataset"
    
    @model_validator(mode="after")
    def validate_save_dir(self):
        if self.directory is None:
            return self
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        return self
    
    @computed_field
    @property
    def save(self) -> bool:
        return bool(self.directory)
    
    @computed_field
    @property
    def save_path(self) -> Path | None:
        if self.directory is None:
            return None
        return self.directory / self.path


class HFLoader(BaseLoader):
    def __init__(self, config: HFLoaderConfig):
        super().__init__(config)
        self.config: HFLoaderConfig

    def load_or_download(self) -> Dataset | DatasetDict:
        path = self.config.save_path
        dataset = None
        if path and path.exists():
            if path.is_dir():
                try:
                    dataset = Dataset.load_from_disk(path.as_posix())
                except Exception:
                    pass
                try:
                    dataset = DatasetDict.load_from_disk(path.as_posix())
                except Exception:
                    pass
        if dataset is not None:
            LOGGER.info(f"Loaded existing dataset from {path}")
            return dataset
        LOGGER.info(f"Downloading dataset from {self.config.path}")
        return load_dataset(
            path=self.config.path,
            name=self.config.name,
            token=self.config.token,
            split=self.config.split,
        )

    def save(self, dataset: Dataset):
        if self.config.save and not self.config.save_path.exists():
            dataset.save_to_disk(self.config.save_path.as_posix())
            LOGGER.info(f"Saved dataset to {self.config.save_path}")

    def load(self) -> DatasetDict:
        dsts = self.load_or_download()
        self.save(dsts)
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
