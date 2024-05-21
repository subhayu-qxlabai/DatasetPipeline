from enum import Enum
from pathlib import Path

from datasets import Dataset, DatasetDict
from pydantic import Field, model_validator

from .base import BaseLoader, BaseConfig


class FileType(str, Enum):
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"


class LocalFileLoaderConfig(BaseConfig):
    path: str | Path = Field(description="Path to the file. Must be one of: `csv`, `json`, `parquet`")
    take_rows: int | None = Field(default=None, description="Number of rows to take from the file.")
    
    @model_validator(mode="after")
    def validate_path(self):
        self.path = Path(self.path)
        assert self.path.exists(), f"Path {self.path} does not exist"
        assert self.path.is_file(), f"Path {self.path} is not a file"
        assert (
            self.path.suffix.replace(".", "") in {x.value for x in FileType}, 
            f"Invalid filetype: {self.path.suffix}"
        )
        return self

class LocalFileLoader(BaseLoader):
    def __init__(self, config: LocalFileLoaderConfig):
        super().__init__(config)
        self.config: LocalFileLoaderConfig
    
    def load(self) -> DatasetDict:
        loader_method = f"from_{self.config.path.suffix.replace('.', '').replace('jsonl', 'json')}"
        dsts: Dataset = getattr(Dataset, loader_method)(self.config.path.as_posix())
        return DatasetDict({self.config.path.as_posix(): dsts})
