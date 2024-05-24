"""
Module for loading datasets from local files.

This module provides a `LocalFileLoader` class that is used to load datasets from local files. It follows a Pydantic-based configuration model, where the `LocalFileLoaderConfig` class defines the parameters for loading a dataset from a local file.

The `LocalFileLoaderConfig` class has two fields:
- `path`: Specifies the path to the file. Must be one of: `csv`, `json`, `parquet`.
- `take_rows`: Specifies the number of rows to take from the file. Defaults to `None`.

The `validate_path` method is a validator that checks if the specified path exists and is a file, and if the file type is one of the allowed types (`csv`, `json`, or `parquet`).

The `LocalFileLoader` class has an `__init__` method that initializes the loader with a `LocalFileLoaderConfig` object. It also has a `_load` method that loads the dataset from the specified file using the `datasets` library. The file type is determined based on the file extension, and the appropriate loader method is called. The loaded dataset is returned as a `DatasetDict`.

Example usage:

```python
from datasets import DatasetDict
from app.loader.local_file import LocalFileLoader, LocalFileLoaderConfig

# Create a LocalFileLoaderConfig object
config = LocalFileLoaderConfig(path="path/to/file.csv", take_rows=10)

# Create a LocalFileLoader object
loader = LocalFileLoader(config)

# Load the dataset
dataset_dict: DatasetDict = loader.load()
```
"""

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
    
    def _load(self) -> DatasetDict:
        loader_method = f"from_{self.config.path.suffix.replace('.', '').replace('jsonl', 'json')}"
        dsts: Dataset = getattr(Dataset, loader_method)(self.config.path.as_posix())
        return DatasetDict({self.config.path.as_posix(): dsts})
