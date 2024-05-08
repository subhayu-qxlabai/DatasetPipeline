from enum import Enum
from pathlib import Path
from warnings import warn

from datasets import Dataset
from pydantic import model_validator, field_validator, computed_field

from .base import BaseSaver, BaseConfig
from ..helpers.utils import get_ts_filename


class FileType(str, Enum):
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"


class LocalDirSaverConfig(BaseConfig):
    directory: Path | str = "processed"
    filetype: FileType | str | None = FileType.PARQUET

    @field_validator('directory')
    @classmethod
    def check_directory(cls, v: str | Path) -> Path:
        if isinstance(v, str):
            v = Path(v)
        v.mkdir(exist_ok=True, parents=True)
        return v
    
    @field_validator('filetype')
    @classmethod
    def check_filetype(cls, v: FileType | str | None) -> FileType:
        if isinstance(v, FileType):
            return v
        if isinstance(v, str) and v.lower() in {x.value for x in FileType}:
            return FileType(v.lower())
        warn(f"Invalid filetype {v!r}, defaulting to {FileType.PARQUET!r}", UserWarning)
        return FileType.PARQUET

class LocalSaverConfig(LocalDirSaverConfig):
    filename: str | None = None
    
    @model_validator(mode="after")
    def validate_fields(self):
        if not self.filename:
            self.filename = get_ts_filename("dataset", add_random=False).name
        
        suffix = self.filename.split(".")[-1]
        if (isinstance(self.filetype, FileType) and self.filetype.value == suffix):
            return self
        
        if not self.filetype:
            if suffix in {x.value for x in FileType}:
                self.filetype = FileType(suffix)
            else:
                warn(f"Invalid filetype {suffix!r}, defaulting to {FileType.PARQUET!r}", UserWarning)
                self.filetype = FileType.PARQUET
        return self

    @computed_field
    @property
    def save_path(self) -> Path:
        suffix = f".{self.filetype.value}"
        if self.filename.endswith(suffix):
            return Path(self.directory) / Path(self.filename)
        return Path(self.directory) / Path(self.filename).with_suffix(suffix).name


class LocalSaver(BaseSaver):
    def __init__(self, dataset: Dataset, config: LocalSaverConfig):
        super().__init__(dataset, config)
        self.config: LocalSaverConfig

    def save(self) -> Path:
        if self.config.filename is None:
            return
        path = self.config.save_path
        path.parent.mkdir(exist_ok=True, parents=True)
        getattr(self.dataset, f"to_{self.config.filetype.value}")(path)
        return path
    