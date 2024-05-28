"""
# DatasetPipeline

A data processing and analysis pipeline designed to handle various jobs related to data transformation, quality assessment, deduplication, and formatting. The pipeline can be configured and executed using YAML configuration files.

## Pipelining Classes

- `Pipeline`: The dataset pipeline.
- `Job`: Represents a single job in the pipeline.
- `JobConfig`: Represents the configuration for a job.

---
## Loaders

- `Loader`: The main data loader.
- `LoaderConfig`: Represents the configuration for Loader class.
- `HFLoader`: A specific type of loader that loads data from the HuggingFace Hub.
- `HFLoaderConfig`: Represents the configuration for an HFLoader.
- `LocalFileLoader`: A specific type of loader that loads data from a local file.
- `LocalFileLoaderConfig`: Represents the configuration for a LocalFileLoader.

---
## Dataset Formatters

- `Format`: Main class for formatting datasets.
- `FormatConfig`: Represents the configuration for a format.
- `MergerFormat`: A specific type of format that merges multiple data formats.
- `MergerFormatConfig`: Represents the configuration for a MergerFormat.
- `FieldConfig`: Represents the configuration for a field in a format.
- `SFTFormat`: A specific type of format that performs semantic field transformation.
- `SFTFormatConfig`: Represents the configuration for an SFTFormat.
- `Role`: Represents a role enumeration in a data format.
- `DPOFormat`: A specific type of format that performs deduplication and projection.
- `DPOFormatConfig`: Represents the configuration for a DPOFormat.
- `DPOColumns`: Represents the columns in a DPO format.
- `ToTextFormat`: A specific type of format that converts data to text format.
- `ToTextFormatConfig`: Represents the configuration for a ToTextFormat.
- `RoleConfig`: Represents the configuration for a role in a format.
- `OutputFormat`: A specific type of format that saves data to an output location.
- `OutputFormatConfig`: Represents the configuration for an OutputFormat.

---
## Deduplication Engines

- `Dedup`: Represents a data deduplicator.
- `DedupConfig`: Represents the configuration for a deduplicator.
- `SemanticDedup`: A specific type of deduplicator that performs semantic deduplication.
- `SemanticDedupConfig`: Represents the configuration for a SemanticDedup.

---
## Data Analyzers

- `Analyzer`: Represents a data analyzer.
- `AnalyzerConfig`: Represents the configuration for an analyzer.
- `QualityAnalyzer`: A specific type of analyzer that analyzes data quality.
- `QualityConfig`: Represents the configuration for a QualityAnalyzer.
- `TEXT_QUALITY_EXAMPLE_MESSAGES`: Represents example OpenAI messages for text quality analysis.

---
## Dataset Savers

- `Saver`: Represents a data saver.
- `SaverConfig`: Represents the configuration for a saver.
- `LocalSaver`: A specific type of saver that saves data to a local location.
- `LocalSaverConfig`: Represents the configuration for a LocalSaver.
- `LocalDirSaverConfig`: Represents the configuration for a LocalSaver that saves data to a local directory.
- `FileType`: Enumeration representing the type of file.

---

### Notes:
- Though, the `Pipeline` class is the main entry point, other classes can be used independently with their respective configurations.
"""

from .pipeline import Pipeline
from .job import Job, JobConfig

from .loader import (
    Loader,
    LoaderConfig,
    HFLoader,
    HFLoaderConfig,
    LocalFileLoader,
    LocalFileLoaderConfig,
)
from .format import (
    Role,
    DPOColumns,
    Format,
    FormatConfig,
    MergerFormat,
    MergerFormatConfig,
    FieldConfig,
    SFTFormat,
    SFTFormatConfig,
    DPOFormat,
    DPOFormatConfig,
    ToTextFormat,
    ToTextFormatConfig,
    RoleConfig,
    OutputFormat,
    OutputFormatConfig,
)
from .dedup import (
    Dedup,
    DedupConfig,
    SemanticDedup,
    SemanticDedupConfig,
)
from .analyzer import (
    Analyzer,
    AnalyzerConfig,
    QualityAnalyzer,
    QualityAnalyzerConfig,
    TEXT_QUALITY_EXAMPLE_MESSAGES,
)
from .saver import (
    Saver,
    SaverConfig,
    LocalSaver,
    LocalSaverConfig,
    LocalDirSaverConfig,
    FileType,
)
from .sample_job import config as sample_job_config

JobConfig.sample = sample_job_config

__all__ = [
    "Pipeline",
    "Job",
    "JobConfig",
    
    "Loader",
    "LoaderConfig",
    "HFLoader",
    "HFLoaderConfig",
    "LocalFileLoader",
    "LocalFileLoaderConfig",
    
    "Format",
    "FormatConfig",
    "MergerFormat",
    "MergerFormatConfig",
    "FieldConfig",
    "SFTFormat",
    "SFTFormatConfig",
    "Role",
    "DPOFormat",
    "DPOFormatConfig",
    "DPOColumns",
    "ToTextFormat",
    "ToTextFormatConfig",
    "RoleConfig",
    "OutputFormat",
    "OutputFormatConfig",
    
    "Dedup",
    "DedupConfig",
    "SemanticDedup",
    "SemanticDedupConfig",
    
    "Analyzer",
    "AnalyzerConfig",
    "QualityAnalyzer",
    "QualityAnalyzerConfig",
    "TEXT_QUALITY_EXAMPLE_MESSAGES",
    
    "Saver",
    "SaverConfig",
    "LocalSaver",
    "LocalSaverConfig",
    "LocalDirSaverConfig",
    "FileType",
]
