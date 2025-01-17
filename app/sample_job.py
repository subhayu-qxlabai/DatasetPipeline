"""
This module defines a configuration for a job that loads a dataset from the Hugging Face dataset repository, formats the dataset, deduplicates it, analyzes its quality, and saves it to a local directory as a Parquet file.

The configuration is defined in the `config` variable, which is an instance of `JobConfig`.

Example usage:

```python
if __name__ == "__main__":
    job = Job(config)
    pprint(job.to_yaml())
```
"""

from pprint import pprint
from .job import Job, JobConfig
from .loader import LoaderConfig, HFLoaderConfig
from .format import (
    FormatConfig,
    MergerFormatConfig,
    FieldConfig,
    SFTFormatConfig,
    Role,
    DPOFormatConfig,
    DPOColumns,
    ConversationalFormatConfig,
    ConversationalTextFormatConfig,
    ToTextFormatConfig,
    RoleConfig,
    OutputFormatConfig,
)
from .analyzer import AnalyzerConfig, QualityAnalyzerConfig
from .dedup import DedupConfig, SemanticDedupConfig
from .saver import SaverConfig, LocalSaverConfig, FileType


config = JobConfig(
    load=LoaderConfig(
        huggingface=HFLoaderConfig(
            path="davanstrien/data-centric-ml-sft",
        )
    ),
    format=FormatConfig(
        merger=MergerFormatConfig(
            user=FieldConfig(
                fields=["book_id", "author", "text"],
                separator="\n",
                merged_field="human",
            ),
        ),
        sft=SFTFormatConfig(
            use_openai=False,
            column_role_map={
                "persona": Role.SYSTEM,
                "human": Role.USER,
                "summary": Role.ASSISTANT,
            },
        ),
        dpo=DPOFormatConfig(
            column_role_map={
                "human": "user",  # we can pass string
                "persona": DPOColumns.SYSTEM,  # or DPOColumns
                "positive": DPOColumns.CHOSEN,
                "negative": DPOColumns.REJECTED,
            }
        ),
        conv=ConversationalFormatConfig(),
        conv_text=ConversationalTextFormatConfig(),
        to_text=ToTextFormatConfig(
            system=RoleConfig(
                template="SYSTEM: {system}",
                key="system",
            ),
            user=RoleConfig(
                template="USER: {user}",
                key="user",
            ),
            assistant=RoleConfig(
                template="ASSISTANT: {assistant}",
                key="assistant",
            ),
            separator="\n\n",
        ),
        output=OutputFormatConfig(
            return_only_messages=True,
        ),
    ),
    deduplicate=DedupConfig(
        semantic=SemanticDedupConfig(
            threshold=0.8,
        )
    ),
    analyze=AnalyzerConfig(
        quality=QualityAnalyzerConfig(
            column_name="messages",
            categories=[
                "code",
                "math",
                "job",
                "essay",
                "translation",
                "literature",
                "history",
                "science",
                "medicine",
                "news",
                "finance",
                "geography",
                "philosophy",
                "psychology",
                "education",
                "art",
                "music",
                "technology",
                "environment",
                "food",
                "sports",
                "fashion",
                "travel",
                "culture",
                "language",
                "religion",
                "politics",
                "space",
                "entertainment",
                "healthcare",
                "animals",
                "weather",
                "architecture",
                "automotive",
                "business",
                "comedy",
                "crime",
                "diy",
                "economics",
                "gaming",
                "law",
                "marketing",
                "parenting",
                "science_fiction",
                "social_media",
                "mythology",
                "folklore",
                "astrology",
                "horror",
                "mystery",
            ],
        )
    ),
    save=SaverConfig(
        local=LocalSaverConfig(
            directory="processed",
            filetype=FileType.PARQUET,  # can also be sent as a string
        )
    ),
)


if __name__ == "__main__":
    job = Job(config)
    pprint(job.to_yaml())
