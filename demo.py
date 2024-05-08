# %%
from app.job import Job, JobConfig
from app.loader import LoaderConfig, HFLoaderConfig
from app.format import (
    FormatConfig, 
    MergerConfig, 
    FieldConfig, 
    SFTConfig, 
    Role, 
    DPOConfig, 
    DPOColumns,  
    ToTextConfig, 
    RoleConfig,
    OutputConfig, 
)
from app.analyzer import AnalyzerConfig, QualityConfig, TEXT_QUALITY_EXAMPLE_MESSAGES
from app.saver import SaverConfig, LocalSaverConfig, FileType


config = JobConfig(
    loader=LoaderConfig(
        huggingface=[
            HFLoaderConfig(
                path="davanstrien/data-centric-ml-sft",
                merge=True,
            ),
        ]
    ),
    format=FormatConfig(
        merger=MergerConfig(
            user=FieldConfig(
                fields=["book_id", "author", "text"],
                separator="\n",
                merged_field="human",
            ),
        ),
        sft=SFTConfig(
            use_openai=False,
            column_role_map={
                "persona": Role.SYSTEM,
                "human": Role.USER,
                "summary": Role.ASSISTANT,
            },
        ),
        dpo=DPOConfig(
            column_role_map={
                "human": "user",  # we can pass string
                "persona": DPOColumns.SYSTEM,  # or DPOColumns
                "positive": DPOColumns.CHOSEN,
                "negative": DPOColumns.REJECTED,
            }
        ),
        to_text=ToTextConfig(
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
        output=OutputConfig(
            return_only_messages=True,
        ),
    ),
    analyzer=AnalyzerConfig(
        quality=QualityConfig(
            column_name="messages",
        )
    ),
    saver=SaverConfig(
        local=LocalSaverConfig(
            directory="processed",
            filetype=FileType.PARQUET,  # can also be sent as a string
        )
    ),
)

# %%

if __name__ == "__main__":
    job = Job(config)
    job.run()
