from datasets import Dataset
from pydantic import Field
from .base import BaseFormat, BaseConfig


class OutputConfig(BaseConfig):
    return_only_messages: bool = Field(default=False,description="Whether to only keep the 'messages' column. Defaults to 'False'")

class OutputFormat(BaseFormat):
    def __init__(self, dataset: Dataset, config: OutputConfig = OutputConfig()):
        super().__init__(dataset, config)
        self.config: OutputConfig
    
    @property
    def is_this_format(self) -> bool:
        return True

    def _format(self) -> Dataset:
        if self.config.return_only_messages:
            return self.dataset.select_columns(self.messages_cols)
        return self.dataset
