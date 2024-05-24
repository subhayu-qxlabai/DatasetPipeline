"""
This module provides a set of classes for formatting datasets into the standard format.

The standard format is a dataset with a column in the following form:

```python
[
    {
        "role": "system",
        "content": "System message"
    },
    {
        "role": "user",
        "content": "User input"
    },
    {
        "role": "assistant",
        "content": "Assistant output"
    }
]
```

The module provides the following classes:

- `Format`: The base class for formatting datasets into the standard format.
- `MergerFormat`: Merges multiple columns of a dataset into a single columns for the system, user and assistant.
- `ConversationalFormat`: Converts one or multiple columns of conversational data to the standard format, irrespective of the current format.
- `ConversationalTextFormat`: Converts a dataset having columns with conversation between between 2-3 entities in text format to a standard conversational format having system, user and assistant.
- `ToTextFormat`: Converts a dataset with one or multiple columns of conversational data to text format specified by the `config`.
- `SFTFormat`: Converts a dataset with one or multiple columns of conversational data in SFT format to the standard format.
- `DPOFormat`: Converts a dataset with one or multiple columns of conversational data in DPO format to the standard format.
- `OutputFormat`: Converts a dataset into the specified output format.
"""

from functools import partial
from dataclasses import dataclass

from datasets import Dataset
from pydantic import Field

from .base import BaseFormat, BaseConfig
from .sft import SFTFormat, SFTConfig, Role
from .merger import MergerFormat, MergerConfig, FieldConfig
from .conv import ConversationalFormat, ConvConfig
from .conv_text import ConversationalTextFormat, ConvTextConfig
from .dpo import DPOFormat, DPOConfig, DPOColumns
from .to_text import ToTextFormat, ToTextConfig, RoleConfig
from .output import OutputFormat, OutputConfig


class FormatConfig(BaseConfig):
    merger: MergerConfig | None = Field(default=MergerConfig(), description="Configuration for merging different columns into 'system', 'user' and 'assistant'")
    sft: SFTConfig | None = Field(default=SFTConfig(), description="Configuration for detecting 'system', 'user' and 'assistant' columns")
    dpo: DPOConfig | None = Field(default=DPOConfig(), description="Configuration for detecting 'system', 'user', 'chosen' and 'rejected' columns")
    conv: ConvConfig | None = Field(default=ConvConfig(), description="Configuration for detecting and converting conversational object formats. Columns having values like `list[dict[str, str]]`")
    conv_text: ConvTextConfig | None = Field(default=None, description="Configuration for detecting and converting conversational text formats.")
    to_text: ToTextConfig | None = Field(default=ToTextConfig(), description="Configuration for converting standardized messages to text format.")
    output: OutputConfig | None = Field(default=OutputConfig(), description="Configuration for outputting the formatted dataset.")


@dataclass
class Format:
    """
    Formats a dataset into the standard format.
    
    Params:
        dataset (Dataset): The dataset to be formatted
        config (FormatConfig): The configuration for the format
    """
    dataset: Dataset
    config: FormatConfig | None = FormatConfig()
    
    def __post_init__(self):
        self.merger: type[MergerFormat] = partial(MergerFormat, config=self.config.merger)
        self.sft: type[SFTFormat] = partial(SFTFormat, config=self.config.sft)
        self.conv_text: type[ConversationalTextFormat] = partial(ConversationalTextFormat, config=self.config.conv_text)
        self.conv: type[ConversationalFormat] = partial(ConversationalFormat, config=self.config.conv)
        self.dpo: type[DPOFormat] = partial(DPOFormat, config=self.config.dpo)
        self.output: type[OutputFormat] = partial(OutputFormat, config=self.config.output)
        self.to_text: type[ToTextFormat] = partial(ToTextFormat, config=self.config.to_text)
    
    @property
    def _base_chain(self) -> DPOFormat:
        return (
            self.merger(self.dataset)
            | self.sft
            | self.conv_text
            | self.conv
            | self.dpo
        )
    
    def format(self, textualize: bool = False) -> Dataset:
        """
        Standardizes the dataset by applying a series of data transformations.

        Params:
            textualize (bool, optional): If True, the messages in the dataset is transformed into text format. Defaults to False.

        Returns:
            Dataset: The analyzed dataset.

        Note:
            - If `SFT` dataset, the returned dataset will have `messages` column.
            - If `DPO` dataset, the returned dataset will have `chosen` and `rejected` columns.
        """
        if self.config is None:
            return self.dataset
        chain = (self._base_chain | self.output)
        if not textualize:
            return chain.format()
        return (chain | self.to_text).format()
