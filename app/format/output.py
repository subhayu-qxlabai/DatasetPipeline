"""
This module provides the `OutputFormat` class for formatting datasets based on the specified configuration.

The `OutputFormat` class is a subclass of `BaseFormat` and is used to format a dataset according to the class `OutputConfig` configuration. It provides the following methods:

- `__init__(self, dataset: Dataset, config: OutputConfig = OutputConfig())`: Initializes the `OutputFormat` object with a dataset and an optional configuration.
- `is_this_format(self) -> bool`: Returns `True` as this format always applies.
- `_format(self) -> Dataset`: Formats the dataset based on the configuration. If `config.return_only_messages` is `True`, it returns a new dataset with only the 'messages' column. Otherwise, it returns the original dataset.

Example:
    ```python
    from my_module import OutputFormat

    dataset = ...
    config = OutputConfig(return_only_messages=True)
    output_formatter = OutputFormat(dataset, config)
    formatted_dataset = output_formatter.format()
    ```
"""

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
