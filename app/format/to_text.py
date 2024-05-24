"""
This module provides the `ToTextFormat` class for converting a dataset with one or multiple columns of conversational data to text format specified by the `config`.

Example Usage:

```python
from datasets import Dataset
from dedup.to_text import ToTextConfig, ToTextFormat

# Create a dataset
dataset = Dataset.from_csv('data.csv')

# Create a configuration object
config = ToTextConfig(
    column='conversation_column',
    # Add any other configuration options here
)

# Create a ToTextFormat object
formatter = ToTextFormat(dataset, config)

# Perform the conversion
converted_dataset = formatter.format()

# Print the converted dataset
print(converted_dataset)
```
"""

from datasets import Dataset

from .base import BaseFormat, BaseConfig
from ..helpers.formatter import MessagesFormatter, FormatterConfig, RoleConfig
from ..helpers import LOGGER


class ToTextConfig(BaseConfig, FormatterConfig):
    pass


class ToTextFormat(BaseFormat):
    """Converts a dataset with one or multiple columns of conversational data to text format specified by the `config`"""
    def __init__(self, dataset: Dataset, config: ToTextConfig = ToTextConfig()):
        super().__init__(dataset, config)
        self.config: ToTextConfig
        self.conv_cols = self.get_standard_columns()
    
    @property
    def is_this_format(self) -> bool:
        if len(self.conv_cols) == 0:
            return False
        return True

    def _format(self) -> Dataset:
        if not self.is_this_format:
            return self.dataset
        dicts = self.dataset.to_dict()
        
        LOGGER.info(f"Formatting the column(s): {', '.join(self.conv_cols)!r} to format: \n{self.config}")
        for col in self.conv_cols:
            dicts[col] = MessagesFormatter(
                messages=dicts[col], config=self.config,
            ).format().formatted_messages
        dataset = Dataset.from_dict(dicts)
        return dataset
