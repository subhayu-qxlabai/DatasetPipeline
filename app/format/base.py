"""
This module provides the `BaseFormatConfig` and `BaseFormat` classes for formatting datasets.

`BaseFormatConfig` is a subclass of `BaseModel` and serves as an empty configuration model.

`BaseFormat` is an abstract base class for formatting datasets. It has an initializer that takes a `Dataset` object and an optional `BaseFormatConfig` object. It also has several properties and methods:

- `has_config`: A boolean property indicating whether the format has a config.
- `format_class`: A property returning the class of the format.
- `format_name`: A property returning the name of the format.
- `is_this_format`: A boolean property indicating whether the format is this format.
- `_format`: An abstract method that returns a `Dataset` object representing the messages.
- `format`: A method that returns a `Dataset` object with the formatted fields.
- `get_conv_columns`: A method returning a list of columns that have a conversion type.
- `get_standard_columns`: A method returning a list of columns that have a standard type.
- `__or__`: A method for chaining formatters. It creates an instance of the provided class and extends the `formats` list with the current format's name and the current format's formats. It also extends the `messages_cols` list.

Usage Example:

```python
from datasets import Dataset
from my_module import BaseFormat

# Create a dataset
dataset = Dataset.from_pandas(pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}))

# Create a BaseFormat instance
formatter = BaseFormat(dataset)

# Format the dataset
formatted_dataset = formatter.format()

# Print the formatted dataset
print(formatted_dataset)
    ```
"""

from typing import Any
from random import randint
from abc import ABC, abstractmethod

from datasets import Dataset
import pandas as pd

from ..models.base import BaseModel
from ..helpers.types import is_conv_type, is_standard_type

class BaseFormatConfig(BaseModel):
    pass

class BaseFormat(ABC):
    def __init__(self, dataset: Dataset, config = BaseFormatConfig()):
        """
        Initializes the Format object with the given dataset.

        Parameters:
            dataset (Dataset): The dataset to be used for initialization.
            config (BaseFormatConfig): The config to be used for initialization.

        Returns:
            None
        """
        self.dataset = dataset
        self.config = config
        self.messages_cols: list[str] = []
        
        rnd_ix = randint(0, len(dataset) - 1)
        self.dict_repr: dict[str, Any] = dataset[rnd_ix]
        """Dictionary representation of the dataset (has 1 element)"""
        self.normalized_repr: dict[str, Any] = pd.json_normalize(dataset.take(10).to_list()).iloc[min(rnd_ix, 9)].to_dict()
        """JSON Normalized representation of the dataset (has 1 row). Basically, nested structures will be flattened and separated by dots."""
        self.formats: list[str] = []
        """List of formats that the dataset is a type of."""
        
    @property
    def has_config(self) -> bool:
        """
        Property that returns a boolean indicating whether the format has a config.

        :return: A boolean indicating whether the format has a config.
        :rtype: bool
        """
        return self.config is not None
    
    @property
    def format_class(self) -> str:
        """
        Method that returns the class of the format.

        :return: The class of the format.
        :rtype: str
        """
        return self.__class__
    
    @property
    def format_name(self) -> str:
        """
        Method that returns the name of the format.

        :return: The name of the format.
        :rtype: str
        """
        return self.format_class.__name__.replace("Format", "")
    
    @property
    def is_this_format(self) -> bool:
        """
        Property that returns a boolean indicating whether the format is this format.

        :return: A boolean indicating whether the format is this format.
        :rtype: bool
        """
        return False
    
    @abstractmethod
    def _format(self) -> Dataset:
        """
        Method that returns a Dataset object representing the messages.

        :return: A Dataset object representing the messages.
        :rtype: Dataset
        """
    
    def format(self) -> Dataset:
        """
        Method that returns a Dataset object with the formatted fields.

        :return: A Dataset object with the formatted fields.
        :rtype: Dataset
        """
        if self.has_config:
            return self._format()
        return self.dataset
    
    def get_conv_columns(self):
        return [key for key, value in self.dict_repr.items() if is_conv_type(value)]
    
    def get_standard_columns(self):
        return [key for key, value in self.dict_repr.items() if is_standard_type(value)]
    
    def __or__(self, cls: type["BaseFormat"]):
        instance = cls(self.format())
        if self.is_this_format:
            instance.formats.append(self.format_name)
        instance.formats.extend(self.formats)
        instance.messages_cols = list(set(instance.messages_cols + self.messages_cols))
        return instance
    