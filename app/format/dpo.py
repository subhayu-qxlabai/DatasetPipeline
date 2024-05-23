"""
This module provides the `DPOFormat` class for handling datasets in the DPO (Dialogue Prompt Optimization) format.

The `DPOFormat` class is a subclass of `BaseFormat` and is designed to handle datasets in the DPO format. It provides methods for formatting the dataset by converting the columns into messages and adding them to the dataset.

The class has the following main components:

- `__init__(self, dataset: Dataset, config: DPOConfig = DPOConfig())`: Initializes the class with a dataset and a configuration object. It sets up the `pattern_role_map` and `col_map` attributes by calling the `_get_col_map` method.

- `_get_col_map(self)`: Creates a dictionary mapping column names to their corresponding roles based on a pattern-role mapping provided in the configuration.

- `is_this_format(self) -> bool`: Checks if the dataset has at least two columns and if it has both `CHOSEN` and `REJECTED` columns.

- `_convert_row_to_messages(self, row: dict[str, Any], assistant_col: DPOColumns = DPOColumns.CHOSEN)`: Converts a row of the dataset into a list of messages based on the column roles.

- `_convert_chosen_rejected_to_messages(self, row: dict[str, Any])`: Converts the `CHOSEN` and `REJECTED` columns of a row into separate messages.

- `format(self) -> Dataset`: Formats the dataset by converting the `CHOSEN` and `REJECTED` columns into messages. If the dataset is not in the DPO format, it returns the original dataset.

Example usage:

```python
from datasets import Dataset
from dpo_format import DPOFormat, DPOConfig

# Create a dataset and configuration object
dataset = Dataset(...)
config = DPOConfig(...)

# Initialize the DPOFormat class
dpo_format = DPOFormat(dataset, config)

# Format the dataset
formatted_dataset = dpo_format.format()
```
"""

from typing import Any

from datasets import Dataset
from pydantic import Field, model_validator

from .base import BaseFormat, BaseConfig
from ..helpers.regex_dict import RegexDict
from ..constants import DPOColumns, MessageRole as Role, MessageField


PATTERN_ROLE_MAP = {
    "chosen.*": DPOColumns.CHOSEN,
    "rejected.*": DPOColumns.REJECTED,
    
    "trajectory.*": DPOColumns.USER,
    "instruction.*": DPOColumns.USER,
    "human.*": DPOColumns.USER,
    "question.*": DPOColumns.USER,
    "^prompt.*": DPOColumns.USER,
    "user.*": DPOColumns.USER,
    
    "system.*": DPOColumns.SYSTEM,
}

class DPOConfig(BaseConfig):
    column_role_map: dict[str, DPOColumns|str] = Field(
        default=PATTERN_ROLE_MAP,
        description="A mapping of column names to role of each column in the dataset. Roles can be `user`, `system`, `chosen` or `rejected`."
    )

    @model_validator(mode="after")
    def validate_column_role_map(self):
        try:
            self.column_role_map = {
                k: v if isinstance(v, DPOColumns) else DPOColumns(v) 
                for k, v in self.column_role_map.items()
            }
        except Exception as e:
            raise ValueError(
                f'Values of `column_role_map` must be in the following: {", ".join(r.value for r in DPOColumns)}'
            )
        return self


class DPOFormat(BaseFormat):
    def __init__(self, dataset: Dataset, config: DPOConfig = DPOConfig()):
        super().__init__(dataset, config)
        self.config: DPOConfig
        self.pattern_role_map = RegexDict(self.config.column_role_map)
        self.col_map: dict[DPOColumns, str] = self._get_col_map()

    def _get_col_map(self):
        data = self.dict_repr
        role_col: list[tuple[DPOColumns, str]] = []
        p_r_map: dict[str, DPOColumns] = self.pattern_role_map
        for col in list(data):
            key = p_r_map.get(col)
            if key is not None:
                role_col.append((key, col))
                p_r_map = RegexDict({
                    k: v for k, v in p_r_map.items() if v != key
                })
        return dict(role_col)

    @property
    def is_this_format(self) -> bool:
        if len(self.col_map) < 2:
            return False
        if {DPOColumns.CHOSEN, DPOColumns.REJECTED}.issubset(self.col_map.keys()):
            return True
        return False
    
    def _convert_row_to_messages(self, row: dict[str, Any], assistant_col: DPOColumns = DPOColumns.CHOSEN):
        messages: list[dict[str, str]] = []
        conv_cols = self.get_conv_columns()

        for col_type in [DPOColumns.SYSTEM, DPOColumns.USER, assistant_col]:
            col = self.col_map.get(col_type)
            if col is not None:
                if col in conv_cols:
                    messages.extend(row[col])
                else:
                    role = (
                        Role.SYSTEM if col_type == DPOColumns.SYSTEM
                        else Role.USER if col_type == DPOColumns.USER 
                        else Role.ASSISTANT
                    )
                    messages.append({MessageField.ROLE.value: role.value, MessageField.CONTENT.value: row[col]})

        deduped_messages: list[dict[str, str]] = []
        for message in messages:
            if message not in deduped_messages:
                deduped_messages.append(message)
        return deduped_messages
    
    def _convert_chosen_rejected_to_messages(self, row: dict[str, Any]):
        d = {
            DPOColumns.CHOSEN.value: self._convert_row_to_messages(row, assistant_col=DPOColumns.CHOSEN),
            DPOColumns.REJECTED.value: self._convert_row_to_messages(row, assistant_col=DPOColumns.REJECTED),
        }
        return d

    def _format(self) -> Dataset:
        if not self.is_this_format:
            return self.dataset
        dataset = self.dataset.map(
            self._convert_chosen_rejected_to_messages, 
            # load_from_cache_file=False
        )
        self.messages_cols += [DPOColumns.CHOSEN.value, DPOColumns.REJECTED.value]
        return dataset
