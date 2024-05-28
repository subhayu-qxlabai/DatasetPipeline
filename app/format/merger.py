"""
In this module it provides a class called MergerFormat which merges multiple columns of a dataset into a single column for the system, user, and assistant.

Usage example:
```
dataset = Dataset(...)
config = MergerFormatConfig(...)
merger_format = MergerFormat(dataset, config)
merged_dataset = merger_format.format()
```

Args:
    dataset (Dataset): The dataset to be merged.
    config (MergerFormatConfig, optional): The configuration for merging. Defaults to MergerFormatConfig().

Attributes:
    config (MergerFormatConfig): The merged configuration.

Methods:
    config_fields: Returns a set of field names to be merged.
    is_this_format: Checks if the dataset can be merged.
    _merge_field_vals: Merges the values of multiple fields into a single field.
    _apply_field_config: Applies the field configuration to the dataset.
    _apply_field_configs: Applies multiple field configurations to the dataset.
    _format: Merges the columns of the dataset based on the configuration.

"""

from typing import Any
from datasets import Dataset

from .base import BaseFormat, BaseFormatConfig
from ..constants import MessageRole as Role
from pydantic import Field


class FieldConfig(BaseFormatConfig):
    fields: list[str] | None = Field(default=None, description="List of column names to merge. Defaults to 'null'")
    separator: str = Field(default=" ", description="Seperator to merge the column. Defaults to ' '")
    merged_field: str | None = Field(default=None, description="Merged column name.")

class MergerFormatConfig(BaseFormatConfig):
    system: FieldConfig | None = FieldConfig(merged_field=Role.SYSTEM.value)
    user: FieldConfig | None = FieldConfig(merged_field=Role.USER.value)
    assistant: FieldConfig | None = FieldConfig(merged_field=Role.ASSISTANT.value)
    remove_other_cols: bool = Field(default=False, description="Whether remove other columns. Defaults to 'False'")


class MergerFormat(BaseFormat):
    """Merges multiple columns of a dataset into a single columns for the system, user and assistant."""
    def __init__(self, dataset: Dataset, config: MergerFormatConfig = MergerFormatConfig()):
        super().__init__(dataset, config)
        self.config: MergerFormatConfig

    @property
    def config_fields(self) -> set[str]:
        return {
            *(self.config.system.fields or []),
            *(self.config.user.fields or []),
            *(self.config.assistant.fields or []),
        }

    @property
    def is_this_format(self) -> bool:
        if (
            self.config.system is None
            and self.config.user is None
            and self.config.assistant is None
        ):
            return False
        if (
            not self.config.system.fields
            and not self.config.user.fields
            and not self.config.assistant.fields
        ):
            return False
        if self.config_fields.issubset(self.dataset.column_names):
            return True
        return False

    @staticmethod
    def _merge_field_vals(data: dict[str, Any], field_config: FieldConfig):
        return field_config.separator.join(
            data.get(field) for field in field_config.fields 
            if isinstance(data.get(field), str)
        )

    @staticmethod
    def _apply_field_config(dataset: Dataset, field_config: FieldConfig):
        if field_config.fields is None or field_config.merged_field is None:
            return dataset
        return dataset.map(lambda x: {
            field_config.merged_field: MergerFormat._merge_field_vals(x, field_config)
        })

    @staticmethod
    def _apply_field_configs(dataset: Dataset, field_configs: list[FieldConfig]):
        for fc in field_configs:
            if fc is None:
                continue
            dataset = MergerFormat._apply_field_config(dataset, fc)
        return dataset

    def _format(self) -> Dataset:
        if not self.is_this_format:
            return self.dataset

        self.config.system.merged_field = self.config.system.merged_field or Role.SYSTEM.value
        self.config.user.merged_field = self.config.user.merged_field or Role.USER.value
        self.config.assistant.merged_field = self.config.assistant.merged_field or Role.ASSISTANT.value

        dataset = self._apply_field_configs(
            self.dataset, 
            [self.config.system, self.config.user, self.config.assistant]
        )

        if self.config.remove_other_cols:
            merged_fields = {
                self.config.system.merged_field, 
                self.config.user.merged_field, 
                self.config.assistant.merged_field
            }
            dataset = dataset.remove_columns(list(self.config_fields - merged_fields))

        return dataset
