from typing import Any
from datasets import Dataset

from .base import BaseFormat, BaseConfig
from ..constants import MessageRole as Role


class FieldConfig(BaseConfig):
    fields: list[str] | None = None
    separator: str = " "
    merged_field: str | None = None

class MergerConfig(BaseConfig):
    system: FieldConfig | None = FieldConfig(merged_field=Role.SYSTEM.value)
    user: FieldConfig | None = FieldConfig(merged_field=Role.USER.value)
    assistant: FieldConfig | None = FieldConfig(merged_field=Role.ASSISTANT.value)


class MergerFormat(BaseFormat):
    """Merges multiple columns of a dataset into a single columns for the system, user and assistant."""
    def __init__(self, dataset: Dataset, config: MergerConfig = MergerConfig()):
        super().__init__(dataset, config)
        self.config: MergerConfig
    
    @property
    def config_fields(self) -> set[str]:
        return {
            *(self.config.system.fields or []),
            *(self.config.user.fields or []),
            *(self.config.assistant.fields or []),
        }
    
    @property
    def is_this_format(self) -> bool:
        if self.config.system is None and self.config.user is None and self.config.assistant is None:
            return False
        if not self.config.system.fields and not self.config.user.fields and not self.config.assistant.fields:
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

    def _format(self) -> Dataset:
        if not self.is_this_format:
            return self.dataset
        
        self.config.system.merged_field = self.config.system.merged_field or Role.SYSTEM.value
        self.config.user.merged_field = self.config.user.merged_field or Role.USER.value
        self.config.assistant.merged_field = self.config.assistant.merged_field or Role.ASSISTANT.value
        
        dataset = self.dataset.map(
            lambda x: {
                self.config.system.merged_field: self._merge_field_vals(x, self.config.system),
                self.config.user.merged_field: self._merge_field_vals(x, self.config.user),
                self.config.assistant.merged_field: self._merge_field_vals(x, self.config.assistant),
            }
        )
        merged_fields = {self.config.system.merged_field, self.config.user.merged_field, self.config.assistant.merged_field}
        dataset = dataset.remove_columns(list(self.config_fields - merged_fields))
        return dataset
