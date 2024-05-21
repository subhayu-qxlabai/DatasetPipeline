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
