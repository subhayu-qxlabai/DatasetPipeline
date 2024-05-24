from warnings import warn
from itertools import chain

import pandas as pd
from datasets import Dataset

from ..helpers import safe_getitem
from ..models.conv import ConvProps
from .base import BaseFormat, BaseConfig
from ..constants import MessageRole as Role


class ConvConfig(BaseConfig):
    pass

class ConversationalFormat(BaseFormat):
    """Converts one or multiple columns of conversational data to the standard format, irrespective of the current format"""
    def __init__(self, dataset: Dataset, config: ConvConfig = ConvConfig()):
        super().__init__(dataset, config)
        self.config: ConvConfig
        self.conv_props = self.get_conv_props()
        
    def get_conv_props(self) -> list[ConvProps]:
        conv_props = [ConvProps(column=column) for column in self.get_conv_columns()]
        conv_props = [self._get_conv_prop(conv_prop) for conv_prop in conv_props]
        return [conv_prop for conv_prop in conv_props if conv_prop.is_valid]

    def _get_conv_prop(self, conv_prop: ConvProps):
        if isinstance(conv_prop.column, str):
            conv_df = pd.DataFrame(chain(*self.dataset[conv_prop.column][:1000]))
            role_key, content_key = self._get_role_and_content_key(conv_df)
            conv_prop.role_key, conv_prop.content_key = role_key, content_key
            conv_prop.has_system = conv_df[role_key].nunique() == 3
            conv_prop.roles_map = self._get_conv_roles(conv_prop)
        return conv_prop
    
    def _get_role_and_content_key(self, conv_df: pd.DataFrame) -> tuple[str, str]:
        if conv_df.empty or len(conv_df) < 6:
            warn("Not enough messages to determine role and content keys", UserWarning)
        role_key: str = safe_getitem(
            [col for col in conv_df.columns if conv_df[col].nunique() in [1, 2, 3]]
        )
        content_key: str = safe_getitem(
            conv_df.drop(role_key, axis=1).nunique().sort_values().reset_index()["index"], -1
        ) if role_key is not None else None
        return role_key, content_key
    
    def _get_conv_roles(self, conv_prop: ConvProps) -> dict[str, Role]:
        if conv_prop.column is None:
            return {}
        
        roles_df = pd.DataFrame(self.dataset[conv_prop.column])
        roles_df = roles_df.map(lambda x: (x or {}).get(conv_prop.role_key))
        if conv_prop.has_system and roles_df.columns.nunique() >= 3:
            system, user, assistant = (
                roles_df[[0, 1, 2]]
                .value_counts()
                .reset_index()
                .rename(columns={0: "a", 1: "b", 2: "c"})
                .query("(a != b) and (b != c) and (a != c)")
                .sort_values("count")
                .iloc[-1]
                .tolist()[:3]
            )
            conv_roles = {system: Role.SYSTEM, user: Role.USER, assistant: Role.ASSISTANT}
        elif roles_df.columns.nunique() == 2:
            user, assistant = (
                roles_df[[0, 1]]
                .value_counts()
                .sort_values()
                .reset_index()[[0, 1]]
                .iloc[-1]
                .tolist()
            )
            conv_roles = {user: Role.USER, assistant: Role.ASSISTANT}
        elif roles_df.columns.nunique() == 1:
            # TODO: This is the only place where hardcoded values are used. Try to remove these constants
            if roles_df[0].iloc[0] in ["system", "instruction", "instructions"]:
                conv_roles = {roles_df[0].iloc[0]: Role.SYSTEM}
            conv_roles = {roles_df[0].iloc[0]: Role.USER}
        else:
            conv_roles = {}

        return conv_roles

    @property
    def is_this_format(self):
        return any(conv_prop.is_valid for conv_prop in self.conv_props)

    def _format(self):
        single_col = "messages" if len(self.conv_props) == 1 else None
        dataset = self.dataset
        for conv_prop in self.conv_props:
            dataset = dataset.map(
                lambda x: {
                    (single_col or conv_prop.column): conv_prop.standardize(
                        x[conv_prop.column]
                    )
                }
            )
        self.messages_cols += [(single_col or conv_prop.column) for conv_prop in self.conv_props]
        return dataset
