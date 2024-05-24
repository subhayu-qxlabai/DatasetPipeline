"""
This module provides the `ConvProps` class, which represents properties of a conversation.

Attributes:
    column (str | None): The column name. Defaults to 'null'.
    role_key (str | None): The key for the role. Defaults to 'null'.
    content_key (str | None): The key for the content. Defaults to 'null'.
    has_system (bool): Indicates whether the conversation has a system message. Defaults to False.
    roles_map (dict[str, MessageRole]): Mapping between role keys and MessageRole objects. Defaults to an empty dictionary.

Methods:
    is_valid (property): Checks if the conversation properties are valid. Returns False if the roles_map, role_key, or content_key are empty or None.
    standardize(conv: list[dict[str, str]]): Standardizes the format of the messages in the conversation. Returns a new list of dictionaries with standardized role and content values.

Example usage:
    conv_props = ConvProps(column='conversation', role_key='role', content_key='content')
    conv = [
        {'role': 'system', 'content': 'Hello'},
        {'role': 'user', 'content': 'How are you?'},
        {'role': 'assistant', 'content': 'I'm good, thanks!'},
    ]
    standardized_conv = conv_props.standardize(conv)
    print(standardized_conv)
    # Output:
    # [
    #     {'role': 'system', 'content': 'Hello'},
    #     {'role': 'user', 'content': 'How are you?'},
    #     {'role': 'assistant', 'content': 'I'm good, thanks!'},
    # ]
"""

from pydantic import Field

from .base import BaseModel
from ..constants import MessageRole, MessageField


class ConvProps(BaseModel):
    column: str | None = Field(default=None,description="Column Name. Defaults to 'null'")
    role_key: str | None = Field(default=None,description="Role key. Defaults to 'null'")
    content_key: str | None = Field(default=None,description="Contetnt key. Defaults to 'null'")
    has_system: bool = Field(default=False,description="Check whether it has system message. Defaults to 'False'")
    roles_map: dict[str, MessageRole] = Field(default_factory=dict,description="Mapping between role and role key. Defaults to '{}'")
    
    @property
    def is_valid(self):
        return not (len(self.roles_map) == 0 or self.role_key is None or self.content_key is None)

    def standardize(self, conv: list[dict[str, str]]):
        if not self.is_valid:
            return conv
        return [
            {
                MessageField.ROLE.value: self.roles_map[m[self.role_key]].value,
                MessageField.CONTENT.value: m[self.content_key],
            } 
            for m in conv
        ]
    