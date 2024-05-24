"""
This module defines classes for messages in a chat application.

Classes:
    Message: Base class for messages with `role` and `content` attributes.
    SystemMessage: Subclass of `Message` with `role` set to "system".
    UserMessage: Subclass of `Message` with `role` set to "user".
    AssistantMessage: Subclass of `Message` with `role` set to "assistant".
    Messages: Container class for a list of `Message` objects.

Usage Example:

    from app.models.messages import Message, SystemMessage, UserMessage, AssistantMessage, Messages

    # Create a list of messages
    messages = [
        SystemMessage(content="Welcome to the chat!"),
        UserMessage(content="Hello!"),
        AssistantMessage(content="Hi there!"),
    ]

    # Create a `Messages` object from the list of messages
    messages_obj = Messages(messages=messages)

    # Get the number of messages
    num_messages = len(messages_obj)

    # Access individual messages by index
    first_message = messages_obj[0]

    # Iterate over all messages
    for message in messages_obj:
        print(message.content)

    # Check if a message is in the `Messages` object
    if UserMessage(content="Hello!") in messages_obj:
        print("Message found!")

    # Get the hash of the messages
    messages_hash = hash(messages_obj)

    # Get the JSON representation of the messages
    messages_json = messages_obj.model_dump_json()
"""

import json

from pydantic import model_validator

from .base import BaseModel
from ..constants import MessageRole
from ..helpers.utils import hash_uuid


def json_dumps_or_str(data):
    if isinstance(data, str):
        return data
    if isinstance(data, BaseModel):
        return data.model_dump_json()
    try:
        return json.dumps(data)
    except:
        return str(data)


class Message(BaseModel):
    role: str
    content: str

    @model_validator(mode="before")
    @classmethod
    def validate(cls, values: dict):
        roles = {x.value for x in MessageRole}
        if isinstance(values["role"], MessageRole):
            values["role"] = values["role"].value
        assert values["role"] in roles, f"Value of `role` must be in the following: {', '.join(roles)}"
        
        if not isinstance(values["content"], str):
            values["content"] = json_dumps_or_str(values["content"])
        return values


class SystemMessage(Message):
    role: str = "system"


class UserMessage(Message):
    role: str = "user"


class AssistantMessage(Message):
    role: str = "assistant"


class Messages(BaseModel):
    messages: list[Message] = []

    def to_list(self) -> list[dict[str, str]]:
        return self.model_dump(mode="json")["messages"]
    
    @classmethod
    def from_list(cls, messages: list[dict[str, str]]):
        return cls(messages=[Message(**x) for x in messages])

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, index: int) -> Message:
        return self.messages[index]

    def __iter__(self):
        return iter(self.messages)

    def __contains__(self, item):
        return item in self.messages

    def __hash__(self):
        return hash_uuid(
            "|".join([x.content for x in self.messages if x.role != "system"])
        ).int

    def __repr__(self):
        return self.model_dump_json()
