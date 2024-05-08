from enum import Enum


class MessageField(str, Enum):
    ROLE = "role"
    CONTENT = "content"

class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class DPOColumns(str, Enum):
    CHOSEN = "chosen"
    REJECTED = "rejected"
    USER = "user"
    SYSTEM = "system"
    