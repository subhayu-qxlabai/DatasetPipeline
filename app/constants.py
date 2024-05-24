"""
This module defines enumerations for message fields, message roles, and DPO columns.

Classes:
    MessageField(str, Enum):
        An enumeration of strings representing the fields in a message object.
        The fields are `ROLE` and `CONTENT`.

    MessageRole(str, Enum):
        An enumeration of strings representing the roles in a message object.
        The roles are `SYSTEM`, `USER`, and `ASSISTANT`.

    DPOColumns(str, Enum):
        An enumeration of strings representing the columns in a dataset used for DPO (Dialogue Policy Optimization).
        The columns are `CHOSEN`, `REJECTED`, `USER`, and `SYSTEM`.

Example usage:

```python
from constants import MessageField, MessageRole, DPOColumns

print(MessageField.ROLE)  # Output: 'role'
print(MessageRole.USER)  # Output: 'user'
print(DPOColumns.SYSTEM)  # Output: 'system'
```
"""

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
    