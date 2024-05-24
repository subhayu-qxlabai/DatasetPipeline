"""
This module contains helper functions for type checking.

Functions:
    is_conv_type(value): Checks if value is of type `list[dict[str, str]]`, where each `dict` has 2 or 3 items.
    is_standard_type(value): Checks if value is of type `list[dict[str, str]]`, where each `dict` has exactly 2 items, has `role` and `content` keys, and the `role` value is one of `Role.SYSTEM`, `Role.USER`, or `Role.ASSISTANT`.

Usage Example:
    from app.helpers.types import is_conv_type, is_standard_type
"""

from typing import Iterable
from ..constants import MessageField as Field, MessageRole as Role


def is_conv_type(value):
    """Checks if value is of type `list[dict[str, str]]`, where each `dict` has 2 items"""
    return (
        isinstance(value, Iterable) 
        and len(value) >= 1
        and all(
            isinstance(item, dict)
            and all(
                isinstance(k, (str, type(None))) 
                and isinstance(v, (str, type(None))) 
                for k, v in item.items()
            )
            and len(item) in [2, 3]
            for item in value
        )
    )


def is_standard_type(value: list[dict[str, str]]):
    """
    Checks if value is of type `list[dict[str, str]]`
        - where each `dict` has exactly 2 items
        - where each `dict` has `role` and `content` keys
        - where `role` is the value of either `Role.SYSTEM`, `Role.USER` or `Role.ASSISTANT`
    """
    return (
        isinstance(value, list) 
        and len(value) >= 1
        and all(
            isinstance(item, dict)
            and all(
                isinstance(k, (str, type(None))) 
                and isinstance(v, (str, type(None))) 
                for k, v in item.items()
            )
            and {Field.ROLE.value, Field.CONTENT.value} == set(item.keys())
            and item[Field.ROLE.value] in {Role.SYSTEM.value, Role.USER.value, Role.ASSISTANT.value}
            for item in value
        )
    )