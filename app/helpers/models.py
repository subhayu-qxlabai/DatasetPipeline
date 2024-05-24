from typing import Any, get_args, get_origin

from pydantic.fields import FieldInfo

from pydantic import BaseModel



def get_field_desc_map(model: BaseModel, desc_key="__desc__") -> dict[str, dict[str, dict[str, dict | str] | str]]:
    """
    Generate a dictionary that maps field names to their descriptions and nested field descriptions.

    Args:
        model (BaseModel): The model to extract field descriptions from.
        desc_key (str, optional): The key to use for the field descriptions. Defaults to "__desc__".

    Returns:
        dict[str, dict[str, dict[str, dict | str] | str]]: A dictionary where the keys are field names and the values
            are dictionaries containing the field descriptions and nested field descriptions.

    Example:
        >>> class MyModel(BaseModel):
        ...     field1: str = Field(description="Description for field1")
        ...     field2: int = Field(description="Description for field2")
        ...     nested_field: NestedModel = Field(description="Description for nested_field")
        ...
        >>> class NestedModel(BaseModel):
        ...     nested_field1: str = Field(description="Description for nested_field1")
        ...     nested_field2: int = Field(description="Description for nested_field2")
        ...
        >>> get_field_desc_map(MyModel())
        {
            'field1': {'__desc__': 'Description for field1'},
            'field2': {'__desc__': 'Description for field2'},
            'nested_field': {
                '__desc__': 'Description for nested_field',
                'nested_field1': {'__desc__': 'Description for nested_field1'},
                'nested_field2': {'__desc__': 'Description for nested_field2'}
            }
        }
    """
    fields: dict[str, FieldInfo] | Any = getattr(model, "model_fields", None)
    if (
        not fields
        or not isinstance(fields, dict)
        or not all(isinstance(x, FieldInfo) for x in fields.values())
    ):
        return fields

    d = {}
    for field_name, field_info in fields.items():
        types = [field_info.annotation, get_origin(field_info.annotation)] + list(
            get_args(field_info.annotation)
        )
        types = tuple(
            (x, get_field_desc_map(x)) for x in types if hasattr(x, "model_fields")
        )
        fields_dict = {desc_key: field_info.description}
        for k, v in types:
            if k is not None:
                fields_dict.update(v)
        
        d[field_name] = fields_dict

    return d
