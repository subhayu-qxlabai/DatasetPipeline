from typing import Any, get_args, get_origin

from pydantic.fields import FieldInfo

from pydantic import BaseModel



def get_field_desc_map(model: BaseModel, desc_key="__desc__") -> dict[str, dict[str, dict[str, dict | str] | str]]:
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
