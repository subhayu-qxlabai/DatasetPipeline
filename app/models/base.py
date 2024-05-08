import json
import yaml
from pathlib import Path
from typing import Any, Literal

from pydantic.main import IncEx
from pydantic import BaseModel as PydanticBaseModel

from ..helpers.utils import clean_json_str, find_best_match

PathLike = str | Path

class BaseModel(PydanticBaseModel):
    def __str__(self):
        return str(self.to_dict())
    
    # def __repr__(self):
    #     return str(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, float|int|str], fuzzy=True, cutoff: float = 0.0):
        """
        Creates a BaseModel object from a dictionary.

        Args:
            data: A dictionary containing the data to create the BaseModel object.
            fuzzy: A boolean indicating whether to perform fuzzy matching.
            cutoff: A float indicating the cutoff score for fuzzy matching. Must be a float between 0 and 1.

        Returns:
            BaseModel: The created BaseModel object.
        """
        if not fuzzy:
            return cls(**data)
        if not isinstance(cutoff, float) or cutoff > 1 or cutoff < 0:
            cutoff = 0.0
        field_col_map = {
            field: find_best_match(field, list(data)).as_tuple()
            for field in cls.model_fields
        }
        data = {
            field: data[col] 
            for field, (col, score) in field_col_map.items() 
            if score >= cutoff
        }
        return cls(**data)
    
    @classmethod
    def from_json(cls, data: str, fuzzy=False, cutoff: float = 0.0):
        """
        Creates a BaseModel object from a JSON string.

        Args:
            data: A JSON string containing the data to create the BaseModel object.
            fuzzy: A boolean indicating whether to perform fuzzy matching.
            cutoff: A float indicating the cutoff score for fuzzy matching. Must be a float between 0 and 1.

        Returns:
            BaseModel: The created BaseModel object.
        """
        return cls.from_dict(json.loads(clean_json_str(data)), fuzzy)
    
    @classmethod
    def from_yaml(cls, data: str, fuzzy=False, cutoff: float = 0.0):
        """
        Creates a BaseModel object from a YAML string.

        Args:
            data: A YAML string containing the data to create the BaseModel object.
            fuzzy: A boolean indicating whether to perform fuzzy matching.
            cutoff: A float indicating the cutoff score for fuzzy matching. Must be a float between 0 and 1.

        Returns:
            BaseModel: The created BaseModel object.
        """
        return cls.from_dict(yaml.safe_load(data), fuzzy)
    
    @classmethod
    def from_file(cls, path: PathLike, fuzzy=False, cutoff: float = 0.0):
        """
        Creates a BaseModel object from a file.

        Args:
            path: A path to the file containing the data to create the BaseModel object.
            fuzzy: A boolean indicating whether to perform fuzzy matching.
            cutoff: A float indicating the cutoff score for fuzzy matching. Must be a float between 0 and 1.

        Returns:
            BaseModel: The created BaseModel object.
        """
        path = Path(path)
        data = path.read_text()
        
        if path.suffix == ".json":
            return cls.from_json(data, fuzzy, cutoff) 
        elif path.suffix in [".yml", ".yaml"]:
            return cls.from_yaml(data, fuzzy, cutoff)
        else:
            raise ValueError("Invalid file format. Must be .json or .yaml.")
    
    def to_dict(
        self,
        mode: Literal['json', 'python'] = 'json',
        include: IncEx = None,
        exclude: IncEx = None,
        context: dict[str, Any] | None = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: bool | Literal['none', 'warn', 'error'] = True,
        serialize_as_any: bool = False
    ):
        """
        Generate a dictionary representation of the model, optionally specifying which fields to include or exclude.

        Args:
            mode: The mode in which `to_python` should run.
                If mode is 'json', the output will only contain JSON serializable types.
                If mode is 'python', the output may contain non-JSON-serializable Python objects.
            include: A set of fields to include in the output.
            exclude: A set of fields to exclude from the output.
            context: Additional context to pass to the serializer.
            by_alias: Whether to use the field's alias in the dictionary key if defined.
            exclude_unset: Whether to exclude fields that have not been explicitly set.
            exclude_defaults: Whether to exclude fields that are set to their default value.
            exclude_none: Whether to exclude fields that have a value of `None`.
            round_trip: If True, dumped values should be valid as input for non-idempotent types such as Json[T].
            warnings: How to handle serialization errors. False/"none" ignores them, True/"warn" logs errors,
                "error" raises a [`PydanticSerializationError`][pydantic_core.PydanticSerializationError].
            serialize_as_any: Whether to serialize fields with duck-typing serialization behavior.

        Returns:
            A dictionary representation of the model.
        """
        return self.model_dump(
            mode=mode,
            include=include,
            exclude=exclude,
            context=context,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            round_trip=round_trip,
            warnings=warnings,
            serialize_as_any=serialize_as_any,
        )

    def to_json(
        self, 
        indent=4, 
        sort_keys=False,
        include: IncEx = None,
        exclude: IncEx = None,
        context: dict[str, Any] | None = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: bool | Literal['none', 'warn', 'error'] = True,
        serialize_as_any: bool = False
    ):
        """
        Generate a JSON representation of the model, optionally specifying which fields to include or exclude.

        Args:
            indent: The indentation level to use when serializing the model.
            sort_keys: Whether to sort the keys in the output.
            include: A set of fields to include in the output.
            exclude: A set of fields to exclude from the output.
            context: Additional context to pass to the serializer.
            by_alias: Whether to use the field's alias in the dictionary key if defined.
            exclude_unset: Whether to exclude fields that have not been explicitly set.
            exclude_defaults: Whether to exclude fields that are set to their default value.
            exclude_none: Whether to exclude fields that have a value of `None`.
            round_trip: If True, dumped values should be valid as input for non-idempotent types such as Json[T].
            warnings: How to handle serialization errors. False/"none" ignores them, True/"warn" logs errors,
                "error" raises a [`PydanticSerializationError`][pydantic_core.PydanticSerializationError].
            serialize_as_any: Whether to serialize fields with duck-typing serialization behavior.

        Returns:
            A JSON representation of the model in string format.
        """
        d = self.to_dict(
            mode="json",
            include=include,
            exclude=exclude,
            context=context,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            round_trip=round_trip,
            warnings=warnings,
            serialize_as_any=serialize_as_any,
        )
        return json.dumps(d, indent=indent, sort_keys=sort_keys)
    
    def to_yaml(
        self, 
        indent=4, 
        sort_keys=False, 
        include: IncEx = None,
        exclude: IncEx = None,
        context: dict[str, Any] | None = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: bool | Literal['none', 'warn', 'error'] = True,
        serialize_as_any: bool = False
    ):
        """
        Generate a YAML representation of the model, optionally specifying which fields to include or exclude.

        Args:
            indent: The indentation level to use when serializing the model.
            sort_keys: Whether to sort the keys in the output.
            include: A set of fields to include in the output.
            exclude: A set of fields to exclude from the output.
            context: Additional context to pass to the serializer.
            by_alias: Whether to use the field's alias in the dictionary key if defined.
            exclude_unset: Whether to exclude fields that have not been explicitly set.
            exclude_defaults: Whether to exclude fields that are set to their default value.
            exclude_none: Whether to exclude fields that have a value of `None`.
            round_trip: If True, dumped values should be valid as input for non-idempotent types such as Json[T].
            warnings: How to handle serialization errors. False/"none" ignores them, True/"warn" logs errors,
                "error" raises a [`PydanticSerializationError`][pydantic_core.PydanticSerializationError].
            serialize_as_any: Whether to serialize fields with duck-typing serialization behavior.

        Returns:
            A YAML representation of the model in string format.
        """
        d = self.to_dict(
            mode="json",
            include=include,
            exclude=exclude,
            context=context,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            round_trip=round_trip,
            warnings=warnings,
            serialize_as_any=serialize_as_any,
        )
        return yaml.safe_dump(d, indent=indent, sort_keys=sort_keys)
    
    def to_file(
        self, 
        path: PathLike, 
        indent=4, 
        sort_keys=False, 
        include: IncEx = None,
        exclude: IncEx = None,
        context: dict[str, Any] | None = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: bool | Literal['none', 'warn', 'error'] = True,
        serialize_as_any: bool = False
    ):
        """
        Save the model to a file (in JSON or YAML), inferring the format from the file extension.

        Args:
            path: The path to the file to save the model.
            indent: The indentation level to use when serializing the model.
            sort_keys: Whether to sort the keys in the output.
            include: A set of fields to include in the output.
            exclude: A set of fields to exclude from the output.
            context: Additional context to pass to the serializer.
            by_alias: Whether to use the field's alias in the dictionary key if defined.
            exclude_unset: Whether to exclude fields that have not been explicitly set.
            exclude_defaults: Whether to exclude fields that are set to their default value.
            exclude_none: Whether to exclude fields that have a value of `None`.
            round_trip: If True, dumped values should be valid as input for non-idempotent types such as Json[T].
            warnings: How to handle serialization errors. False/"none" ignores them, True/"warn" logs errors,
                "error" raises a [`PydanticSerializationError`][pydantic_core.PydanticSerializationError].
            serialize_as_any: Whether to serialize fields with duck-typing serialization behavior.
        """
        common_params = dict(
            indent=indent,
            sort_keys=sort_keys,
            include=include,
            exclude=exclude,
            context=context,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            round_trip=round_trip,
            warnings=warnings,
            serialize_as_any=serialize_as_any,
        )
        path = Path(path)
        if path.suffix == ".json":
            path.write_text(self.to_json(**common_params))
        elif path.suffix in [".yml", ".yaml"]:
            path.write_text(self.to_yaml(**common_params))
        else:
            raise ValueError("Invalid file format. Must be .json or .yaml.")
    
    def __hash__(self) -> int:
        return hash(self.to_yaml())

    def __eq__(self, other: "BaseModel") -> bool:
        return self.to_yaml() == other.to_yaml()
    