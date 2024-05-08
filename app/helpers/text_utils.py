import re
import json
from typing import Any, Callable


class TextUtils:
    curly_word_pattern = re.compile(r"{(\w*)}")
    group_sub_pattern = re.compile(r"_%_!_@_(\w+)_@_!_%_")

    @staticmethod
    def parse_to_dict(template: str, formatted_text: str):
        """
        A static method to parse the input formatted text into a dictionary based on the given template.
        
        Parameters:
            template (str): The template string to be used for parsing.
            formatted_text (str): The input text to be parsed based on the template.
        
        Returns:
            dict: A dictionary containing the parsed values based on the template.

        Example:
            >>> template = "Hello, {name}!"
            >>> formatted_text = "Hello, John!"
            >>> result = TextUtils.parse_to_dict(template, formatted_text)
            >>> print(result)
            {'name': 'John'}
        """
        if "{}" in template:
            raise ValueError(
                "Template should not contain {}. Only named placeholders are supported."
            )
        encoded: str = re.sub(
            TextUtils.curly_word_pattern, r"_%_!_@_\1_@_!_%_", template
        )
        escaped: str = re.escape(encoded)
        regex: str = re.sub(TextUtils.group_sub_pattern, r"(?P<\1>.+)", escaped)
        match = re.match(f".*{regex}.*", formatted_text, re.DOTALL)
        if not match:
            return {}
        return match.groupdict()

    @staticmethod
    def replace_text(
        text: str, replacements: dict[str, Any], curly_braces: bool = True
    ):
        """
        Replaces text in a string based on the given replacements dictionary.
        
        Args:
            text (str): The original text to perform replacements on.
            replacements (dict[str, Any]): A dictionary containing the replacements to be made in the text.
            curly_braces (bool, optional): Indicates whether the keys in the replacements should be enclosed in curly braces. Defaults to True.
        
        Returns:
            str: The modified text after performing the replacements.

        Example:
            >>> text = "Hello, {name}!"
            >>> replacements = {"name": "John"}
            >>> result = TextUtils.replace_text(text, replacements, curly_braces=True)
            >>> print(result)
            Hello, John!
            >>> result = TextUtils.replace_text(text, replacements, curly_braces=False)
            >>> print(result)
            Hello, {John}!
        """
        
        for key, value in replacements.items():
            if curly_braces:
                key = f"{{{key}}}"
            text = text.replace(key, str(value))
        return text

    @staticmethod
    def get_middle_text(
        text: str, begin: str, end: str, serializer: Callable[[str], Any] | None = None
    ):
        """
        Retrieves the text between the given begin and end strings in the input text.

        Args:
            text (str): The input text to search in.
            begin (str): The beginning delimiter of the text to retrieve.
            end (str): The ending delimiter of the text to retrieve.
            serializer (Callable[[str], Any] | None, optional): A function to process the extracted text. Defaults to None.

        Returns:
            str | None: The extracted text, or None if no match is found.

        Example:
            >>> text = "Hello, John!"
            >>> begin = "Hello, "
            >>> end = "!"
            >>> result = TextUtils.get_middle_text(text, begin, end)
            >>> print(result)
            John
        """
        if len(text) < len(begin):
            raise ValueError("Output is too short to contain the begin")
        pattern = f"{re.escape(begin)}(.+?)\s*{re.escape(end)}"
        matches: list[str] = re.findall(pattern, text)
        if not matches:
            pattern = f"{re.escape(begin)}(.+?)\s*"
            matches = re.findall(pattern, text, re.DOTALL)
        if not matches:
            return
        if callable(serializer):
            try:
                return serializer(matches[0])
            except Exception as e:
                print(f"Error during serialization: {e}. Returning original text")
        return matches[0]

    @staticmethod
    def get_replacement_keys(text: str):
        """
        Get all the keys that can be replaced in a string
        
        Args:
            text (str): The string to search for keys
        
        Example:
            >>> text = "Hello, {name1}!\nMy name is {name2}!"
            >>> get_replacement_keys(text)
            ['name1', 'name2']
        """
        return re.findall(r"\{(\w+?)\}", text)
    