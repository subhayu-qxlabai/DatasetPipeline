import re

class RegexDict(dict):
    """
    A dictionary subclass that allows retrieving values based on regular expression matches.

    Example:
    >>> regex_dict = RegexDict({
    ...     r'^\d+$': 'Number',
    ...     r'\d+.\d+': 'Float',
    ...     r'[a-z]+': 'Lowercase',
    ...     r'[A-Z]+': 'Uppercase',
    ... })
    >>> regex_dict['3.14']  # Matches '\d+.\d+' pattern
    'Float'
    >>> regex_dict['1234']  # Matches '^\d+$' pattern
    'Number'
    >>> regex_dict['abcd']  # Matches '[a-z]+' pattern
    'Lowercase'
    >>> regex_dict['WXYZ']  # Matches '[A-Z]+' pattern
    'Uppercase'
    """

    def __getitem__(self, item):
        """
        Retrieve the value associated with the first regular expression pattern that matches the given item.

        Args:
            item: The key to search for.

        Returns:
            The value associated with the first matching regular expression pattern.

        Raises:
            KeyError: If no regular expression pattern matches the given item.
        """
        for pattern, value in self.items():
            if re.search(pattern, item, re.IGNORECASE):
                return value
        raise KeyError

    def get(self, item, default=None):
        """
        Retrieve the value associated with the first regular expression pattern that matches the given item,
        or return a default value if no match is found.

        Args:
            item: The key to search for.
            default: The value to return if no matching regular expression pattern is found (default is None).

        Returns:
            The value associated with the first matching regular expression pattern, or the default value if no match is found.

        Example:
        >>> regex_dict = RegexDict({
        ...     r'\d+': 'Number',
        ...     r'[a-z]+': 'Lowercase',
        ...     r'[A-Z]+': 'Uppercase'
        ... })
        >>> regex_dict.get('123')  # Matches '\d+' pattern
        'Number'
        >>> regex_dict.get('abc')  # Matches '[a-z]+' pattern
        'Lowercase'
        >>> regex_dict.get('XYZ')  # Matches '[A-Z]+' pattern
        'Uppercase'
        >>> regex_dict.get('xyz', 'Not Found')  # No match, returns default value
        'Not Found'
        """
        try:
            return self[item]
        except KeyError:
            return default
