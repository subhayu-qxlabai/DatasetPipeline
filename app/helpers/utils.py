import re
import uuid
import traceback
from pathlib import Path
from random import randrange
from datetime import datetime
from concurrent import futures
from typing import Any, Iterable, Callable, MutableMapping, NamedTuple

from fuzzywuzzy import process
from ruamel.yaml import CommentedMap


def safe_getitem(iterable: Iterable, key_or_index: int | str = 0, default: Any = None):
    """
    Get the value associated with the given key or index from the iterable.

    Parameters:
        iterable (Iterable): The iterable object to retrieve the value from.
        key_or_index (int | str, optional): The key or index to retrieve the value for. Defaults to 0.
        default (Any, optional): The default value to return if the key or index is not found. Defaults to None.

    Returns:
        Any: The value associated with the given key or index, or the default value if not found.
    """
    if isinstance(iterable, dict):
        return iterable.get(key_or_index, default)
    try:
        return iterable[key_or_index]
    except Exception:
        try:
            return list(iterable)[key_or_index]
        except Exception:
            pass
    return default

def get_openai_rate_limit_seconds(error_text: str) -> int:
    try:
        return int((re.findall(r"\s(\d+)\s(?:sec|min)?", error_text) or ['60'])[-1])
    except:
        return 60

def hash_uuid(text: str, base_uuid: uuid.UUID | None = None):
    """
    Generates a hash-based UUID using the given text and an optional base UUID.

    Args:
        text (str): The text to be used for generating the hash-based UUID.
        base_uuid (uuid.UUID | None, optional): The optional base UUID to use for hashing. Defaults to None.

    Returns:
        uuid.UUID: The hash-based UUID generated from the input text and base UUID.
    """
    if not isinstance(base_uuid, uuid.UUID):
        base_uuid = uuid.NAMESPACE_DNS
    return uuid.uuid3(base_uuid, text)


def get_trace(e: Exception, n: int = 5):
    """Get the last n lines of the traceback for an exception"""
    return "".join(traceback.format_exception(e)[-n:])

def run_parallel_exec(exec_func: Callable, iterable: Iterable, *func_args, **kwargs):
    """
    Runs the `exec_func` function in parallel for each element in the `iterable` using a thread pool executor.
    
    Parameters:
        exec_func (Callable): The function to be executed for each element in the `iterable`.
        iterable (Iterable): The collection of elements for which the `exec_func` function will be executed.
        *func_args: Additional positional arguments to be passed to the `exec_func` function.
        **kwargs: Additional keyword arguments to customize the behavior of the function.
            - max_workers (int): The maximum number of worker threads in the thread pool executor. Default is 100.
            - quiet (bool): If True, suppresses the traceback printing for exceptions. Default is False.
            - error_logger (Callable[[str], None]): A function to print/log any error messages. Default is `print`.
    
    Returns:
        list[tuple]: A list of tuples where each tuple contains the element from the `iterable` and the result of executing the `exec_func` function on that element.

    Example:
        >>> from app.utils.helpers import run_parallel_exec
        >>> run_parallel_exec(str, [1, 2, 3])
        [(1, '1'), (2, '2'), (3, '3')]
        >>> run_parallel_exec(len, ["Rahul", "Sachin", "Dhoni"])
        [('Sachin', 6), ('Rahul', 5), ('Dhoni', 5)]
    """
    error_logger: Callable[[str], None] = kwargs.pop("error_logger", print)
    with futures.ThreadPoolExecutor(
        max_workers=kwargs.pop("max_workers", 100)
    ) as executor:
        # Start the load operations and mark each future with each element
        future_element_map = {
            executor.submit(exec_func, element, *func_args): element
            for element in iterable
        }
        result: list[tuple] = []
        for future in futures.as_completed(future_element_map):
            element = future_element_map[future]
            try:
                data = future.result()
            except Exception as exc:
                log_trace = exc if kwargs.pop("quiet", False) else get_trace(exc, 3)
                error_logger(f"Got error while running parallel_exec: {element}: \n{log_trace}")
                result.append((element, exc))
            else:
                result.append((element, data))
        return result
    
def run_parallel_exec_but_return_in_order(exec_func: Callable, iterable: Iterable, *func_args, **kwargs):
    """
    Runs the `exec_func` function in parallel for each element in the `iterable` using a thread pool executor.
    Returns the result in the same order as the `iterable`.
    
    Example:
        >>> from app.utils.helpers import run_parallel_exec_but_return_in_order
        >>> run_parallel_exec_but_return_in_order(str, [1, 2, 3])
        ['1', '2', '3']
        >>> run_parallel_exec_but_return_in_order(len, ["Rahul", "Sachin", "Dhoni"])
        [5, 6, 5]
    """
    # note this is usable only when iterable has types that are hashable
    result = run_parallel_exec(exec_func, iterable:=list(iterable), *func_args, **kwargs)

    # sort the result in the same order as the iterable
    result.sort(key=lambda x: iterable.index(x[0]))

    return [x[-1] for x in result]

def remove_backticks(text: str) -> str:
    return re.sub(r"```\w+\n(.*)\n```", r"\1", text, flags=re.DOTALL)

def remove_comments(text: str) -> str:
    return re.sub(r'\s+//\s+.*', '', text)

def clean_json_str(text: str) -> str:
    cleaned_text = remove_backticks(text)
    cleaned_text = remove_comments(cleaned_text)
    return cleaned_text

def clean_yaml_str(text: str) -> str:
    cleaned_text = remove_backticks(text)
    return cleaned_text

class Match(NamedTuple):
    text: str
    score: float
    
    def as_tuple(self) -> tuple[str, float]:
        return self.text, self.score

def find_best_match(query: str, options: list[str], cutoff: int = 0):
    """Find the best match from a list of options"""
    return Match(*(process.extractOne(query, options, score_cutoff=cutoff) or (None, 0)))

def get_match_score(q1: str, q2: str) -> int:
    return find_best_match(q1, [q2], cutoff=0).score

def recursive_string_operator(
    data, fn: Callable[[str], str], skip_keys: list[str] = [], max_workers=4
):
    """
    Recursively applies the given function to the input data, handling strings, lists, tuples, sets, dictionaries, and BaseModel objects.

    Args:
        data: The input data to be processed.
        fn: The function to be applied to the data.
        skip_keys: A list of keys to be skipped when processing dictionaries.
        max_workers: The maximum number of workers for parallel execution. Defaults to 4.

    Returns:
        The processed data in the same format as the input.
        
    Note:
        The `fn` function should take a single argument (a string) and return a string. 
        Also, any Exceptions raised by the `fn` function should be caught and handled appropriately.
        
        The `skip_keys` parameter works only when the input data is a dictionary or a BaseModel object. 
    
    Example:
        >>> data = "Hello, World!"
        >>> fn = lambda x: x.upper()
        >>> recursive_string_operator(data, fn)
        'HELLO, WORLD!'
        >>> data = {"a": 1, "b": [2, "hello", ["world"]], "c": {"d": "hi", "e": "hello"}, "f": "world", "g": fn}
        >>> recursive_string_operator(data, fn, skip_keys=["d", "f"])
        {'a': 1, 'b': [2, 'HELLO', ['WORLD']], 'c': {'d': 'hi', "e": "HELLO"}, "f": "world", "g": <function __main__.<lambda>(x)>}
    """
    if isinstance(data, str):
        return fn(data)
    base_parallel_func = lambda _data: recursive_string_operator(
        data=_data, fn=fn, skip_keys=skip_keys or [], max_workers=max_workers
    )
    if isinstance(data, (list, tuple, set)):
        are_all_strings = all([isinstance(x, str) for x in data])
        if are_all_strings:
            _combined = "||".join(data)
            if len(_combined) < 1000:
                _operated = fn(_combined).split("||")
                if len(_operated) == len(data):
                    return _operated
        return [
            x
            for x in run_parallel_exec_but_return_in_order(
                base_parallel_func, data, max_workers=max_workers
            )
        ]
    if isinstance(data, dict):
        v_return_tuples = run_parallel_exec(
            base_parallel_func,
            [v for k, v in data.items() if k not in skip_keys],
            max_workers=max_workers,
        )
        v_return_tuples.sort(
            key=lambda x: [v for k, v in data.items() if k not in skip_keys].index(x[0])
        )
        return {
            k: (
                ([y for x, y in v_return_tuples if x == v] or [v])[0]
                if k not in skip_keys
                else v
            )
            for k, v in data.items()
        }
    return data

def recursive_dict_operator(d: dict, fn: Callable[[dict], MutableMapping]):
    has_dict = any(isinstance(v, dict) for _, v in d.items())
    if has_dict:
        return fn(
            {
                k: recursive_dict_operator(v, fn) if isinstance(v, dict) else v
                for k, v in d.items()
            }
        )
    else:
        return fn(d)


def get_timestamp_uid(make_uuid=True):
    """Get a unique id for a timestamp. If `make_uuid` is True, an UUID will be generated from the timestamp."""
    uid = datetime.now().strftime('%Y%m%d%H%M%S%f')
    if make_uuid:
        rndm = str(randrange(10 ** 11, 10 ** 12))
        uid = uuid.UUID(f'{uid[:8]}-{uid[8:12]}-{uid[12:16]}-{uid[16:20]}-{rndm}')
    return uid

def datetime_from_uid(uid: str|uuid.UUID):
    if not isinstance(uid, (str, uuid.UUID)):
        return 
    if isinstance(uid, uuid.UUID):
        uid = str(uid)
    if isinstance(uid, str):
        uid = uid.replace("-", "")[:20]
        assert len(uid) == 20, f"Invalid UID: {uid!r}. Should be at least 20 characters long."
        assert uid.isdigit(), f"Invalid UID: {uid!r}. Should be all digits."
        return datetime.strptime(uid, "%Y%m%d%H%M%S%f")

def get_ts_filename(filepath: str | Path, add_random: bool = True):
    filepath: Path = Path(filepath)
    extra_suffix = f"_{str(randrange(10 ** 11, 10 ** 12))}" if add_random else ""
    filepath = (
        filepath.parent
        / f"{filepath.stem}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}{extra_suffix}{filepath.suffix}"
    )
    return filepath

TS_FILE_STEM_REGEX = re.compile(r"(?P<stem>.+)_(?P<uid>[0-9]{20,})(?:_[0-9]+)?(?P<suffix>\.\w+)")

def datetime_from_tsfile(filepath: str | Path):
    ts = (TS_FILE_STEM_REGEX.findall(Path(filepath).name) or [(None, None)])[0][1]
    return datetime_from_uid(ts)

def parts_from_tsfile(filepath: str | Path) -> dict[str, str | datetime]:
    match = TS_FILE_STEM_REGEX.search(Path(filepath).name)
    if match is None:
        return {}
    return match.groupdict() | {"datetime": datetime_from_uid(match.group("uid"))}


def add_comments(
    data: dict, desc_map: dict, parent_key="", desc_key="__desc__"
):
    indent_spaces=2
    commented_map = CommentedMap()
    for key, value in data.items():
        full_key = f"{parent_key}.{key}" if parent_key else key
        subdesc = desc_map.get(key, {})
        if "__desc__" in subdesc:
            indent = full_key.count(".") * indent_spaces
            commented_map.yaml_set_comment_before_after_key(
                key, before=subdesc["__desc__"], indent=indent
            )
        if isinstance(value, dict):
            commented_map[key] = add_comments(
                value, subdesc, full_key, desc_key=desc_key
            )
        else:
            commented_map[key] = value
    return commented_map
