

from typing import Union, Any


def manage_params(dictionary: dict[str, Any], param: Union[str, list[str]], default: Any = None):
    if isinstance(param, str):
        param = [param]
    for param in param:
        if param in dictionary:
            return dictionary[param]
    return default
