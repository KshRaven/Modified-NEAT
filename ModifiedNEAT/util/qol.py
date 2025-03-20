

from typing import Union, Any


def manage_params(dictionary: dict[str, Any], param: Union[str, list[str]], default: Any = None):
    if isinstance(param, str):
        param = [param]
    for param in param:
        if param in dictionary:
            return dictionary[param]
    return default


class Indexer(object):
    def __init__(self, start=0):
        self.current = int(start)

    def set(self, index: int):
        self.current = index

    def update(self, count: int):
        self.current += count

    def get(self):
        return self.current

    def __next__(self):
        current = self.current
        self.current += 1
        return current

    def __repr__(self):
        return f"Indexer(at {self.current})"
