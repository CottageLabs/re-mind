from itertools import tee
from typing import Iterable


def peek(values: Iterable):
    a, b = tee(values)
    first = next(b, None)
    return first


