"""Iterable helpers."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TypeVar

T = TypeVar("T")


def chunked(items: Sequence[T], chunk_size: int) -> Iterator[list[T]]:
    """Yield fixed-size list chunks from a sequence."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    for index in range(0, len(items), chunk_size):
        yield list(items[index : index + chunk_size])
