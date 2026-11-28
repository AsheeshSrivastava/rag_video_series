"""The index record, build and search (spec 0001, sections 3 and 4)."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import batched
from typing import Protocol

import numpy as np

from .embeddings import (
    ConfigurationError,
    Identity,
    Registry,
    embed,
    is_valid_revision,
    load_registry,
)


class IndexRecordError(Exception):
    """A missing or unreadable record, or a build into a store that has one."""


class ModelMismatchError(Exception):
    def __init__(self, expected: Identity, actual: Identity) -> None:
        super().__init__(
            f"model mismatch: expected {expected.model} "
            f"(revision {expected.revision}, dimension {expected.dimension}), "
            f"got {actual.model} "
            f"(revision {actual.revision}, dimension {actual.dimension})"
        )
        self.expected = expected
        self.actual = actual


@dataclass(frozen=True, kw_only=True)
class IndexRecord:
    record_version: int
    model: str
    revision: str | None
    dimension: int

    @classmethod
    def from_identity(cls, identity: Identity) -> IndexRecord:
        return cls(
            record_version=1,
            model=identity.model,
            revision=identity.revision,
            dimension=identity.dimension,
        )


def record_to_json(record: IndexRecord) -> str:
    return json.dumps(
        {
            "record_version": record.record_version,
            "model": record.model,
            "revision": record.revision,
            "dimension": record.dimension,
        }
    )


def record_from_json(text: str, registry: Registry) -> IndexRecord:
    try:
        data = json.loads(text)
    except ValueError:
        raise IndexRecordError("index record is not valid JSON") from None
    keys = {"record_version", "model", "revision", "dimension"}
    if not isinstance(data, dict) or set(data) != keys:
        raise IndexRecordError(f"index record must be an object with exactly {sorted(keys)}")
    version, model, revision, dimension = (data[k] for k in ("record_version", "model", "revision", "dimension"))
    # type(...) is int, because bool is a subclass of int
    if type(version) is not int or version != 1:
        raise IndexRecordError("index record_version must be the integer 1")
    if not isinstance(model, str) or not (revision is None or isinstance(revision, str)):
        raise IndexRecordError("index record model must be a string and revision a string or null")
    if type(dimension) is not int:
        raise IndexRecordError("index record dimension must be an integer")
    try:
        spec = registry[model]
    except KeyError:
        raise IndexRecordError(f"index record model {model!r} is not in the registry") from None
    if not is_valid_revision(model, revision):
        raise IndexRecordError(f"index record revision {revision!r} is not allowed for {model}")
    if dimension != spec.dimension:
        raise IndexRecordError(
            f"index record dimension {dimension} differs from the registry's {spec.dimension}"
        )
    return IndexRecord(record_version=1, model=model, revision=revision, dimension=dimension)


class Store(Protocol):
    def add(self, vectors: np.ndarray, payloads: list[str]) -> None: ...
    def vectors(self) -> np.ndarray: ...
    def write_record(self, text: str) -> None: ...
    def read_record(self) -> str | None: ...
    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]: ...


def build_index(
    model_id: str,
    texts: Iterable[str],
    store: Store,
    *,
    batch_size: int = 64,
    registry: Registry | None = None,
    backend_factory: Callable | None = None,
    embed_fn: Callable | None = None,
) -> IndexRecord:
    embed_fn = embed_fn or embed
    if store.read_record() is not None:
        raise IndexRecordError("the store already has an index record")
    first: Identity | None = None
    for batch in batched(texts, batch_size):
        result = embed_fn(
            model_id, list(batch), registry=registry, backend_factory=backend_factory
        )
        if first is None:
            first = result.identity
        elif result.identity != first:
            raise ModelMismatchError(first, result.identity)
        store.add(result.vectors, list(batch))
    if first is None:
        # B6: a record for an empty index would look like a valid index
        raise ConfigurationError("cannot build an index from no texts")
    record = IndexRecord.from_identity(first)
    store.write_record(record_to_json(record))
    return record


def search(
    question: str,
    store: Store,
    k: int,
    *,
    registry: Registry | None = None,
    backend_factory: Callable | None = None,
    embed_fn: Callable | None = None,
) -> list[tuple[str, float]]:
    embed_fn = embed_fn or embed
    registry = registry or load_registry()
    text = store.read_record()
    if text is None:
        raise IndexRecordError("the index has no record")
    record = record_from_json(text, registry)
    result = embed_fn(
        record.model, [question], registry=registry, backend_factory=backend_factory
    )
    expected = Identity(
        model=record.model, revision=record.revision, dimension=record.dimension
    )
    if result.identity != expected:
        raise ModelMismatchError(expected, result.identity)
    return store.search(result.vectors[0], k)
