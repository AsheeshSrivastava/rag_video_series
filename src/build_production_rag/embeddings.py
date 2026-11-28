"""The embedding function, the model registry and the backends (spec 0001).

Importing this module never imports torch or transformers. The local backend
lives in ``local_backend`` and is loaded the first time ``LocalBackend`` is used.
"""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

OPENAI_ID = "text-embedding-3-small"
KNOWN_IDS = (
    OPENAI_ID,
    "jinaai/jina-embeddings-v2-base-en",
    "BAAI/bge-m3",
)

_HEX40 = re.compile(r"[0-9a-f]{40}")
_OPENAI_MAX_INPUTS = 2048
_OPENAI_MAX_TOKENS = 300_000
_OPENAI_RETRIES = 2


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class EmbeddingError(Exception):
    """Base class of the two embedding failures."""


class ConfigurationError(EmbeddingError):
    """The call can never succeed as asked. Retrying will not help."""

    def __init__(
        self,
        message: str,
        *,
        position: int | None = None,
        token_count: int | None = None,
    ) -> None:
        super().__init__(message)
        self.position = position
        self.token_count = token_count


class EmbeddingUnavailable(EmbeddingError):
    """The call might succeed later."""


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class ModelSpec:
    id: str
    revision: str | None
    dimension: int
    max_input_tokens: int
    remote_code_repo: str | None
    remote_code_revision: str | None


@dataclass(frozen=True, kw_only=True)
class Identity:
    model: str
    revision: str | None
    dimension: int


@dataclass(frozen=True, kw_only=True)
class Embeddings:
    vectors: np.ndarray
    identity: Identity


def is_valid_revision(model_id: str, revision: object) -> bool:
    """M3 for the OpenAI model (must be None), M2 for every other model."""
    if model_id == OPENAI_ID:
        return revision is None
    return isinstance(revision, str) and _HEX40.fullmatch(revision) is not None


class Registry:
    def __init__(self, specs: Iterable[ModelSpec]) -> None:
        self._specs = {spec.id: spec for spec in specs}

    def __getitem__(self, model_id: str) -> ModelSpec:
        return self._specs[model_id]

    def ids(self) -> list[str]:
        return list(self._specs)

    @classmethod
    def from_toml(cls, path: str | Path) -> Registry:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
        specs = []
        for entry in data.get("model", []):
            spec = ModelSpec(
                id=entry["id"],
                revision=entry.get("revision") or None,
                dimension=entry["dimension"],
                max_input_tokens=entry["max_input_tokens"],
                remote_code_repo=entry.get("remote_code_repo") or None,
                remote_code_revision=entry.get("remote_code_revision") or None,
            )
            if not is_valid_revision(spec.id, spec.revision):
                raise ConfigurationError(
                    f"{spec.id}: revision {spec.revision!r} is not allowed "
                    "(OpenAI must be empty, a local model needs a 40-character "
                    "lowercase hex commit hash)"
                )
            if spec.remote_code_revision is not None and not is_valid_revision(
                "local", spec.remote_code_revision
            ):
                raise ConfigurationError(
                    f"{spec.id}: remote_code_revision {spec.remote_code_revision!r} "
                    "is not a 40-character lowercase hex commit hash"
                )
            specs.append(spec)
        if sorted(spec.id for spec in specs) != sorted(KNOWN_IDS):
            raise ConfigurationError(
                f"the registry must hold exactly these models: {', '.join(KNOWN_IDS)}"
            )
        return cls(specs)


def load_registry() -> Registry:
    return Registry.from_toml(Path("models.toml"))


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------


class OpenAIBackend:
    def __init__(
        self,
        spec: ModelSpec,
        *,
        api_key: str | None = None,
        http_client=None,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ConfigurationError("OPENAI_API_KEY is missing or empty")
        import openai

        self._spec = spec
        self.revision = None
        self._token_counter = token_counter
        self._encoding = None
        self._client = openai.OpenAI(
            api_key=api_key, http_client=http_client, max_retries=_OPENAI_RETRIES
        )

    def count_tokens(self, text: str) -> int:
        if self._token_counter is not None:
            return self._token_counter(text)
        if self._encoding is None:
            import tiktoken

            self._encoding = tiktoken.get_encoding("cl100k_base")
        return len(self._encoding.encode(text))

    def _batches(self, texts: Sequence[str]) -> list[Sequence[str]]:
        # A text never has more tokens than UTF-8 bytes, so when the bytes fit
        # the token limit there is no need to count (and no encoding download).
        if sum(len(t.encode()) for t in texts) <= _OPENAI_MAX_TOKENS:
            weights = None
        else:
            weights = [self.count_tokens(t) for t in texts]
        batches, start, total = [], 0, 0
        for i in range(len(texts)):
            weight = 0 if weights is None else weights[i]
            if i > start and (
                i - start >= _OPENAI_MAX_INPUTS or total + weight > _OPENAI_MAX_TOKENS
            ):
                batches.append(texts[start:i])
                start, total = i, 0
            total += weight
        if len(texts) > start:
            batches.append(texts[start:])
        return batches

    def embed_raw(self, texts: Sequence[str]) -> np.ndarray:
        import openai

        rows = []
        for batch in self._batches(texts):
            try:
                response = self._client.embeddings.create(
                    model=self._spec.id, input=list(batch)
                )
            except openai.APITimeoutError:
                raise EmbeddingUnavailable("OpenAI request timed out") from None
            except openai.APIConnectionError:
                raise EmbeddingUnavailable("OpenAI connection failed") from None
            except openai.APIStatusError as error:
                status = error.status_code
                if status in (408, 409, 429) or status >= 500:
                    raise EmbeddingUnavailable(
                        f"OpenAI request failed with status {status}"
                    ) from None
                raise ConfigurationError(
                    f"OpenAI rejected the request with status {status}"
                ) from None
            data = sorted(response.data, key=lambda item: item.index)
            rows.extend(item.embedding for item in data)
        return np.asarray(rows, dtype=np.float32).reshape(len(rows), -1)


def __getattr__(name: str):
    # LocalBackend is loaded lazily so that importing this module never pulls in torch.
    if name == "LocalBackend":
        from .local_backend import LocalBackend

        return LocalBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_local_backends: dict[tuple[str, str | None, str | None], object] = {}


def default_backend_factory(spec: ModelSpec):
    if spec.id == OPENAI_ID:
        return OpenAIBackend(spec)
    key = (spec.id, spec.revision, spec.remote_code_revision)
    if key not in _local_backends:
        _local_backends[key] = __getattr__("LocalBackend")(spec)
    return _local_backends[key]


# --------------------------------------------------------------------------
# The embedding function
# --------------------------------------------------------------------------


def embed(
    model_id: str,
    texts: Sequence[str],
    *,
    registry: Registry | None = None,
    backend_factory: Callable[[ModelSpec], object] | None = None,
) -> Embeddings:
    if registry is None:
        registry = load_registry()
    try:
        spec = registry[model_id]
    except KeyError:
        raise ConfigurationError(
            f"unknown model {model_id!r}; known models: {', '.join(registry.ids())}"
        ) from None

    for position, text in enumerate(texts):
        if not text.strip():
            raise ConfigurationError(
                f"text at position {position} is empty or only whitespace",
                position=position,
            )

    if len(texts) == 0:
        return Embeddings(
            vectors=np.zeros((0, spec.dimension), dtype=np.float32),
            identity=Identity(
                model=spec.id, revision=spec.revision, dimension=spec.dimension
            ),
        )

    backend = (backend_factory or default_backend_factory)(spec)

    for position, text in enumerate(texts):
        count = backend.count_tokens(text)
        if count > spec.max_input_tokens:
            raise ConfigurationError(
                f"text at position {position} has {count} tokens, "
                f"more than the {spec.max_input_tokens} allowed for {spec.id}",
                position=position,
                token_count=count,
            )

    raw = np.asarray(backend.embed_raw(list(texts)), dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] != len(texts):
        raise ConfigurationError(
            f"{spec.id} returned shape {raw.shape} for {len(texts)} texts"
        )
    if raw.shape[1] != spec.dimension:
        raise ConfigurationError(
            f"{spec.id} returned vectors of length {raw.shape[1]}, "
            f"expected {spec.dimension}"
        )
    if not np.isfinite(raw).all():
        raise ConfigurationError(f"{spec.id} returned a NaN or infinite value")
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ConfigurationError(f"{spec.id} returned a zero vector")

    return Embeddings(
        vectors=(raw / norms).astype(np.float32),
        identity=Identity(
            model=spec.id, revision=backend.revision, dimension=spec.dimension
        ),
    )
