"""Shared fakes, fixtures and helpers for the spec 0001 tests.

Nothing from ``build_production_rag`` is imported at module top level, so a
missing implementation fails each test on its own instead of breaking
collection.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
import pathlib
import time

import numpy as np
import pytest

import httpx2


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

OPENAI_ID = "text-embedding-3-small"
JINA_ID = "jinaai/jina-embeddings-v2-base-en"
BGE_ID = "BAAI/bge-m3"
REAL_IDS = (OPENAI_ID, JINA_ID, BGE_ID)

REV_A = "a" * 40
REV_B = "b" * 40
REV_1 = "1" * 40
REV_2 = "2" * 40

FAKE_DIM = 8
FAKE_MAX_TOKENS = 5

# T4: the embedders every contract test runs against.
EMBEDDER_PARAMS = [
    "fake",
    pytest.param("jina", marks=pytest.mark.model),
    pytest.param("bge-m3", marks=pytest.mark.model),
    pytest.param("openai", marks=pytest.mark.network),
]
EMBEDDER_MODEL_IDS = {
    "fake": "fake-a",
    "jina": JINA_ID,
    "bge-m3": BGE_ID,
    "openai": OPENAI_ID,
}
# Subsets for rules whose Test column does not include every embedder.
E9_PARAMS = [p for p in EMBEDDER_PARAMS if getattr(p, "values", (p,))[0] != "openai"]
E10_PARAMS = [p for p in EMBEDDER_PARAMS if getattr(p, "values", (p,))[0] != "fake"]
E11_PARAMS = [
    p
    for p in EMBEDDER_PARAMS
    if getattr(p, "values", (p,))[0] in ("jina", "bge-m3")
]


# --------------------------------------------------------------------------
# Lazy module access
# --------------------------------------------------------------------------


@pytest.fixture
def emb():
    return importlib.import_module("build_production_rag.embeddings")


@pytest.fixture
def idx():
    return importlib.import_module("build_production_rag.index")


@pytest.fixture
def repo_root(monkeypatch):
    """Repo root, with cwd moved there so ``load_registry()`` finds models.toml."""
    monkeypatch.chdir(REPO_ROOT)
    return REPO_ROOT


@pytest.fixture
def no_sleep(monkeypatch):
    """Skip retry backoff. Patches only ``time.sleep``, not any src internals."""
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)


@pytest.fixture
def fake_specs(emb):
    return {
        "fake-a": emb.ModelSpec(
            id="fake-a",
            revision=REV_A,
            dimension=FAKE_DIM,
            max_input_tokens=FAKE_MAX_TOKENS,
            remote_code_repo=None,
            remote_code_revision=None,
        ),
        "fake-b": emb.ModelSpec(
            id="fake-b",
            revision=REV_B,
            dimension=FAKE_DIM,
            max_input_tokens=FAKE_MAX_TOKENS,
            remote_code_repo=None,
            remote_code_revision=None,
        ),
    }


@pytest.fixture
def fake_registry(emb, fake_specs):
    return emb.Registry(fake_specs.values())


@pytest.fixture
def real_registry(emb, repo_root):
    return emb.load_registry()


# --------------------------------------------------------------------------
# models.toml writer (for M2, M3, F1)
# --------------------------------------------------------------------------

BASE_MODELS = {
    OPENAI_ID: {
        "id": OPENAI_ID,
        "revision": "",
        "dimension": 1536,
        "max_input_tokens": 8192,
    },
    JINA_ID: {
        "id": JINA_ID,
        "revision": "322d4d7e2f35e84137961a65af894fda0385eb7a",
        "dimension": 768,
        "max_input_tokens": 8192,
        "remote_code_repo": "jinaai/jina-bert-implementation",
        "remote_code_revision": "f3ec4cf7de7e561007f27c9efc7148b0bd713f81",
    },
    BGE_ID: {
        "id": BGE_ID,
        "revision": "5617a9f61b028005a4858fdac845db406aefb181",
        "dimension": 1024,
        "max_input_tokens": 8192,
    },
}


def write_models_toml(path, overrides=None):
    """Write a three-model file in the models.toml format.

    ``overrides`` maps model id -> dict of keys to change.
    """
    overrides = overrides or {}
    lines = []
    for model_id, base in BASE_MODELS.items():
        entry = {**base, **overrides.get(model_id, {})}
        lines.append("[[model]]")
        for key, value in entry.items():
            if isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            else:
                lines.append(f"{key} = {value}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Fake backend and factory (T1 to T3)
# --------------------------------------------------------------------------


class FakeBackend:
    """Follows the backend protocol: ``revision``, ``count_tokens``, ``embed_raw``."""

    def __init__(
        self,
        spec,
        *,
        revision=None,
        dim=None,
        nan=False,
        inf=False,
        short=False,
        unavailable=False,
    ):
        self.spec = spec
        self.model_id = spec.id
        self.revision = spec.revision if revision is None else revision
        self.dim = spec.dimension if dim is None else dim
        self.nan = nan
        self.inf = inf
        self.short = short
        self.unavailable = unavailable
        self.embed_raw_calls = 0
        self.received = []

    def count_tokens(self, text):
        return len(text.split())

    def _vector(self, text):
        key = f"{self.model_id}\x00{self.revision}\x00{text}".encode()
        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self.dim) * 3.0  # deliberately not unit length

    def embed_raw(self, texts):
        self.embed_raw_calls += 1
        self.received.append(list(texts))
        if self.unavailable:
            emb = importlib.import_module("build_production_rag.embeddings")
            raise emb.EmbeddingUnavailable("fake backend unavailable")
        n = len(texts) - 1 if self.short and texts else len(texts)
        out = np.zeros((n, self.dim), dtype=np.float32)
        for i in range(n):
            out[i] = self._vector(texts[i])
        if self.nan and n:
            out[0, 0] = np.nan
        if self.inf and n:
            out[0, 0] = np.inf
        return out


class FakeFactory:
    """A ``backend_factory``: callable(spec) -> FakeBackend. Counts its calls."""

    def __init__(self, *, revisions=None, revision=None, **backend_kwargs):
        self.revisions = revisions  # list, one per call (B2); last one repeats
        self.revision = revision  # fixed backend revision differing from spec (B1)
        self.backend_kwargs = backend_kwargs
        self.calls = 0
        self.specs = []
        self.backends = []

    def __call__(self, spec):
        self.calls += 1
        self.specs.append(spec)
        if self.revisions:
            rev = self.revisions[min(self.calls - 1, len(self.revisions) - 1)]
        else:
            rev = self.revision
        backend = FakeBackend(spec, revision=rev, **self.backend_kwargs)
        self.backends.append(backend)
        return backend

    @property
    def embed_raw_calls(self):
        return sum(b.embed_raw_calls for b in self.backends)


# --------------------------------------------------------------------------
# In-memory store (store protocol)
# --------------------------------------------------------------------------


class InMemoryStore:
    def __init__(self, record_text=None):
        self._vectors = []
        self._payloads = []
        self._record = record_text
        self.ops = []  # operation order, for B4
        self.search_calls = 0

    def add(self, vectors, payloads):
        vectors = np.asarray(vectors)
        assert len(vectors) == len(payloads)
        self.ops.append("add")
        self._vectors.extend(vectors)
        self._payloads.extend(payloads)

    def vectors(self):
        if not self._vectors:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack(self._vectors)

    def write_record(self, text):
        self.ops.append("write_record")
        self._record = text

    def read_record(self):
        return self._record

    def search(self, vector, k):
        self.search_calls += 1
        self.ops.append("search")
        return [(p, 1.0) for p in self._payloads[:k]]

    @property
    def payloads(self):
        return list(self._payloads)


@pytest.fixture
def make_store():
    return InMemoryStore


@pytest.fixture
def record_for(emb, idx, fake_specs):
    """record_for(model_id, revision=None) -> record JSON text for a fake model."""

    def _make(model_id="fake-a", revision=None, dimension=None):
        spec = fake_specs[model_id]
        identity = emb.Identity(
            model=model_id,
            revision=spec.revision if revision is None else revision,
            dimension=spec.dimension if dimension is None else dimension,
        )
        return idx.record_to_json(idx.IndexRecord.from_identity(identity))

    return _make


@pytest.fixture
def identity_for(emb, fake_specs):
    def _make(model_id, revision=None):
        spec = fake_specs[model_id]
        return emb.Identity(
            model=model_id,
            revision=spec.revision if revision is None else revision,
            dimension=spec.dimension,
        )

    return _make


# --------------------------------------------------------------------------
# Embedder parametrisation (T4)
# --------------------------------------------------------------------------


class Embedder:
    def __init__(self, kind, emb, fake_registry):
        self.kind = kind
        self.emb = emb
        self.model_id = EMBEDDER_MODEL_IDS[kind]
        if kind == "fake":
            self.registry = fake_registry
            self.factory = FakeFactory()
        else:
            self.registry = None
            self.factory = None

    def embed(self, texts):
        if self.kind == "fake":
            return self.emb.embed(
                self.model_id,
                texts,
                registry=self.registry,
                backend_factory=self.factory,
            )
        return self.emb.embed(self.model_id, texts)

    @property
    def spec(self):
        registry = self.registry if self.registry is not None else self.emb.load_registry()
        return registry[self.model_id]


@pytest.fixture
def embedder(request, emb, fake_registry, monkeypatch):
    kind = request.param
    if kind != "fake":
        monkeypatch.chdir(REPO_ROOT)
    return Embedder(kind, emb, fake_registry)


# --------------------------------------------------------------------------
# OpenAI mock transport
# --------------------------------------------------------------------------


class MockOpenAI:
    """Builds an ``httpx2.Client`` over a mock transport and records requests."""

    def __init__(self, behaviour="ok", dim=1536):
        self.behaviour = behaviour  # "ok", int status, or an exception instance
        self.dim = dim
        self.requests = []
        self.bodies = []
        self.client = httpx2.Client(transport=httpx2.MockTransport(self._handle))

    def _handle(self, request):
        self.requests.append(request)
        try:
            body = json.loads(request.content or b"{}")
        except ValueError:
            body = {}
        self.bodies.append(body)
        if isinstance(self.behaviour, Exception):
            raise self.behaviour
        if isinstance(self.behaviour, int) and self.behaviour != 200:
            return httpx2.Response(
                self.behaviour,
                json={"error": {"message": "mock failure", "type": "mock", "code": None}},
            )
        inputs = body.get("input", [])
        if isinstance(inputs, str):
            inputs = [inputs]
        data = []
        for i, _ in enumerate(inputs):
            vec = np.random.default_rng(i).standard_normal(self.dim).astype(np.float32)
            vec /= np.linalg.norm(vec)
            if body.get("encoding_format") == "base64":
                payload = base64.b64encode(vec.tobytes()).decode()
            else:
                payload = vec.tolist()
            data.append({"object": "embedding", "index": i, "embedding": payload})
        return httpx2.Response(
            200,
            json={
                "object": "list",
                "data": data,
                "model": OPENAI_ID,
                "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)},
            },
        )


@pytest.fixture
def mock_openai():
    return MockOpenAI


@pytest.fixture
def cosine():
    def _cos(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    return _cos
