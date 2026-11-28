"""Spec 0001 section 1: the model registry (M1 to M5)."""

import re
import threading
import urllib.parse

import pytest

from conftest import BGE_ID, JINA_ID, OPENAI_ID, write_models_toml

HEX40 = re.compile(r"^[0-9a-f]{40}$")


def test_M1_registry_has_exactly_the_three_model_ids(emb, repo_root):
    registry = emb.load_registry()
    assert list(registry.ids()) == [OPENAI_ID, JINA_ID, BGE_ID]  # models.toml file order


M2_BAD_CASES = [
    pytest.param(JINA_ID, "revision", "main", id="jina-revision-main"),
    pytest.param(BGE_ID, "revision", "main", id="bge-revision-main"),
    pytest.param(JINA_ID, "revision", "v1.0", id="jina-revision-tag"),
    pytest.param(
        JINA_ID,
        "revision",
        "322D4D7E2F35E84137961A65AF894FDA0385EB7A",
        id="jina-revision-uppercase-hex",
    ),
    pytest.param(BGE_ID, "revision", "5617a9f", id="bge-revision-short-hex"),
    pytest.param(JINA_ID, "remote_code_revision", "main", id="jina-remote-code-main"),
    pytest.param(
        JINA_ID,
        "remote_code_revision",
        "F3EC4CF7DE7E561007F27C9EFC7148B0BD713F81",
        id="jina-remote-code-uppercase",
    ),
    pytest.param(None, None, None, id="real-models-toml-loads"),
]


@pytest.mark.parametrize(("model_id", "key", "value"), M2_BAD_CASES)
def test_M2_local_revisions_must_be_40_char_lowercase_hex(
    emb, repo_root, tmp_path, model_id, key, value
):
    if model_id is None:  # the real file is accepted and its pins are hex
        registry = emb.Registry.from_toml(repo_root / "models.toml")
        for local_id in (JINA_ID, BGE_ID):
            assert HEX40.match(registry[local_id].revision)
        assert HEX40.match(registry[JINA_ID].remote_code_revision)
        return
    path = write_models_toml(tmp_path / "models.toml", {model_id: {key: value}})
    with pytest.raises(emb.ConfigurationError):
        emb.Registry.from_toml(path)


@pytest.mark.parametrize(
    "revision", [None, "2024-01-25", "v1", "main"], ids=["real-file", "date", "tag", "main"]
)
def test_M3_openai_revision_is_null_and_never_filled_in(
    emb, repo_root, tmp_path, revision
):
    if revision is None:
        assert emb.load_registry()[OPENAI_ID].revision is None
        return
    path = write_models_toml(
        tmp_path / "models.toml", {OPENAI_ID: {"revision": revision}}
    )
    with pytest.raises(emb.ConfigurationError):
        emb.Registry.from_toml(path)


def test_M4_openai_request_asks_for_full_dimension(
    emb, repo_root, mock_openai
):
    mock = mock_openai()
    result = emb.embed(
        OPENAI_ID,
        ["hello world", "another text"],
        backend_factory=lambda spec: emb.OpenAIBackend(
            spec,
            api_key="sk-test",
            http_client=mock.client,
            # a fake counter, so this unit test never loads tiktoken's encoding
            token_counter=lambda text: len(text.split()),
        ),
    )
    assert mock.bodies, "no request reached the mock transport"
    for body in mock.bodies:
        # either no `dimensions` key, or exactly the registry dimension
        assert body.get("dimensions", 1536) == 1536
    assert result.vectors.shape == (2, 1536)


@pytest.mark.model
@pytest.mark.parametrize("model_id", [JINA_ID, BGE_ID], ids=["jina", "bge-m3"])
def test_M5_local_load_fetches_nothing_from_an_unpinned_revision(
    emb, real_registry, model_id, monkeypatch
):
    """Every Hub request made while loading names a commit pinned in models.toml.

    Requests are seen through huggingface_hub's own HTTP session (requests-based
    in huggingface_hub 0.x), so a fetch made by transformers, by remote code, or
    by a background thread that loading starts is caught the same way. Threads
    started during the load are joined before checking.
    """
    import huggingface_hub
    import requests

    from build_production_rag.local_backend import LocalBackend

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)  # offline would hide the fetches
    spec = real_registry[model_id]
    pins = {spec.id: spec.revision}
    if spec.remote_code_repo is not None:
        pins[spec.remote_code_repo] = spec.remote_code_revision

    urls = []
    lock = threading.Lock()

    def session_factory():
        session = requests.Session()

        def record(response, *_args, **_kwargs):
            with lock:
                urls.append(urllib.parse.unquote(response.request.url))

        session.hooks["response"].append(record)
        return session

    threads_before = set(threading.enumerate())
    huggingface_hub.configure_http_backend(backend_factory=session_factory)
    try:
        LocalBackend(spec)
        for thread in set(threading.enumerate()) - threads_before:
            thread.join(timeout=120)
    finally:
        huggingface_hub.configure_http_backend()

    def pinned(url):
        path = urllib.parse.urlsplit(url).path
        return any(f"/{repo}/" in path + "/" and rev in path for repo, rev in pins.items())

    unpinned = [url for url in urls if not pinned(url)]
    assert unpinned == [], "fetched from an unpinned revision:\n" + "\n".join(unpinned)
