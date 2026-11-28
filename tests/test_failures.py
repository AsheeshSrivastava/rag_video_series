"""Spec 0001 section 2, "Failures": F1 to F5."""

import logging

import pytest

import httpx2

from conftest import FAKE_MAX_TOKENS, OPENAI_ID, FakeFactory, write_models_toml

SECRET = "sk-test-SECRET-123"


def _openai_spec(emb):
    return emb.load_registry()[OPENAI_ID]


def _exception_chain(exc):
    seen = []
    while exc is not None and exc not in seen:
        seen.append(exc)
        exc = exc.__cause__ or exc.__context__
    return seen


CONFIG_CASES = [
    "unknown_model",
    "bad_revision",
    "missing_key",
    "empty_text",
    "text_too_long",
    "wrong_dimension",
]
UNAVAILABLE_CASES = [
    "backend_unavailable",
    "http_500",
    "http_429",
    "connect_error",
    "timeout",
]


def _run_case(case, emb, fake_registry, mock_openai, tmp_path, monkeypatch):
    """Trigger one failure from the F1 lists."""
    texts = ["alpha beta", "gamma"]
    if case == "unknown_model":
        emb.embed("fake-z", texts, registry=fake_registry, backend_factory=FakeFactory())
    elif case == "bad_revision":
        path = write_models_toml(
            tmp_path / "models.toml",
            {"BAAI/bge-m3": {"revision": "main"}},
        )
        emb.Registry.from_toml(path)
    elif case == "missing_key":
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        emb.OpenAIBackend(_openai_spec(emb), http_client=mock_openai().client)
    elif case == "empty_text":
        emb.embed("fake-a", ["ok", " "], registry=fake_registry, backend_factory=FakeFactory())
    elif case == "text_too_long":
        long_text = " ".join(["w"] * (FAKE_MAX_TOKENS + 1))
        emb.embed("fake-a", [long_text], registry=fake_registry, backend_factory=FakeFactory())
    elif case == "wrong_dimension":
        emb.embed(
            "fake-a", texts, registry=fake_registry, backend_factory=FakeFactory(dim=3)
        )
    elif case == "backend_unavailable":
        emb.embed(
            "fake-a",
            texts,
            registry=fake_registry,
            backend_factory=FakeFactory(unavailable=True),
        )
    else:
        behaviour = {
            "http_500": 500,
            "http_429": 429,
            "connect_error": httpx2.ConnectError("mock connect error"),
            "timeout": httpx2.ReadTimeout("mock timeout"),
        }[case]
        mock = mock_openai(behaviour)
        backend = emb.OpenAIBackend(_openai_spec(emb), api_key="sk-test", http_client=mock.client)
        backend.embed_raw(texts)


@pytest.mark.parametrize(
    "case", [*CONFIG_CASES, *UNAVAILABLE_CASES, "error_types_are_distinct"]
)
def test_F1_each_failure_raises_its_own_error_type(
    case, emb, repo_root, fake_registry, mock_openai, tmp_path, monkeypatch, no_sleep
):
    if case == "error_types_are_distinct":
        assert issubclass(emb.ConfigurationError, emb.EmbeddingError)
        assert issubclass(emb.EmbeddingUnavailable, emb.EmbeddingError)
        assert not issubclass(emb.ConfigurationError, emb.EmbeddingUnavailable)
        assert not issubclass(emb.EmbeddingUnavailable, emb.ConfigurationError)
        return
    expected = (
        emb.ConfigurationError if case in CONFIG_CASES else emb.EmbeddingUnavailable
    )
    with pytest.raises(expected):
        _run_case(case, emb, fake_registry, mock_openai, tmp_path, monkeypatch)


@pytest.mark.parametrize(
    "env_value", [None, "", "via_embed"], ids=["unset", "empty", "via-embed"]
)
def test_F2_missing_api_key_fails_before_any_request(
    emb, repo_root, mock_openai, monkeypatch, env_value
):
    if env_value is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    elif env_value == "":
        monkeypatch.setenv("OPENAI_API_KEY", "")
    else:  # default factory through embed, no key set
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(emb.ConfigurationError):
            emb.embed(OPENAI_ID, ["hello"])
        return
    mock = mock_openai()
    with pytest.raises(emb.ConfigurationError):
        emb.OpenAIBackend(_openai_spec(emb), http_client=mock.client)
    assert len(mock.requests) == 0


def test_F3_a_short_result_raises_instead_of_returning_partial_vectors(
    emb, fake_registry
):
    factory = FakeFactory(short=True)
    with pytest.raises(emb.EmbeddingError):
        emb.embed(
            "fake-a",
            ["one", "two", "three"],
            registry=fake_registry,
            backend_factory=factory,
        )


def test_F4_transient_failure_is_retried_at_most_twice(
    emb, repo_root, mock_openai, no_sleep
):
    mock = mock_openai(500)
    backend = emb.OpenAIBackend(_openai_spec(emb), api_key="sk-test", http_client=mock.client)
    with pytest.raises(emb.EmbeddingUnavailable):
        backend.embed_raw(["hello world"])
    assert len(mock.requests) == 3  # 1 try + 2 retries


@pytest.mark.parametrize("status", [401, 500], ids=["http_401", "http_500"])
def test_F5_api_key_never_appears_in_errors_or_logs(
    emb, repo_root, mock_openai, no_sleep, caplog, status
):
    caplog.set_level(logging.DEBUG)
    mock = mock_openai(status)
    backend = emb.OpenAIBackend(_openai_spec(emb), api_key=SECRET, http_client=mock.client)
    with pytest.raises(emb.EmbeddingError) as info:
        backend.embed_raw(["hello world"])
    for exc in _exception_chain(info.value):
        assert SECRET not in str(exc)
        assert SECRET not in repr(exc)
        assert all(SECRET not in str(arg) for arg in exc.args)
    assert SECRET not in caplog.text
    for record in caplog.records:
        assert SECRET not in record.getMessage()
        assert SECRET not in str(record.args)
