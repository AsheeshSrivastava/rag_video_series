"""Spec 0001 section 2: the embedding function (E1 to E12)."""

import numpy as np
import pytest

from conftest import (
    E9_PARAMS,
    E10_PARAMS,
    E11_PARAMS,
    EMBEDDER_PARAMS,
    FAKE_DIM,
    FAKE_MAX_TOKENS,
    OPENAI_ID,
    REAL_IDS,
    FakeBackend,
    FakeFactory,
)

TEXTS = ["alpha beta", "gamma", "delta epsilon zeta"]


def _text_with_token_count(count_fn, target, word="a"):
    """Smallest repeated-word text whose count_fn equals ``target`` exactly."""
    # "a" is one token in both local tokenizers; "hello" is two in bge-m3,
    # which makes odd counts such as 8193 unreachable.
    lo, hi = 1, target + 8
    while lo < hi:
        mid = (lo + hi) // 2
        if count_fn(" ".join([word] * mid)) >= target:
            hi = mid
        else:
            lo = mid + 1
    text = " ".join([word] * lo)
    assert count_fn(text) == target, "could not build a text of the exact token count"
    return text


def test_E1_returns_one_vector_per_text_in_order(emb, fake_registry):
    texts = ["one", "two words", "three more words", "four", "five five"]
    factory = FakeFactory()
    result = emb.embed("fake-a", texts, registry=fake_registry, backend_factory=factory)
    assert result.vectors.shape == (len(texts), FAKE_DIM)
    assert result.vectors.dtype == np.float32
    for i, text in enumerate(texts):
        single = emb.embed(
            "fake-a", [text], registry=fake_registry, backend_factory=FakeFactory()
        )
        np.testing.assert_allclose(result.vectors[i], single.vectors[0], atol=1e-6)
    # distinct texts give distinct rows, so an order mix-up would be seen
    assert not np.allclose(result.vectors[0], result.vectors[1])


def test_E2_empty_list_returns_empty_without_loading(emb, fake_registry, fake_specs):
    factory = FakeFactory()
    result = emb.embed("fake-a", [], registry=fake_registry, backend_factory=factory)
    assert result.vectors.shape == (0, FAKE_DIM)
    assert result.identity == emb.Identity(
        model="fake-a", revision=fake_specs["fake-a"].revision, dimension=FAKE_DIM
    )
    assert factory.calls == 0


@pytest.mark.parametrize("registry_kind", ["real", "fake"])
def test_E3_unknown_model_lists_the_known_ids_and_never_loads(
    emb, repo_root, fake_registry, registry_kind
):
    factory = FakeFactory()
    if registry_kind == "real":
        with pytest.raises(emb.ConfigurationError) as info:
            emb.embed("not/a-model", ["hello"], backend_factory=factory)
        for model_id in REAL_IDS:
            assert model_id in str(info.value)
    else:
        with pytest.raises(emb.ConfigurationError):
            emb.embed(
                "fake-z", ["hello"], registry=fake_registry, backend_factory=factory
            )
    assert factory.calls == 0


@pytest.mark.parametrize(
    "texts",
    [
        pytest.param(["fine text", ""], id="empty"),
        pytest.param(["fine text", "   "], id="spaces"),
        pytest.param(["fine text", " \t\n "], id="mixed-whitespace"),
    ],
)
def test_E4_empty_or_blank_text_names_its_position(emb, fake_registry, texts):
    factory = FakeFactory()
    with pytest.raises(emb.ConfigurationError) as info:
        emb.embed("fake-a", texts, registry=fake_registry, backend_factory=factory)
    assert info.value.position == 1
    assert factory.embed_raw_calls == 0


@pytest.mark.parametrize("embedder", EMBEDDER_PARAMS, indirect=True)
def test_E5_text_over_max_tokens_fails_and_is_never_truncated(embedder, emb):
    if embedder.kind == "fake":
        too_long = " ".join(["w"] * (FAKE_MAX_TOKENS + 1))
        exact = " ".join(["w"] * FAKE_MAX_TOKENS)
        with pytest.raises(emb.ConfigurationError) as info:
            embedder.embed(["short", too_long])
        assert info.value.position == 1
        assert info.value.token_count == FAKE_MAX_TOKENS + 1
        assert embedder.factory.embed_raw_calls == 0
        ok = embedder.embed(["short", exact])
        assert ok.vectors.shape == (2, FAKE_DIM)
        return

    spec = embedder.spec
    maximum = spec.max_input_tokens
    if embedder.kind in ("jina", "bge-m3"):
        backend = emb.LocalBackend(spec)
        count = backend.count_tokens
        too_long = _text_with_token_count(count, maximum + 1)
        exact = _text_with_token_count(count, maximum)
        with pytest.raises(emb.ConfigurationError) as info:
            embedder.embed(["short", too_long])
        assert info.value.position == 1
        assert info.value.token_count == maximum + 1
        assert embedder.embed([exact]).vectors.shape == (1, spec.dimension)
        return

    # openai: settles fact 2. Counted with cl100k_base.
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
    token = encoding.decode(encoding.encode(" hello"))
    exact = token * maximum
    over = token * (maximum + 1)
    assert len(encoding.encode(exact)) == maximum
    assert len(encoding.encode(over)) == maximum + 1
    assert embedder.embed([exact]).vectors.shape == (1, spec.dimension)
    # straight to the backend: embed() would reject the text before sending it
    with pytest.raises(emb.EmbeddingError):
        emb.OpenAIBackend(spec).embed_raw([over])


@pytest.mark.parametrize("embedder", EMBEDDER_PARAMS, indirect=True)
def test_E6_vectors_have_the_registry_dimension(embedder, emb, fake_registry):
    result = embedder.embed(TEXTS)
    assert result.vectors.shape == (len(TEXTS), embedder.spec.dimension)
    assert result.identity.dimension == embedder.spec.dimension
    if embedder.kind == "fake":  # a backend returning another length fails the call
        factory = FakeFactory(dim=FAKE_DIM + 3)
        with pytest.raises(emb.ConfigurationError):
            emb.embed("fake-a", TEXTS, registry=fake_registry, backend_factory=factory)


@pytest.mark.parametrize("embedder", EMBEDDER_PARAMS, indirect=True)
def test_E7_every_vector_has_unit_length(embedder):
    result = embedder.embed(TEXTS)
    norms = np.linalg.norm(result.vectors.astype(np.float64), axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-3)
    if embedder.kind == "fake":  # the fake's raw vectors are not unit length
        raw = FakeBackend(embedder.spec).embed_raw(TEXTS)
        assert not np.allclose(np.linalg.norm(raw, axis=1), 1.0, atol=1e-3)


@pytest.mark.parametrize(
    "bad", [pytest.param("nan", id="nan"), pytest.param("inf", id="inf")]
)
def test_E8_nan_or_infinity_fails_the_call(emb, fake_registry, bad):
    factory = FakeFactory(**{bad: True})
    with pytest.raises(emb.EmbeddingError):
        emb.embed("fake-a", TEXTS, registry=fake_registry, backend_factory=factory)


@pytest.mark.parametrize("embedder", E9_PARAMS, indirect=True)
def test_E9_identity_matches_the_model_actually_used(embedder, emb, fake_registry, fake_specs):
    result = embedder.embed(TEXTS)
    spec = embedder.spec
    assert result.identity.model == embedder.model_id
    assert result.identity.dimension == spec.dimension
    if embedder.kind == "fake":
        assert result.identity.revision == embedder.factory.backends[-1].revision
        # the revision comes from the backend, not from the registry
        other = emb.embed(
            "fake-a", TEXTS, registry=fake_registry, backend_factory=FakeFactory(revision="c" * 40)
        )
        assert other.identity.revision == "c" * 40
        assert other.identity.revision != fake_specs["fake-a"].revision
    else:
        assert result.identity.revision == spec.revision


@pytest.mark.parametrize("embedder", E10_PARAMS, indirect=True)
def test_E10_results_do_not_depend_on_batching(embedder, cosine):
    together = embedder.embed(TEXTS).vectors
    for i, text in enumerate(TEXTS):
        alone = embedder.embed([text]).vectors[0]
        assert cosine(together[i], alone) >= 0.9999


@pytest.mark.parametrize("embedder", E11_PARAMS, indirect=True)
def test_E11_same_text_gives_same_vector_across_calls(embedder, cosine):
    first = embedder.embed(TEXTS).vectors
    second = embedder.embed(TEXTS).vectors
    for i in range(len(TEXTS)):
        assert cosine(first[i], second[i]) >= 0.9999


@pytest.mark.parametrize(
    "sizes, counted, requests",
    [
        ((150_000, 150_000), 0, 1),  # exactly 300,000 bytes: no counting needed
        ((150_000, 150_001), 2, 2),  # one byte over: every text is counted, and split
    ],
    ids=["at-300000-bytes", "above-300000-bytes"],
)
def test_E12_openai_counts_tokens_above_300000_bytes(
    emb, real_registry, mock_openai, sizes, counted, requests
):
    calls = []

    def counter(text):
        calls.append(text)
        return len(text.encode())  # one token per byte: the upper bound

    mock = mock_openai()
    backend = emb.OpenAIBackend(
        real_registry[OPENAI_ID],
        api_key="sk-test",
        http_client=mock.client,
        token_counter=counter,
    )
    texts = ["a" * size for size in sizes]
    vectors = backend.embed_raw(texts)
    assert len(calls) == counted
    assert len(mock.requests) == requests
    # every request stays within 300,000 tokens, and all rows come back
    for body in mock.bodies:
        assert sum(counter(t) for t in body["input"]) <= 300_000
    assert vectors.shape == (len(texts), 1536)
