"""Spec 0001 section 4: the query check (Q1 to Q6)."""

import inspect

import numpy as np
import pytest

from conftest import FAKE_DIM, FakeFactory, InMemoryStore


def _store_with(record_text):
    store = InMemoryStore(record_text=record_text)
    store.add(np.ones((1, FAKE_DIM), dtype=np.float32), ["a passage"])
    store.ops.clear()
    return store


def _stub_embed(emb, identity):
    """An ``embed_fn`` that reports a chosen identity, whatever model was asked for."""
    calls = []

    def embed_fn(model_id, texts, *, registry=None, backend_factory=None):
        calls.append(model_id)
        vectors = np.ones((len(texts), identity.dimension), dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return emb.Embeddings(vectors=vectors, identity=identity)

    embed_fn.calls = calls
    return embed_fn


@pytest.mark.parametrize(
    "record_text",
    [None, "{not json", '{"record_version": 1}'],
    ids=["missing", "invalid-json", "missing-keys"],
)
def test_Q1_missing_or_unreadable_record_stops_before_embedding(
    emb, idx, fake_registry, record_text
):
    store = InMemoryStore(record_text=record_text)
    factory = FakeFactory()
    with pytest.raises(idx.IndexRecordError):
        idx.search(
            "a question", store, 3, registry=fake_registry, backend_factory=factory
        )
    assert factory.calls == 0


def test_Q2_question_is_embedded_with_the_records_model(
    emb, idx, fake_registry, record_for
):
    params = inspect.signature(idx.search).parameters
    assert "model" not in params
    assert "model_id" not in params
    for model_id in ("fake-a", "fake-b"):
        store = _store_with(record_for(model_id))
        factory = FakeFactory()
        idx.search(
            "a question", store, 3, registry=fake_registry, backend_factory=factory
        )
        assert [spec.id for spec in factory.specs] == [model_id]


@pytest.mark.parametrize("case", ["different-model", "different-revision-only"])
def test_Q3_changed_identity_raises_mismatch_naming_both(
    emb, idx, fake_registry, record_for, identity_for, case
):
    store = _store_with(record_for("fake-a"))
    record_identity = identity_for("fake-a")
    if case == "different-revision-only":
        question_identity = identity_for("fake-a", revision="c" * 40)
        factory = FakeFactory(revision="c" * 40)  # backend loads another revision
        with pytest.raises(idx.ModelMismatchError) as info:
            idx.search(
                "a question", store, 3, registry=fake_registry, backend_factory=factory
            )
    else:
        question_identity = identity_for("fake-b")
        with pytest.raises(idx.ModelMismatchError) as info:
            idx.search(
                "a question",
                store,
                3,
                registry=fake_registry,
                embed_fn=_stub_embed(emb, question_identity),
            )
    error = info.value
    # expected is the record's identity; actual is what embedded the question
    assert error.expected == record_identity
    assert error.actual == question_identity
    assert record_identity.model in str(error)
    assert question_identity.model in str(error)


@pytest.mark.parametrize("case", ["mismatch", "missing-record", "unreadable-record"])
def test_Q4_failures_search_nothing_and_never_return(
    emb, idx, fake_registry, record_for, identity_for, case
):
    if case == "mismatch":
        store = _store_with(record_for("fake-a"))
        embed_fn = _stub_embed(emb, identity_for("fake-b"))
        expected = idx.ModelMismatchError
    else:
        store = InMemoryStore(record_text=None if case == "missing-record" else "garbage")
        embed_fn = _stub_embed(emb, identity_for("fake-a"))
        expected = idx.IndexRecordError
    with pytest.raises(expected):
        idx.search("a question", store, 3, registry=fake_registry, embed_fn=embed_fn)
    assert store.search_calls == 0
    assert "search" not in store.ops


@pytest.mark.skip(reason="deferred to the eval spec (spec 0001, section 5)")
def test_Q5_eval_run_fails_on_mismatch_or_record_error():
    ...


def test_Q6_same_dimension_different_model_is_still_a_mismatch(
    emb, idx, fake_registry, fake_specs, record_for
):
    assert fake_specs["fake-a"].dimension == fake_specs["fake-b"].dimension
    store = _store_with(record_for("fake-a"))
    # model B, same dimension and same revision as the record: only the model differs
    actual = emb.Identity(
        model="fake-b", revision=fake_specs["fake-a"].revision, dimension=FAKE_DIM
    )
    with pytest.raises(idx.ModelMismatchError):
        idx.search(
            "a question",
            store,
            3,
            registry=fake_registry,
            embed_fn=_stub_embed(emb, actual),
        )
    assert store.search_calls == 0
