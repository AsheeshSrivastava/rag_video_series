"""Spec 0001 section 2, "The fake embedder": T1 to T4."""

import numpy as np
import pytest

from conftest import EMBEDDER_PARAMS, FAKE_DIM, FakeBackend, FakeFactory


def test_T1_fake_follows_the_backend_protocol_and_the_contract(
    emb, fake_registry, fake_specs
):
    backend = FakeBackend(fake_specs["fake-a"])
    assert isinstance(backend.revision, str)
    assert callable(backend.count_tokens)
    assert callable(backend.embed_raw)
    assert backend.count_tokens("one two three") == 3

    result = emb.embed(
        "fake-a",
        ["one two", "three"],
        registry=fake_registry,
        backend_factory=FakeFactory(),
    )
    assert result.vectors.shape == (2, FAKE_DIM)
    assert result.identity == emb.Identity(
        model="fake-a", revision=fake_specs["fake-a"].revision, dimension=FAKE_DIM
    )


def test_T2_two_fakes_can_differ_only_in_model_id(emb, fake_registry):
    same_revision = "c" * 40
    a = emb.embed(
        "fake-a", ["hi"], registry=fake_registry, backend_factory=FakeFactory(revision=same_revision)
    ).identity
    b = emb.embed(
        "fake-b", ["hi"], registry=fake_registry, backend_factory=FakeFactory(revision=same_revision)
    ).identity
    assert a.model != b.model
    assert a.revision == b.revision
    assert a.dimension == b.dimension


def test_T3_fake_is_deterministic_and_depends_on_model_id(emb, fake_registry):
    texts = ["same text", "other text"]

    def run(model_id):
        return emb.embed(
            model_id, texts, registry=fake_registry, backend_factory=FakeFactory(revision="c" * 40)
        ).vectors

    first, second = run("fake-a"), run("fake-a")
    np.testing.assert_array_equal(first, second)
    other_model = run("fake-b")
    assert not np.allclose(first[0], other_model[0])
    assert not np.allclose(first[1], other_model[1])


def test_T4_contract_tests_run_against_fake_and_real_models_by_marker():
    params = list(EMBEDDER_PARAMS)
    assert params[0] == "fake"  # unmarked, so it runs in the default run

    marked = {}
    for param in params[1:]:
        assert isinstance(param, type(pytest.param("x")))
        marked[param.values[0]] = {m.name for m in param.marks}
    assert marked == {
        "jina": {"model"},
        "bge-m3": {"model"},
        "openai": {"network"},
    }
