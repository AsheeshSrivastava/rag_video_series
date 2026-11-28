"""Spec 0001 section 3, "Building an index": B1 to B5."""

import json

import numpy as np
import pytest

from conftest import FAKE_DIM, REV_1, REV_2, FakeFactory, InMemoryStore

TEXTS = ["one", "two words", "three more words", "four"]


class NeedsApproval(Exception):
    """Stands in for a file that needs approval (R8)."""


def test_B1_record_holds_the_identity_the_backend_reported(
    emb, idx, fake_registry, fake_specs
):
    backend_revision = "c" * 40
    assert backend_revision != fake_specs["fake-a"].revision
    store = InMemoryStore()
    record = idx.build_index(
        "fake-a",
        TEXTS,
        store,
        registry=fake_registry,
        backend_factory=FakeFactory(revision=backend_revision),
    )
    assert record.revision == backend_revision
    stored = json.loads(store.read_record())
    assert stored["revision"] == backend_revision
    assert stored["model"] == "fake-a"
    assert stored["dimension"] == FAKE_DIM


def test_B2_a_batch_with_a_different_identity_stops_the_build(
    emb, idx, fake_registry
):
    store = InMemoryStore()
    factory = FakeFactory(revisions=[REV_1, REV_2])
    with pytest.raises(idx.ModelMismatchError):
        idx.build_index(
            "fake-a",
            TEXTS,
            store,
            batch_size=2,
            registry=fake_registry,
            backend_factory=factory,
        )
    assert factory.calls == 2
    assert store.read_record() is None


def test_B3_stored_vectors_have_the_records_dimension(emb, idx, fake_registry):
    store = InMemoryStore()
    record = idx.build_index(
        "fake-a",
        TEXTS,
        store,
        batch_size=3,
        registry=fake_registry,
        backend_factory=FakeFactory(),
    )
    vectors = store.vectors()
    assert vectors.shape == (len(TEXTS), record.dimension)
    assert vectors.shape[1] == record.dimension
    assert store.payloads == TEXTS


@pytest.mark.parametrize("case", ["record-written-last", "early-stop-leaves-no-record"])
def test_B4_record_is_written_last_and_an_early_stop_writes_none(
    emb, idx, fake_registry, case
):
    store = InMemoryStore()
    if case == "record-written-last":
        idx.build_index(
            "fake-a",
            TEXTS,
            store,
            batch_size=2,
            registry=fake_registry,
            backend_factory=FakeFactory(),
        )
        assert store.ops.count("write_record") == 1
        assert store.ops.count("add") >= 1
        assert store.ops.index("write_record") > max(
            i for i, op in enumerate(store.ops) if op == "add"
        )
        return

    def texts():
        yield "first text"
        yield "second text"
        yield "third text"
        raise NeedsApproval("file needs approval")

    with pytest.raises(NeedsApproval):
        idx.build_index(
            "fake-a",
            texts(),
            store,
            batch_size=2,
            registry=fake_registry,
            backend_factory=FakeFactory(),
        )
    assert store.read_record() is None
    assert "write_record" not in store.ops


@pytest.mark.parametrize("case", ["existing-record", "other-model-new-index"])
def test_B5_existing_record_is_never_changed(
    emb, idx, fake_registry, record_for, case
):
    if case == "existing-record":
        original = record_for("fake-a")
        store = InMemoryStore(record_text=original)
        store.add(np.ones((1, FAKE_DIM), dtype=np.float32), ["old"])
        before = store.vectors().copy()
        factory = FakeFactory()

        with pytest.raises(idx.IndexRecordError):
            idx.build_index(
                "fake-a", TEXTS, store, registry=fake_registry, backend_factory=factory
            )

        assert store.read_record() == original
        np.testing.assert_array_equal(store.vectors(), before)
        assert store.payloads == ["old"]
        assert factory.calls == 0  # nothing was embedded
        return

    store_a, store_b = InMemoryStore(), InMemoryStore()
    idx.build_index(
        "fake-a", TEXTS, store_a, registry=fake_registry, backend_factory=FakeFactory()
    )
    record_a = store_a.read_record()
    vectors_a = store_a.vectors().copy()

    idx.build_index(
        "fake-b", TEXTS, store_b, registry=fake_registry, backend_factory=FakeFactory()
    )

    assert json.loads(store_b.read_record())["model"] == "fake-b"
    assert store_a.read_record() == record_a
    np.testing.assert_array_equal(store_a.vectors(), vectors_a)
    assert not np.allclose(store_a.vectors(), store_b.vectors())
