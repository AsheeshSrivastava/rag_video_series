"""Spec 0001 section 3, "Content": the index record (X1 to X3)."""

import json

import pytest

from conftest import BGE_ID, OPENAI_ID, REAL_IDS

BGE_REV = "5617a9f61b028005a4858fdac845db406aefb181"


def _good():
    return {"record_version": 1, "model": BGE_ID, "revision": BGE_REV, "dimension": 1024}


def _with(**changes):
    return {**_good(), **changes}


def _without(key):
    data = _good()
    del data[key]
    return data


X1_CASES = [
    pytest.param("{not json", id="invalid-json"),
    pytest.param("", id="empty-text"),
    pytest.param(json.dumps([1, 2]), id="not-an-object"),
    pytest.param(json.dumps(_without("revision")), id="missing-revision"),
    pytest.param(json.dumps(_without("model")), id="missing-model"),
    pytest.param(json.dumps({**_good(), "extra": 1}), id="extra-key"),
    pytest.param(json.dumps(_with(record_version=2)), id="record-version-2"),
    pytest.param(json.dumps(_with(record_version="1")), id="record-version-string"),
    pytest.param(json.dumps(_with(dimension="1024")), id="dimension-string"),
    pytest.param(json.dumps(_with(dimension=True)), id="dimension-true"),
    pytest.param(json.dumps(_with(revision=5617)), id="revision-number"),
    pytest.param(json.dumps(_with(model=7)), id="model-number"),
    pytest.param(json.dumps(_good()), id="well-formed-is-readable"),
]


@pytest.mark.parametrize("text", X1_CASES)
def test_X1_only_a_well_formed_record_is_readable(emb, idx, real_registry, text):
    if text == json.dumps(_good()):
        record = idx.record_from_json(text, real_registry)
        assert (
            record.record_version,
            record.model,
            record.revision,
            record.dimension,
        ) == (1, BGE_ID, BGE_REV, 1024)
        return
    with pytest.raises(idx.IndexRecordError):
        idx.record_from_json(text, real_registry)


X2_CASES = [
    pytest.param(_with(model="not/in-registry"), id="model-not-in-registry"),
    pytest.param(_with(revision="main"), id="revision-main"),
    pytest.param(_with(revision=BGE_REV.upper()), id="revision-uppercase"),
    pytest.param(_with(revision=None), id="local-model-null-revision"),
    pytest.param(_with(dimension=768), id="dimension-differs-from-registry"),
    pytest.param(
        {**_good(), "model": OPENAI_ID, "revision": "2024-01-25", "dimension": 1536},
        id="openai-non-null-revision",
    ),
    pytest.param(
        {**_good(), "model": OPENAI_ID, "revision": None, "dimension": 1024},
        id="openai-wrong-dimension",
    ),
]


@pytest.mark.parametrize("data", X2_CASES)
def test_X2_record_must_be_consistent_with_the_registry(emb, idx, real_registry, data):
    with pytest.raises(idx.IndexRecordError):
        idx.record_from_json(json.dumps(data), real_registry)


@pytest.mark.parametrize("model_id", [*REAL_IDS, "fake-a", "fake-b"])
def test_X3_record_round_trips(emb, idx, repo_root, fake_registry, model_id):
    registry = fake_registry if model_id.startswith("fake") else emb.load_registry()
    spec = registry[model_id]
    identity = emb.Identity(
        model=spec.id, revision=spec.revision, dimension=spec.dimension
    )
    record = idx.IndexRecord.from_identity(identity)
    assert record.record_version == 1

    text = idx.record_to_json(record)
    assert isinstance(text, str)
    assert set(json.loads(text)) == {"record_version", "model", "revision", "dimension"}

    back = idx.record_from_json(text, registry)
    assert back.record_version == 1
    assert back.model == spec.id
    assert back.revision == spec.revision
    assert back.dimension == spec.dimension
