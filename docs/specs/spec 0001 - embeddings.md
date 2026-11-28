---
created: 2026-11-28
status: accepted
description: Spec for the one embedding function (R9) and the index record (ADR 0001).
modified: 2026-11-28T10:31:36+05:30
---
# Spec: the embedding function and the index record

- Requirements: R9, R10 in [prd.md](../prd.md)
- Decision: [ADR 0001](../adr/0001-embedding-model.md), one index per embedding model
- Open decisions this spec must not settle: D2 (chunking), D4 (which model wins), D7 (vector store), D11 ("unavailable" vs "not found")

## Scope

This spec covers the first piece only:

1. **The embedding function.** One function that turns texts into vectors with any of the three models from R9.
2. **The model registry.** The fixed list of those three models and what the function knows about each.
3. **The index record.** The small record written beside an index's vectors that says which model built it.
4. **The query check.** How search uses the record before it embeds a question.

Not covered: where vectors or the record are stored (D7), how indexes are found or named in a store (D7), chunking (D2), retrieval and scoring (D3), refusal (D8), and how an embedding failure maps to an answer state (D11).

## How to read the rules

Each rule has an ID and a test column:

- **unit**: runs in the default test run with a fake embedder. No network, no GPU.
- **model**: loads a real local model on the GPU. Marked `model` in pytest, run on purpose with `-m model`.
- **network**: calls OpenAI. Marked `network` in pytest, run on purpose with `-m network`.

Every value in the registry was checked against the model's own files or the OpenAI docs on 2026-11-28. The sources are in "Facts checked" at the end. If a source changes, the registry changes and the rules stay.

## 1. The model registry

The registry holds exactly three entries. The model ID is the exact string used everywhere: in config, in the record and in error messages.

The values come from `models.toml` at the repo root, which is checked in. It holds each model's ID, revision, dimension and max input tokens. For jina it also holds the repo and revision of the remote code the model loads. A model name such as `main` can point to new files at any time, so the pins live in that one file. The table below copies it. If the two differ, `models.toml` is right and this table is out of date.

| Model ID                            | Runs on    | Dimension | Max input tokens | Revision                                   | Remote code                                                                    |
| ----------------------------------- | ---------- | --------- | ---------------- | ------------------------------------------ | ------------------------------------------------------------------------------ |
| `text-embedding-3-small`            | OpenAI API | 1536      | 8192 (fact 2)    | empty                                      | none                                                                           |
| `jinaai/jina-embeddings-v2-base-en` | local, GPU | 768       | 8192             | `322d4d7e2f35e84137961a65af894fda0385eb7a` | `jinaai/jina-bert-implementation` at `f3ec4cf7de7e561007f27c9efc7148b0bd713f81` |
| `BAAI/bge-m3`                       | local, GPU | 1024      | 8192             | `5617a9f61b028005a4858fdac845db406aefb181` | none                                                                           |

The revisions are each repo's `main` commit on 2026-11-28. They are candidates until D5 pins them.

Notes for D5, not rules (facts 4 and 8): jina v2 uses mean pooling, then L2 normalisation, and needs remote code enabled to load. bge-m3 dense vectors use the first token (CLS), then L2 normalisation. Neither model's tokenizer settings give a usable length limit on their own (fact 2), so the limit comes from the registry, not from the tokenizer.

| ID  | Rule                                                                                                                                                                     | Test |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---- |
| M1  | The registry contains exactly the three model IDs above, and no others. M1 also applies when the file is loaded: `Registry.from_toml` rejects a file that doesn't hold exactly these three with a configuration error. | unit |
| M2  | A local model's revision, and the revision of any remote code it loads, is a 40-character lowercase hexadecimal commit hash. A branch or tag name (for example `main`) is rejected with a configuration error. | unit |
| M3  | The OpenAI model's revision is empty (null). It is never filled with a date, a guess or the API version.                                                                 | unit |
| M4  | The function never asks OpenAI for a shortened vector. It always requests the full dimension in the registry.                                                           | unit |

## 2. The embedding function

### Inputs and outputs

The function takes a model ID and an ordered list of texts. It returns the vectors, one per text, and the identity of the model that made them. The identity is three values: model ID, revision and dimension. These are the same three values the index record holds (section 3).

There is one function for both passages and questions. It has no "document" or "query" mode, and it adds no prefix or instruction to the text. Neither local model asks for a query prefix (fact 5).

| ID  | Rule                                                                                                                                                               | Test         |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------ |
| E1  | For N texts the function returns exactly N vectors, in the same order as the texts.                                                                                | unit         |
| E2  | An empty list returns an empty list and the identity, without calling the model or loading it.                                                                     | unit         |
| E3  | An unknown model ID fails with a configuration error that lists the three known IDs. It fails before any network call or model load.                               | unit         |
| E4  | A text that is empty or only whitespace fails the whole call with a configuration error that names the text's position in the list.                                | unit         |
| E5  | A text longer than the model's max input tokens fails the whole call with a configuration error that names its position and its token count. Text is never truncated. | unit, model, network |
| E6  | Every returned vector has exactly the registry dimension for that model. If the model returns a different length, the call fails.                                  | unit, model, network |
| E7  | Every returned vector has length 1 (L2 norm within 1e-3 of 1.0). The function normalises every vector itself, whatever the model already does.                     | unit, model, network |
| E8  | Every value in every vector is a finite number. A NaN or infinity fails the call with a configuration error, because a retry can't fix it. | unit         |
| E9  | The returned identity matches the model actually used: the requested model ID, the revision that was loaded (local) or null (OpenAI), and the registry dimension.  | unit, model  |
| E10 | Results do not depend on batching. Embedding a list in one call, or one text at a time, gives vectors whose cosine similarity is at least 0.9999 for each text.     | model, network |
| E11 | A local model embeds the same text to the same vector across two calls in one process (cosine similarity at least 0.9999).                                         | model        |

### Batching

The function may split a long list into smaller requests or GPU batches. That is an internal detail. Rules E1 and E10 hold whatever the split. One OpenAI request takes at most 2048 inputs and at most 300,000 tokens summed across them (fact 7), so any split must stay inside both limits.

| ID  | Rule | Test |
| --- | ---- | ---- |
| E12 | The OpenAI backend may skip counting tokens only when the texts' total UTF-8 size is at most 300,000 bytes, because a text never has more `cl100k_base` tokens than UTF-8 bytes. Above that size it counts every text and splits the requests so that none goes over 300,000 tokens. | unit |

### Failures

There are two kinds of failure. They are separate error types so callers can tell them apart.

- **Configuration error.** The call can never succeed as asked: unknown model, bad revision, missing API key, empty text, text too long, wrong dimension. Retrying won't help.
- **Embedding unavailable.** The call might succeed later: network failure, timeout, rate limit, OpenAI server error, GPU out of memory.

How "embedding unavailable" maps to an answer state is D11. This spec only requires the two to be distinct.

| ID  | Rule                                                                                                                                                              | Test |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- |
| F1  | Each failure listed above under "configuration error" raises the configuration error type, and each listed under "embedding unavailable" raises the unavailable type. A NaN or infinite value, a wrong number of vectors and a zero vector are configuration errors too, because none of them can succeed on a retry. | unit |
| F2  | A missing OpenAI API key is a configuration error raised before any request is sent.                                                                              | unit |
| F3  | A call either returns all N vectors or raises. It never returns some of them.                                                                                     | unit |
| F4  | A transient OpenAI failure is retried at most 2 times. If it still fails, the call raises "embedding unavailable". The count is our choice; see fact 9.            | unit |
| F5  | No error message or log line contains the API key.                                                                                                                | unit |

### The fake embedder

Tests need an embedder with no network and no GPU (PRD constraints, ADR 0001).

| ID  | Rule                                                                                                                                         | Test |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------- | ---- |
| T1  | A fake embedder obeys the same contract as the real function: rules E1 to E9 and F1 to F3 pass against it.                                    | unit |
| T2  | The fake's identity (model ID, revision, dimension) is set by the test, so a test can build two fakes that differ only in model ID.           | unit |
| T3  | The fake is deterministic: the same text and the same identity always give the same vector. Different model IDs give different vectors for the same text. | unit |
| T4  | The same contract tests run against the fake by default, and against each real model under the `model` or `network` marker.                    | unit |

## 3. The index record

The record says which model built an index. It is the only source of truth for that: not a config file, not a folder name, not the vector size (ADR 0001).

### Content

The record is a JSON object, UTF-8, with exactly these keys:

| Key              | Type               | Value                                                                       |
| ---------------- | ------------------ | --------------------------------------------------------------------------- |
| `record_version` | integer            | `1`                                                                         |
| `model`          | string             | one of the three registry model IDs                                         |
| `revision`       | string or null     | the 40-character commit hash for a local model, null for OpenAI             |
| `dimension`      | integer            | the registry dimension for `model`                                          |

Where the record is stored, and how it sits beside the vectors, is D7. Any store must be able to save and load this JSON object unchanged.

| ID  | Rule                                                                                                                                                                         | Test |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- |
| X1  | A record is readable only if it is valid JSON, has exactly the four keys above with the stated types, and `record_version` is `1`. Anything else is an unreadable record.    | unit |
| X2  | A readable record also has to be consistent with the registry: `model` is a registry ID, `revision` follows M2 or M3, and `dimension` equals the registry dimension. If not, it is unreadable. | unit |
| X3  | Writing a record and reading it back gives the same four values.                                                                                                             | unit |

### Building an index

| ID  | Rule                                                                                                                                                         | Test |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---- |
| B1  | The record's values are the identity returned by the embedding function during the build (E9), not values copied from config.                               | unit |
| B2  | Every vector in an index was made by the model in its record. A build never adds a vector whose identity differs from the record.                           | unit |
| B3  | Every vector in an index has length equal to the record's `dimension`.                                                                                      | unit |
| B4  | The record is written last, after every vector is stored. A build that stops early (for example on a file that needs approval, R8) leaves an index with no record. | unit |
| B5  | An existing record is never changed. Building with a different model creates a different index, and leaves the first index's record and vectors untouched (R10). | unit |
| B6  | Building an index from no texts raises a configuration error and writes no record, because a record for an empty index would look like a valid index. | unit |

## 4. The query check

This is ADR 0001's "which model embeds the question", stated as rules.

| ID  | Rule                                                                                                                                                                              | Test |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- |
| Q1  | Search reads the index's record before it embeds the question. If the record is missing or unreadable, search stops with an index record error. The embedder is not called.     | unit |
| Q2  | The question is embedded with the record's `model`. The query side never takes the model from config or anywhere else.                                                           | unit |
| Q3  | After embedding, search compares the returned identity with the record. If model, revision or dimension differs, search stops with a model mismatch error that names both models. | unit |
| Q4  | A model mismatch or a record error searches nothing and produces no answer. It is never reported as "not found" or as "unavailable".                                              | unit |
| Q5  | An eval run that hits a model mismatch or a record error exits with a failure, not with a score.                                                                                  | unit |
| Q6  | Two models with the same dimension still count as a mismatch. Test: a fake index recorded as model A, queried through a fake identity of model B with the same dimension, stops at Q3. | unit |

## 5. Interface

These are the names the tests import. They are not code. They say only what each name takes and gives, so the rules above can be tested. Everything not listed here is internal.

### Modules

| Module                          | Holds                                                                 |
| ------------------------------- | --------------------------------------------------------------------- |
| `build_production_rag.embeddings` | the registry, the identity, the embedding function, backends, embedding errors |
| `build_production_rag.index`      | the index record, the store protocol, build and search, index errors  |

### Registry (`embeddings`)

| Name                     | What it is                                                                                                                                                                                                                       |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ModelSpec`              | One registry entry. A frozen dataclass with keyword-only fields: `id` (str), `revision` (str or `None`), `dimension` (int), `max_input_tokens` (int), `remote_code_repo` (str or `None`), `remote_code_revision` (str or `None`). An empty string in `models.toml` (`revision = ""`) becomes `None`. A missing remote-code key becomes `None`. |
| `Registry`               | The set of `ModelSpec`, looked up by model ID. `Registry.from_toml(path)` reads a file in the `models.toml` format and applies M2 and M3 while reading. A violation raises `ConfigurationError`. `Registry(specs)` builds one directly from an iterable of `ModelSpec`, so a test can use fake models. `registry[model_id]` returns the `ModelSpec`, and raises `KeyError` for an unknown ID (`embed` turns that into `ConfigurationError`, E3). `registry.ids()` returns the IDs in file order. |
| `load_registry()`        | `Registry.from_toml` on `models.toml` in the current working directory. Used when no registry is passed in.                                                                                                                                     |

### Identity and result (`embeddings`)

| Name         | What it is                                                                                                                                 |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `Identity`   | A frozen dataclass with keyword-only fields, compared by value: `model` (str), `revision` (str or `None`), `dimension` (int). The same three values as the index record. |
| `Embeddings` | What the function returns. A frozen dataclass with keyword-only fields: `vectors`, a float32 NumPy array of shape (N, dimension), row *i* for text *i*; and `identity`, an `Identity`. For N = 0 the shape is (0, dimension). |

### The embedding function (`embeddings`)

`embed(model_id, texts, *, registry=None, backend_factory=None) -> Embeddings`

- `model_id` is a registry ID, and `texts` is a sequence of strings.
- `registry` defaults to `load_registry()`.
- `backend_factory` is called with a `ModelSpec` and returns a backend. Calling it counts as "loading the model" for E2 and E3. It defaults to the real backends: OpenAI for `text-embedding-3-small`, and local for the other two.
- Order of work: look up the model (E3), check every text (E4), call `backend_factory`, count tokens (E5), embed, then check and normalise the result (E6, E7, E8). Identity comes from the backend (E9).

### Backend protocol (`embeddings`)

A backend is anything with these three members. The real backends and the tests' fake all follow it.

| Member               | What it does                                                                                                                         |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `revision`           | The revision actually loaded (str), or `None` for OpenAI. `embed` puts it in the identity (E9).                                      |
| `count_tokens(text)` | The model's own token count for one text, including special tokens (fact 2). `embed` compares it with `max_input_tokens` (E5).        |
| `embed_raw(texts)`   | Raw vectors, one row per text, before any check or normalisation. Batching happens here (fact 7). Raises `EmbeddingUnavailable` on transient failures. |

`OpenAIBackend(spec, *, api_key=None, http_client=None, token_counter=None)` is the real OpenAI backend. The API key comes from the `OPENAI_API_KEY` environment variable when `api_key` is `None`. A missing or empty key raises `ConfigurationError` when the backend is created, before any request (F2). `http_client` lets a test pass an `httpx2.Client` with a mock transport (`openai` 3.59 is built on `httpx2`, not `httpx`), so F4, F5 and M4 can be tested with no network. Its retry count is 2 (F4). `token_counter` is a callable that takes one text and returns its token count, and `count_tokens` uses it. When it is `None`, the backend counts with tiktoken's `cl100k_base` (fact 2). That encoding is loaded the first time a text is counted, not when the backend is created, because tiktoken downloads it on first use. A `unit` test that goes through `count_tokens` passes a fake counter, so no `unit` test reaches the network. Only the E5 `network` case uses the real encoding.

`LocalBackend(spec)` is the real local backend. It loads the model at `spec.revision`, and remote code at `spec.remote_code_revision`.

### Embedding errors (`embeddings`)

| Name                   | Raised for                                                                                                     |
| ---------------------- | -------------------------------------------------------------------------------------------------------------- |
| `EmbeddingError`       | Base class of the two below.                                                                                   |
| `ConfigurationError`   | The configuration failures in F1. It has `position` (int or `None`) and `token_count` (int or `None`) for E4 and E5. |
| `EmbeddingUnavailable` | The unavailable failures in F1.                                                                                |

### Index record (`index`)

| Name                             | What it does                                                                                                     |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `IndexRecord`                    | Immutable: `record_version`, `model`, `revision`, `dimension`. `IndexRecord.from_identity(identity)` sets `record_version` to 1. |
| `record_to_json(record) -> str`  | The JSON object from section 3.                                                                                  |
| `record_from_json(text, registry) -> IndexRecord` | Applies X1 and X2. An unreadable record raises `IndexRecordError`.                                |

### Store protocol (`index`)

A store holds one index. This is the smallest surface the rules need, not a decision on D7. The tests use an in-memory fake.

| Member                    | What it does                                                                     |
| ------------------------- | -------------------------------------------------------------------------------- |
| `add(vectors, payloads)`  | Stores vectors (N, dimension) and one payload string per vector.                 |
| `vectors()`               | Returns every stored vector as an array.                                         |
| `write_record(text)`      | Stores the record's JSON text.                                                   |
| `read_record()`           | Returns the record's JSON text, or `None` if there is none.                      |
| `search(vector, k)`       | Returns up to `k` (payload, score) pairs. Scoring is D3.                         |

### Build and search (`index`)

`build_index(model_id, texts, store, *, batch_size=64, registry=None, backend_factory=None, embed_fn=None) -> IndexRecord`

- `texts` is any iterable of strings, embedded `batch_size` at a time through `embed`. Each text is its own payload until D10 gives the passage shape.
- If the store already has a record, the build raises `IndexRecordError` before it embeds anything, and leaves the store unchanged (B5).
- If a batch's identity differs from the first batch's, the build raises `ModelMismatchError` and writes no record (B2).
- If the iterable raises (for example a file that needs approval, R8), the exception passes through and no record is written (B4).
- The record is written once, after the last `add` (B4).
- If `texts` yields nothing, the build raises `ConfigurationError` and writes no record (B6).

`search(question, store, k, *, registry=None, backend_factory=None, embed_fn=None) -> list[tuple[str, float]]`

`embed_fn` defaults to `embed` and has the same signature. A test passes a stub that returns a chosen identity. Through `embed`, the identity's model always equals the model asked for (E9), so the stub is the only way to test a different model (Q3, Q6).

- It has no model argument (Q2). The order is: read the record (Q1), embed the question with the record's model, compare identities (Q3), then call `store.search`.
- On a record error or a mismatch it raises. It never returns an empty list for these cases, because an empty list would look like "not found" (Q4).

### Index errors (`index`)

| Name                 | Raised for                                                                                             |
| -------------------- | ------------------------------------------------------------------------------------------------------ |
| `IndexRecordError`   | A missing or unreadable record (X1, X2, Q1), or a build into a store that already has a record (B5).  |
| `ModelMismatchError` | A different identity (B2, Q3). It has `expected` and `actual` (both `Identity`). In search (Q3), `expected` is the identity in the index record and `actual` is the identity that embedded the question. In a build (B2), `expected` is the first batch's identity and `actual` is the batch that differs. Its message names both models. |

### Tests

- The fake embedder (T1 to T3) is a backend in `tests/`, with a factory that counts calls. It needs no network and no GPU. Its token count is the number of whitespace-separated words. Its vectors come from a hash of (model ID, revision, text), and they are not normalised, so E7's normalisation is exercised.
- Contract tests are parametrised over embedders: the fake by default, `jina` and `bge-m3` under the `model` marker, and OpenAI under the `network` marker (T4).
- **E5 at `network`** settles fact 2. `embed` with exactly 8192 `cl100k_base` tokens succeeds. `OpenAIBackend.embed_raw` with 8193 tokens is rejected by OpenAI. The second call goes to the backend directly, because `embed` would reject the text itself before sending it.
- **Q3 and Q6** pass an `embed_fn` stub that returns an identity for model B, while the record says model A.
- **Q5 is deferred** to the eval spec. No eval run exists yet. Its test is a skipped placeholder, so the rule ID still shows in the test list.

## Facts checked

Checked on 2026-11-28. Each fact feeds the rules named in brackets. Sources:

- **[OA-guide]** OpenAI embeddings guide, https://developers.openai.com/api/docs/guides/embeddings
- **[OA-ref]** OpenAI API reference, create embeddings, https://developers.openai.com/api/reference/resources/embeddings/methods/create
- **[OA-model]** OpenAI model page, https://developers.openai.com/api/docs/models/text-embedding-3-small
- **[OA-cookbook]** OpenAI cookbook, embedding long inputs (marked archived), https://developers.openai.com/cookbook/examples/embedding_long_inputs
- **[jina]** https://huggingface.co/jinaai/jina-embeddings-v2-base-en: `config.json`, `tokenizer_config.json`, `sentence_bert_config.json`, `modules.json`, `1_Pooling/config.json`, `README.md`
- **[bge]** https://huggingface.co/BAAI/bge-m3: the same files
- **[client]** the installed `openai` Python package, version 3.59.0: `openai/_constants.py` and `openai/_base_client.py`

| #   | Fact                                                  | Finding                                                                                                                                                                                                                                                                                                                                                                  | Source |
| --- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| 1   | Dimensions (registry, E6, X2)                         | OpenAI: "By default, the length of the embedding vector is `1536` for `text-embedding-3-small`". jina: `hidden_size` 768, and the pooling config gives `word_embedding_dimension` 768. bge-m3: `hidden_size` 1024, the pooling config gives 1024, and the card's table gives 1024.                                                                                                                   | [OA-guide], [jina], [bge] |
| 2   | Max input tokens and tokenizer (E5)                   | **OpenAI: 8192.** [OA-ref]: "8192 tokens for all embedding models", and [OA-guide]'s table gives 8192. The archived [OA-cookbook] says 8191, but its own example error message says "maximum context length is 8192 tokens". Counted with `cl100k_base` ([OA-guide] FAQ). **jina: 8192.** `max_position_embeddings` and `max_seq_length` are both 8192. Its tokenizer's `model_max_length` is a placeholder (2147483648), so the tokenizer won't enforce the limit. **bge-m3: 8192.** `max_seq_length` and the tokenizer's `model_max_length` are both 8192. `max_position_embeddings` is 8194 because XLM-RoBERTa reserves two positions. Both local limits count the special tokens the tokenizer adds. | OpenAI: the real API (see the note below the table), [OA-ref], [OA-guide], [OA-cookbook]. Local: [jina], [bge] |
| 3   | OpenAI vectors already length 1 (E7)                  | Yes: "OpenAI embeddings are normalized to length 1" ([OA-guide] FAQ). E7 still normalises every vector, as written.                                                                                                                                                                                                                                                                        | [OA-guide] |
| 4   | Pooling (D5, E10)                                     | jina: `pooling_mode_mean_tokens: true`, followed by a `Normalize` module. The card says "Please apply mean pooling when integrating the model" and needs `trust_remote_code=True`. bge-m3: `pooling_mode_cls_token: true`, followed by a `Normalize` module.                                                                                                                                          | [jina], [bge] |
| 5   | Query prefix (one function for passages and questions) | bge-m3: "the BGE-M3 model no longer requires adding instructions to the queries". jina: the card gives no query or passage prefix, and its examples embed every text the same way.                                                                                                                                                                                                | [bge], [jina] |
| 6   | OpenAI model revision (M3)                            | None. The model page lists "Default snapshot: `text-embedding-3-small`" and no dated snapshot. The response's `model` field only echoes the name.                                                                                                                                                                                                                                   | [OA-model], [OA-ref] |
| 7   | OpenAI limits per request (batching)                  | At most 2048 inputs per request, and "a maximum of 300,000 tokens summed across all inputs in a single request". An input "cannot be an empty string".                                                                                                                                                                                                                       | [OA-ref] |
| 8   | Commit hashes (D5, M2)                                | The current `main` commit on 2026-11-28. jina: `322d4d7e2f35e84137961a65af894fda0385eb7a` (last modified 2025-01-06). bge-m3: `5617a9f61b028005a4858fdac845db406aefb181` (last modified 2024-07-03). These are candidates. D5 confirms them when it pins the revisions.                                                                                                                    | Hugging Face model API, `https://huggingface.co/api/models/<id>` |
| 9   | OpenAI client retries (F4)                            | `DEFAULT_MAX_RETRIES = 2`, so F4's count matches the default. The client retries timeouts, connection errors, and the status codes 408, 409, 429 and 5xx. It also follows an `x-should-retry` header from the server. All of these fall under "embedding unavailable" in F1.                                                                                                     | [client] |

Fact 2, OpenAI limit: confirmed against the real API on 2026-11-28. `test_E5_text_over_max_tokens_fails_and_is_never_truncated[openai]` (in `tests/test_embed.py`, `network` marker) showed that `text-embedding-3-small` accepts an input of exactly 8192 `cl100k_base` tokens and rejects one of 8193. The source for this limit is now the API itself, not the docs.

History, kept to explain why that test exists: the docs disagreed. The API reference and the guide gave 8192, while the archived cookbook page gave 8191 (and showed an error message saying 8192). The registry took 8192 from the API reference, and the test was written to decide between the two.

## Notes for other decisions

- **Chunking (D2) must fit the smallest limit.** E5 fails a too-long text instead of truncating it. A truncated vector would quietly describe only part of the passage, and nobody would see it. So every passage has to fit under the smallest max input tokens of the three models, counted with each model's own tokenizer.
- **Model in the trace (R11).** R11's trace line needs a model field, so traces from different models can be told apart during D4. The PRD is changed separately.
- **Trace line on a mismatch** is not settled here. It depends on the answer states and trace fields, which belong to D10 and D11. Q4 stands as written.

## Status

Draft. Every fact is now checked and its source recorded (2026-11-28). It moves to accepted when you review "Facts checked", and in particular the 8192 limit in fact 2.
