# Decisions still open

Date: 2026-11-20. Taken from the open table in [interview-notes.md](interview-notes.md) and the "Decided later" list in [prd.md](prd.md). None of these is settled here. When one is decided, its row moves to an ADR.

★ marks the decision the others depend on.

| ID | Decision | Serves | When we decide it | What we need to know first |
|---|---|---|---|---|
| D1 ★ | The wording of the eleven questions and their file and section labels | How we measure it (the pass mark) | Before anything is compared, so before the demo app's baseline run | Nothing. They're written by hand from the sources. D2, D3, D4 and D8 are all decided by running these questions, and D8 is checked by the eleventh. |
| D2 | Chunking strategy for each source type | R5, R6 | When we can test it against the eleven questions | D1. Whether a candidate strategy keeps citations and keeps a list together when it runs across a PDF page break. |
| D3 | Retrieval method: dense, hybrid, or with a reranker | R1, R3 | After the baseline run | D1 and the demo app's baseline scores. The hypothesis to test: exact names like `__init__` favour keyword search (BM25). |
| D4 | Which embedding model wins: OpenAI `text-embedding-3-small` or local jina | R9 | When both run behind the one function | D1, D5 and D6. Hit rate for each model on the eleven questions. |
| D5 | Which jina model and version, and its runtime | R9 | When we reach the local model | Which jina models fit the corpus and run locally. |
| D6 | Torch and CUDA support: Python 3.14 on an RTX 5070 Ti (Blackwell) needs a torch build with CUDA 12.8 or later | R9 | When we reach the local model | Whether a torch build for that combination exists and runs. This is a check, not a choice. |
| D7 | Vector store | R10 | Deferred, not scheduled | How the index is rebuilt when the embedding model changes (first ADR), and the passage shape (D10). |
| D8 | How refusal is decided: a score threshold, the model, or both | R3, R4 | Deferred, not scheduled | D1 (the eleventh question), D3, and what scores retrieval returns. |
| D9 | Which OpenAI model generates answers, and the prompt format for citations | R1, R2 | Not discussed yet | The answer shape (D10). |
| D10 | Exact fields and types for document, passage and answer | R2, R4, R5, R11 | In the spec | The design. |
| D11 | How "unavailable" differs from "not found" | R4 | In the spec | The design. Which failures the system can hit. |
| D12 | Approval workflow: who approves, and how approval is recorded | R8 | A later episode | How it works for now: a changed file stops with a message. |
| D13 | Scoring answers with a model as judge, a larger question set, cost and latency tracking | Out of scope for now | Later | A working first version that generates answers. |
