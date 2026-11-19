# Interview notes

Design interview held on 2026-11-19. These are the decisions reached and what is still open.

## Decided

### Purpose and scope
- A portfolio and video-series project that shows production RAG practice.
- Built from an empty project. No code is copied from `build-rag`.
- Interface: CLI only for now.

### Baseline
- `build-rag` (the one-prompt demo app from the start of the video) stays frozen as the baseline.
- The same eleven questions run against both systems, and each score for the new system sits next to the baseline's.
- If the new system doesn't beat the baseline, that is a finding in its own right.

### Corpus
- Three frozen sources, already downloaded and listed in a manifest. Nothing is crawled.
  - Python Tutorial 3.14, HTML (17 files), PSF-2.0.
  - Prompt Engineering Guide by DAIR.AI, `pages/**/*.en.mdx` only (134 pages), pinned to commit `57673726396dd94acb23bdb1e67f27c78ee85a8e`, MIT.
  - Two arXiv papers, text-based, CC BY 4.0:
    - Barnett et al., 2024, *Seven Failure Points When Engineering a Retrieval Augmented Generation System*, arXiv:2401.05856.
    - Es et al., 2023, *Ragas: Automated Evaluation of Retrieval Augmented Generation*, arXiv:2309.15217.
- The corpus files are third-party copies and are not committed.
- Ingestion stores a content hash per file.
- A changed file is not re-ingested. It is flagged and stops with a message that it needs approval, because the lead approved the earlier version. The approval gate itself comes later in the series.
- Changing the embedding model rebuilds the whole index. This goes in the first ADR.

### Answer contract
- Every answer cites its sources.
- When the corpus doesn't cover a question, the system says so. It does not answer from general knowledge.
- Answer states: found, not found, unavailable.

### Chunking
The strategy is not decided yet. Two commitments hold now:
- Every passage keeps its citation.
- A list that runs across a PDF page break stays together. This is what the demo app got wrong.

### Models
- Embeddings: OpenAI `text-embedding-3-small` and a local jina model, behind one function. The eleven questions decide between them.
- Generation: OpenAI.

### Architecture
- No RAG framework (no LangChain or LlamaIndex). A thin layer of our own code with small interfaces that tests can replace with fakes.
- The shapes of the data passed between steps (document, passage, answer) are agreed in the spec before any code is written.

### Evaluation (first version)
- Eleven hand-written questions: ten with a known source, and one with no answer on purpose. The set grows later.
- Each expected answer is labelled by file and section, not by chunk. Chunking isn't decided, and chunk IDs will change.
- Metric for now: hit rate on the right source.
- The LLM judge waits until the system is generating answers.

### Tracing (first version)
- One JSONL trace line per question, holding:
  - the question
  - the passages that came back, with their scores
  - the answer state (found, not found, unavailable)
  - the sources cited
- Cost and latency tracking wait until the system works.

## Open

| # | Decision | Waiting on |
|---|---|---|
| 1 | Chunking strategy for each source type | Being able to test it |
| 2 | Retrieval approach: dense, hybrid, or with a reranker | The baseline results. Hypothesis to test: exact names such as `__init__` favour keyword search (BM25) over dense search. |
| 3 | Which embedding model wins: OpenAI or jina | The eleven questions |
| 4 | Which jina model and version, and its runtime | Not discussed yet |
| 5 | Torch and CUDA support for the local model: Python 3.14 on an RTX 5070 Ti (Blackwell) needs a torch build with CUDA 12.8 or later | To check when we get to the local model |
| 6 | Vector store | Deferred |
| 7 | How refusal is decided: a score threshold, the model, or both | Deferred |
| 8 | Which OpenAI generation model, and the prompt format for citations | Not discussed yet |
| 9 | Exact fields and types for document, passage and answer | The spec |
| 10 | How "unavailable" differs from "not found" | The spec |
| 11 | The wording of the eleven questions and their file and section labels | Written by hand |
| 12 | Approval-gate design: who approves, and how approval is recorded | A later episode |
| 13 | LLM judge, a larger question set, cost and latency tracing | Later |

## Next steps

1. One-page requirements.
2. Decisions table.
3. First ADR: changing the embedding model rebuilds the whole index.
4. Design.
5. Spec, including the shapes of document, passage and answer.
