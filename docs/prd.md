# Requirements: answering questions from approved sources

Date: 2026-11-19. Based on [interview-notes.md](interview-notes.md). Decisions still open there are not settled here.

## Problem

Developers on a small AI team have questions about Python and AI engineering. An answer is only useful if they can check where it came from. A lead decides which sources may be searched. The first demo app (`build-rag`) answers questions, but nobody has measured it, so it isn't trusted yet.

## Users

- **Developer.** Asks a question from the command line. Gets an answer and the sources it came from, or is told that the sources don't cover it.
- **Lead.** Approves which source files, at which version, can be searched.

## Goal

Answer only from the approved sources, cite every answer, and measure the result against the demo app.

## Sources (first version)

Frozen copies, already downloaded and listed in a manifest. Nothing is crawled.

- The Python Tutorial 3.14 (HTML), PSF-2.0.
- Prompt Engineering Guide by DAIR.AI, English guide pages pinned to one commit, MIT.
- Two arXiv papers, CC BY 4.0: *Seven Failure Points When Engineering a RAG System* (Barnett et al., 2024) and *Ragas* (Es et al., 2023).

## Requirements

| ID | Requirement |
|---|---|
| R1 | A developer asks a question from the command line and gets an answer. |
| R2 | Every answer cites the sources it used. |
| R3 | If the sources don't cover a question, the system says so. It never answers from general knowledge. |
| R4 | Every answer has one of three states: found, not found, or unavailable. |
| R5 | Every passage the system can return keeps its citation. |
| R6 | A list that runs across a PDF page break stays together. |
| R7 | Ingestion records a content hash for each source file. |
| R8 | A file whose content has changed since it was approved is not ingested. The system stops and reports that the file needs approval. |
| R9 | Two embedding models, OpenAI `text-embedding-3-small` and a local jina model, sit behind one function. |
| R10 | Changing the embedding model rebuilds the whole index. |
| R11 | Every question writes one trace line holding the question, the passages returned with their scores, the answer state, and the sources cited. |

## How we measure it

- Eleven hand-written questions. Ten have a known source, labelled by file and section. One has no answer on purpose.
- The same questions run against the demo app. Each score for the new system is reported next to the demo app's score.
- The metric for now is hit rate on the right source.
- The result is reported either way. If the new system doesn't beat the demo app, that is a finding.

## Constraints

- The source files are third-party copies and are never committed. Licences and attribution are recorded.
- No RAG framework. The system is a thin layer of our own code that tests can run with fakes.
- The shapes of the data passed between steps (document, passage, answer) are agreed in the spec before any code is written.
- OpenAI generates the answers.

## Out of scope for now

- An approval workflow (for now, a changed file just stops with a message).
- Scoring answers with a model as judge.
- Tracking cost and latency.
- A web page or API.
- Crawling or adding sources.
- More than eleven questions.

## Decided later

Chunking, retrieval method, which embedding model wins, vector store, how refusal is decided, which generation model is used, and what "unavailable" means. See the open table in [interview-notes.md](interview-notes.md).
