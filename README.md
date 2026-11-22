# Building a Production RAG System

Companion repository for the YouTube video series on building a citation-grounded RAG system from scratch.

A one-prompt demo app (`build-rag`) can answer questions, but it fails silently across PDF page breaks and gives no way to measure accuracy. Rather than starting with retrieval or framework code, this project starts with requirements, open decisions, an evaluation benchmark, and Architecture Decision Records (ADRs) before writing implementation code.

## Current Status: Episode 1 (No Implementation Code Yet)

The repository currently holds the requirements and architecture artifacts produced in Episode 1:

- **[Interview Notes](docs/interview-notes.md)** — Initial scope, constraints, and open questions from the design interview.
- **[Product Requirements (PRD)](docs/prd.md)** — Requirements (`R1`–`R11`), answer states (`found`, `not found`, `unavailable`), and pass criteria against the baseline demo.
- **[Open Decisions](docs/decisions.md)** — Unsettled choices (`D2`–`D14`) and what must be checked or measured before deciding each one.
- **[Evaluation Benchmark](docs/eval-questions.md)** — Eleven hand-written questions (ten mapped to specific files and sections across HTML, Markdown, and PDF sources, plus one deliberate unanswerable refusal check).
- **[ADR 0001: Embedding Model Isolation](docs/adr/0001-embedding-model.md)** — Decision to maintain separate per-model indexes with recorded model metadata so switching models never silently mixes vector spaces.

## Setup

Third-party source documents (`sources/`) are excluded from version control per the [PRD constraints](docs/prd.md).

- **Python:** `>= 3.14`
- **Environment & Dependency Management:** [`uv`](https://docs.astral.sh/uv/)

```bash
uv sync
```
