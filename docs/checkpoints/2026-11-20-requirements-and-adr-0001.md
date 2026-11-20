# Checkpoint 2026-11-20: requirements, decisions table, ADR 0001

## Where we are

Design is done up to the first ADR. No code has been written yet.

Commits on `main`, oldest first:
- `b52e36e` Interview notes (`docs/interview-notes.md`)
- `19eb525` One-page requirements (`docs/prd.md`)
- `d56699b`, `9c27e85`, `b187a17` (by the user) Eleven evaluation questions (`docs/eval-questions.md`), the pass mark in the PRD, and "not found" for question 11
- `6a794db`, `99b9a3d` Open decisions table (`docs/decisions.md`): D1 removed once the questions landed, star dropped, D14 added
- `af6e1e1` ADR 0001 (`docs/adr/0001-embedding-model.md`) and R10 reworded

## Key decisions so far

- Empty project, no code copied. `E:\build-rag` (the one-prompt demo app) is the frozen baseline. The same eleven questions run against both, and scores sit side by side.
- CLI only. OpenAI for generation. Embeddings: OpenAI `text-embedding-3-small` and a local jina model behind one function.
- Pass mark: the right source for at least 8 of the 10 answerable questions, and "not found" for question 11.
- ADR 0001 (accepted): one index per embedding model. Each index records its model (name, revision where available, embedding size). The question is embedded with the index's model. A mismatch, or a missing record, stops with an error and is never reported as "not found" or "unavailable".
- A changed source file is not ingested. It stops with a message that it needs approval. The approval workflow comes in a later episode.

## Next

1. Design.
2. Spec: settles D10 (shapes of document, passage and answer) and D11 ("unavailable" vs "not found").
3. D14 (does a hit on questions 9 and 10 need either source or both?) must be settled before the demo app's baseline run can be scored.

## Loose ends

- The scaffold (`.gitignore`, `.python-version`, `README.md`, `pyproject.toml`, `src/`, `uv.lock`) is untracked on purpose. The user wants to review it before it's committed.
- Python 3.14 on an RTX 5070 Ti needs a torch build with CUDA 12.8 or later (D6). Check this when we get to the local model.
- The corpus root is `sources/` in this project, and it's gitignored. Verified on 2026-11-20: all 155 files have the same SHA-256 hashes as `E:\build-rag\sources\`, the counts and sizes match the manifest, and every path cited in `eval-questions.md` exists under `sources/`. The header of `eval-questions.md` was corrected from `prep/sources/` to `sources/` in `d581b29`.
- Git warns about LF to CRLF conversion on every docs file. This is harmless. Consider adding a `.gitattributes` when the scaffold is committed.
- This checkpoint file is not committed yet.
