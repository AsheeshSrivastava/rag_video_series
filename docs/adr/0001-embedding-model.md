---
created: 2026-11-20
modified: 2026-11-25T12:45:38+05:30
status: accepted
description:
---
# ADR 0001: One index per embedding model

- Requirements: R9, R10 in [prd.md](../prd.md)
- Related open decisions: D4, D5, D7 in [decisions.md](../decisions.md)
- Amendment 1 (2026-11-25): bge-m3 added as a second local model (R9)
---
## Context

R9 says three embedding models sit behind one function: OpenAI `text-embedding-3-small`, and two local models, jina-embeddings-v2-base-en and BAAI/bge-m3. The eleven questions in [eval-questions.md](../eval-questions.md) decide which one wins (D4).

R10 was first worded 'changing the embedding model rebuilds the whole index', and the interview notes give no reason for it. The usual reason is that vectors from two different models can't be compared with each other, so an index must never hold a mix (to verify against each model's documentation).

R7 and R8 constrain any rebuild. Ingestion records a content hash for each file, and a file that changed since it was approved is not ingested. A rebuild re-reads every file, so it has to pass the same check.

Unknown so far:
- Revisions of the two local models, and their runtime (D5): models chosen, revisions to record.
- Torch with CUDA on Python 3.14 and an RTX 5070 Ti (D6): checked 2026-11-25, it runs.
- How many passages the corpus produces. This depends on chunking (D2): to verify.
- How long a full rebuild takes with each model, and what it costs with OpenAI: to verify.
- The vector store (D7). It's deferred, so this ADR must not depend on any one store.

The question this ADR answers: **how does the system hold indexes for three models, given that a model change must never leave an index with mixed vectors?**

## Options

**A. One index, rebuilt in place.** The index records which model built it. If the configured model differs, the whole index is thrown away and rebuilt before any question is answered. Comparing the three models means rebuilding every time you switch.

**B. One index per model.** Each model has its own index, built from scratch the first time that model is used. Switching models selects the matching index and doesn't rebuild one that already exists. Both indexes can be kept, so the eleven questions can run against each without rebuilding. This follows the reason behind R10 (never mix vectors) but not its wording (a switch doesn't always rebuild). R10 was reworded to match (af6e1e1).

**C. No stored index.** Every run embeds the corpus from scratch with the configured model. Nothing can go stale, but every run pays the full embedding time, and the OpenAI cost on every run (to verify).

## What we care about

1. **No mixed vectors.** An index never holds vectors from two models (the reason behind R10).
2. **A fair comparison for D4.** All three models are scored on the eleven questions over the same passages, so the only thing that differs is the model.
3. **Cheap, repeatable comparison.** Running the eleven questions against each model shouldn't mean waiting for a rebuild or paying for one more often than needed (time and cost: to verify).
4. **Approval still holds.** Every rebuild passes the same per-file hash and approval check as first ingestion (R7, R8).
5. **Independent of the store.** The choice works whatever vector store D7 picks.
6. **Testable with fakes.** Tests can check the rule (for example, "changing the model never reuses the old vectors") with a fake embedder, with no network and no GPU (PRD constraints).
7. **Simple.** Thin code of our own, no framework (PRD constraints).

## Decision

**Option B: one index per model.**

Each embedding model has its own index, built from scratch the first time that model is used. An index is only ever built by one model. Switching models selects the matching index. Mixing vectors is impossible by construction, which is what R10 protects, and D4 can run the eleven questions against all three models without repeated rebuilds.

Not chosen:
- **A** meets R10 word for word, but every switch during D4 costs a full rebuild.
- **C** pays the full embedding time and the OpenAI cost on every run, for no gain over B.

**Which model embeds the question.** One index per model keeps vectors from mixing inside an index, but it doesn't stop a question embedded by one model from being searched against another model's index. If the embedding sizes match, that search runs and returns scores that mean nothing, and it fails silently as "not found" or with wrong citations. To prevent this:

1. **The model is recorded in the index.** When an index is built, a small record is written with the vectors: the model name, the model revision (for the local model; whether OpenAI exposes one is to verify) and the embedding size. It sits alongside the vectors, not inside a particular store's format, so it doesn't depend on D7. The record, not a config file or folder name, is what says which model built the index.
2. **The question is embedded with the index's model.** Search reads the index's record and embeds the question with that model through the one function from R9. The query side never chooses a model on its own.
3. **A mismatch stops before searching.** As a safety net, search checks that the model that embedded the question matches the record. If it doesn't, the system stops with an error that names both models. It writes no answer, and an eval run fails. A mismatch is treated as a setup error, never as "not found" and never as "unavailable". D11 stays open for failures at answer time.

R10 in the PRD now reads: "An index is only ever built by one embedding model. Using a different model means a separate index built from scratch."

## Consequences

Whichever option is chosen:
- A rebuild can be stopped by a file that needs approval (R8), so it can't be assumed to always finish.

For option B, as chosen:
- Each index carries a record of the model that built it: name, revision where available, and embedding size. That record is the only source of truth for which model an index belongs to.
- The one embedding function from R9 is told which model to use. At query time it gets the model from the index's record.
- An index with no record, or with a record that can't be read, can't be searched. It is treated like a mismatch.
- Tests with a fake embedder can check three rules: an index is built by one model only, a question is embedded with the index's model, and a forced mismatch stops with an error instead of answering.
- R11's trace line doesn't include the model name. If traces from different models have to be told apart during D4, R11 may need that field: to verify.
- The three models have different vector sizes (to verify: 1536, 768 and 1024), so a wrong pairing often shows up as a size error. The record check must still be what stops it, because a size match isn't proof of the right model.

Depending on the option:
- **A:** Simplest storage. Each switch during D4 costs a full rebuild (time and cost: to verify).
- **B:** More storage and a rule for choosing between indexes. D4 runs without repeated rebuilds.
- **C:** No index to manage. Every run pays the full embedding time and cost. Whether that's acceptable is to verify once D2 gives the corpus size.
