# Changelog

## 3.0.0 — 2026-07-14

- Removed the rule-based fact extractor, hybrid extractor mode, candidate seed facts, content guessing, and rule-based canonical rewriting.
- Automatic fact extraction is now LLM-only and activates only when both `llm_model` and `llm_base_url` are configured.
- Kept explicit Hermes memory-tool writes, episodic capture, consolidation, FTS recall, privacy controls, and all brain-inspired memory layers independent of automatic extraction.
- Stopped consolidation from re-extracting already processed turns, avoiding duplicate model calls and duplicate observations.
- Made configured extraction failures enter durable retry/dead-letter recovery instead of silently falling back or losing the turn.

## 2.0.0 — 2026-07-14

- Isolated gateway memory by user or agent with separately hashed database files; legacy CLI installs continue using the configured database.
- Added nested atomic transactions, evidence observations, reliability-weighted beliefs, temporal validity, revision numbers, pinning, and configurable evidence/newest conflict policies.
- Added working, procedural, prospective, autobiographical, and associative memory systems with pattern-completion recall.
- Added sensitive-data classification with deny/ask/allow policies, an approval inbox, export redaction, restrictive file permissions, and optional fail-closed SQLCipher encryption.
- Added bounded in-memory work queues, durable overflow and companion-write spooling, cross-process maintenance leases, backlog continuation, retention budgets, and network circuit breakers.
- Added precise deletion previews, dry-run consolidation, fact explanations, manual merge/split, database doctor/repair, backups, portable JSON import/export, and an offline administration CLI.
- Added migration history, database growth maintenance, stronger observability, and six end-to-end v2 regression suites.
- Added bounded durable-operation retries, dead-letter diagnostics, exponential backoff, retention, and explicit confirmed recovery.
- Made failed in-memory writes replay durably, keyed episode capture for idempotent retry, and expanded doctor/repair to every internal reference type.
- Prevented future-dated facts from appearing early, separated logical size budgeting from WAL size, and tightened graph curation validation.
- Kept sensitive text off configured LLM and embedding endpoints unless separately opted in, with credential processing requiring both explicit privacy switches.

## 1.0.0 — 2026-07-14

- Updated the provider lifecycle for Hermes Agent 0.18.2, including session switching, structured turns, write metadata, context isolation, and external backup paths.
- At that release, rule-based extraction and FTS retrieval were the private, network-free defaults; v3 removes the rule-based extractor.
- Added immediate per-turn fact extraction and fast synchronous recall.
- Fixed shutdown write loss, session misattribution, stale/unbounded recall caches, and consolidation cursor data loss above 500 episodes.
- Fixed exclusive-subject contradiction resolution, duplicate reactivation, unsafe signature supersession, numeric coercion, and Unicode normalization.
- Preserved closed sessions during migration and cleaned episode provenance correctly.
- Added FTS schema repair and natural-language query expansion.
- Hardened optional LLM/embedding response parsing, bounds, ordering, and validation.
- Made wiki writes atomic, escaped dynamic Markdown/HTML, and limited pruning to manifest-owned files.
- Added a cross-platform installer, Hermes discovery smoke coverage, 19 regression tests, lint/format configuration, CI, and updated documentation.
