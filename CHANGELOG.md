# Changelog

## 3.3.0 — 2026-07-15

- Added structured fact time with explicit temporal kind, event time, validity interval, precision,
  source timezone, and confidence.
- Supplied model extraction with the local ISO reference time and IANA timezone so relative dates
  are resolved without inventing missing hours.
- Distinguished event time from observation, creation, and update time throughout recall, timeline,
  wiki, import/export, split, and direct tool writes.
- Linked dated explicit `remember` facts into the autobiographical timeline and refused to fabricate
  an unknown event date from observation time.
- Added current-time and relative-age labels to every recalled memory system, including working,
  prospective, autobiographical, provenance, and contradiction views.
- Expired one-time schedules from current-state recall while retaining their linked timeline record;
  past plans are explicitly not represented as confirmed outcomes.
- Added the additive `structured_temporal_context` migration and conservative legacy backfill.
- Hardened all temporal storage values against nonnumeric, infinite, negative, and NaN input.
- Added migration, extraction, expiry, timeline, rendering, and malformed-input regression tests.

## 3.2.0 — 2026-07-14

- Added opt-in strict non-thinking extraction through `llm_disable_thinking` for compatible OpenAI-style Qwen endpoints.
- Matched Hermes compression by sending `chat_template_kwargs.enable_thinking=false` in the raw chat-completion request.
- Rejected reasoning-only responses in strict mode so scratch reasoning cannot be parsed and stored as memory.
- Preserved backward compatibility for other endpoints and Codex Responses backends.
- Added transport, response-safety, and provider-integration regression coverage; validated the release against a live Qwen 35B extraction endpoint.

## 3.1.0 — 2026-07-14

- Added `hermes consolidating_local onboard`, a guided user-profile interview with a full preview and explicit approval before any write.
- Mapped onboarding answers into existing semantic facts, preferences, policies, procedures, and prospective intentions with deterministic keys and idempotent reruns.
- Added JSON answer templates, non-interactive previews, confirmed application, strict input bounds, and a `--skip-sensitive` mode.
- Added explicit platform/user/agent scope targeting that derives exactly the same privacy-preserving database path as the live provider.
- Rejected credential-like onboarding answers without echoing their content into the plan or database.
- Added `local_only` provenance so approved profile memories remain locally searchable without being sent to remote embedding endpoints; rebuilt topic summaries inherit the restriction.
- Kept never-remember policies visible even when they name excluded sensitive categories, and stripped terminal BOM markers from piped first answers.
- Added regression coverage for preview, cancellation, validation, credential rejection, atomic/idempotent writes, memory classification, exact scope derivation, topic propagation, and the live hybrid privacy gate.

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
