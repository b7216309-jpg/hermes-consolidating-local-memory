# Changelog

## 3.6.0 - 2026-07-17

- Added an explicit assistant-origin turn class for Conscious Agency 1.2 heartbeat execution in
  the real Hermes conversation.
- Discarded the synthetic heartbeat trigger as user content while preserving the transformed final
  assistant output as an episode, trace, and optional assistant-source extraction.
- Kept assistant-initiated turns out of genuine-user prefetch and the `current-request` working
  memory slot, preventing model speech from being reclassified as a user fact.
- Removed disposable-heartbeat thread exclusions; ordinary sessions whose historical thread names
  resemble the retired marker now use Memory normally.
- Added regressions for same-session capture, assistant provenance, legacy-marker neutrality, and
  the unchanged internal-turn fail-closed boundary.

## 3.5.0 - 2026-07-17

- Made repeated installs preserve existing plugin enablement and grant settings instead of
  re-enabling an already loaded lifecycle observer and returning a false tool-override error.

- Made durable background work owner-bound with renewable SQLite leases, atomic claims, safe
  cross-process recovery, and dead-letter handling that cannot let a stale worker finalize another
  worker's operation.
- Kept the background worker alive across transient claim/finalization failures and made timed-out
  shutdown leave the database open until the worker safely drains and closes it.
- Hardened automatic capture against forged or stale gateway origin markers, orphan assistant
  messages, session-end transcript replay, malformed extraction timestamps, and non-finite time.
- Made pre-compression preserve only the latest genuine user turn and rebuild topics only after a
  new fact was actually inserted.
- Added true read-only store support for Control Center and FTS content-consistency diagnostics.
- Isolated exact Conscious Agency disposable heartbeat threads from Memory sessions, maintenance,
  prefetch warming, compression extraction, mirroring, and worker startup while preserving
  explicit read access.
- Added current-Hermes compatibility coverage, concurrency/recovery regressions, and a clean SPDX
  package license declaration.

## 3.4.1 — 2026-07-16

- Added cross-suite regression coverage proving Conscious Agency 1.0 native heartbeat polls and
  responses remain internal and cannot contaminate episodes, prefetch, or fact extraction.
- Documented main-session heartbeat continuity without changing the 3.4 storage, extraction,
  retrieval, configuration, or migration contracts.

## 3.4.0 — 2026-07-16

- Reduced the injected system contract to a two-line capability marker and removed operational
  diagnostics from normal model context.
- Limited the model-facing schema to 19 conversational actions while retaining operator and
  compatibility handlers for maintenance, diagnostics, and export.
- Bounded recall to 4,500 total characters, 500 characters per line, and 20 requested results.
- Made automatic prefetch strictly relevant: no unrelated global fallback and at least two useful
  lexical overlaps for automatic recall. Explicit context/search requests remain available.
- Made snapshot disable cleanup remove only plugin-owned marked blocks, preserving manual Hermes
  memory files.
- Added regression coverage for prompt size, relevance, fallback behavior, line boundaries,
  operator action visibility, and snapshot ownership.

## 3.3.2 — 2026-07-16

- Removed an unused wiki importance renderer and the unreachable fallback for a store API that is
  part of the installed provider contract.
- Removed the stale duplicated Control Center integration guide; the companion app now owns its
  operational documentation and this repository links to it directly.
- Kept the database schema, configuration contract, Hermes hooks, extraction, retrieval, and live
  memory behavior unchanged.

## 3.3.1 — 2026-07-15

- Added deterministic human-turn provenance using Hermes' user-only `pre_gateway_dispatch` hook
  and a bounded thread-safe handoff to asynchronous memory-provider work.
- Added explicit dual-path Hermes 0.18.2 registration: provider discovery supplies storage, general
  plugin discovery supplies hooks, and both namespaces share one in-process origin ledger.
- Updated the installer to enable the lifecycle observer without granting tool overrides and fail
  visibly when Hermes cannot activate it.
- Failed closed for unclassified gateway turns while preserving direct human CLI capture.
- Excluded background process, delegation, recall, compression, kanban, background-review, and
  other synthetic turns from prefetch, episodes, traces, working memory, extraction, and summaries.
- Added regression coverage for genuine Telegram capture, internal Telegram exclusion, background
  review pairing, plugin hook registration, and scoped gateway privacy behavior.

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
