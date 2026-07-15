# Plugin architecture and operations

## Runtime flow

Hermes calls `sync_turn` after a completed turn. The provider adds a bounded task containing the user and assistant text plus the current structured message roles/tool names. Critical overflow is spooled to SQLite and replayed; optional prefetch work may be dropped under pressure. Companion-memory writes are always durably spooled before execution. If an extraction model and endpoint are configured, the completed turn is sent to that model after privacy admission; otherwise automatic extraction is disabled. With `llm_disable_thinking: true`, OpenAI-compatible chat requests carry `chat_template_kwargs.enable_thinking=false`. This mirrors Hermes compression while keeping the plugin's endpoint opt-in and independent.

Recall stays responsive: synchronous `prefetch` uses local FTS only. Hermes can call `queue_prefetch` after a turn to precompute an embedding-reranked result for the next turn. Cache entries expire and every mutation invalidates them. The recall header supplies localized current time and labels every recalled memory with its relevant absolute/relative time.

Session IDs can rotate without process restart. `on_session_switch` changes the provider target immediately and records continuation lineage. Already queued writes retain their original explicit session ID.

Shutdown is drain-first. New tasks are rejected, a FIFO sentinel is appended, and accepted work completes in order. If the worker cannot exit within the bounded wait, queued tasks are moved to the durable spool and the connection remains open only until the in-flight call returns; the worker then closes it itself.

## Fact state model

Facts have normalized content, a fingerprint, topic, category, confidence, importance, salience, provenance, review scheduling, structured temporal state, sensitivity, pin state, revision number, observation count, and belief score. Every observation retains its source role, reliability, session, timestamp, confidence, correction flag, and metadata.

Temporal state is deliberately decomposed:

- `temporal_kind`: `atemporal`, `current`, `event`, `scheduled`, or `temporary`;
- `event_at`: when something happened or is planned, not when it was learned;
- `valid_from` / `valid_until`: the interval during which a state applies;
- `temporal_precision`: the finest supported unit actually known;
- `temporal_timezone`: the IANA interpretation used for local input;
- `temporal_confidence`: confidence in the temporal interpretation.

Creation, update, observation, and event timestamps are never collapsed. The extractor receives the current Unix reference, localized ISO timestamp, and Hermes timezone, resolves relative expressions, and cannot invent an hour when only a date is known. An undated event remains an undated fact rather than being placed on the timeline at observation time. Dated extracted and explicit `remember` facts share the same deterministic fact-to-timeline link. One-time scheduled facts expire from current-state recall after their precision window, while an autobiographical timeline link survives as an unconfirmed plan.

- `subject_key` identifies the property, such as `project:database`.
- `value_key` identifies a normalized value or facet.
- `exclusive` means a new state may replace an older state.
- `polarity` distinguishes positive and negative assertions.

For a normal exclusive subject, the strongest evidence wins and the losing assertion remains as inactive history. Direct user corrections and statements outrank assistant inference. `conflict_policy: newest` is available for last-write-wins compatibility. Faceted subjects still resolve only within the same value.

Working memory is session-bound, capacity-limited, prioritized, and expiring. Procedures retain prerequisites, steps, success criteria, recovery, and outcome statistics. Prospective memory tracks due or conditional intentions. Autobiographical events form a temporal timeline. Weighted associations strengthen on co-observation and support pattern-completion recall.

## Consolidation

The consolidation gate uses elapsed hours and distinct pending sessions. Batch size and immediate backlog passes are configurable. A database lease prevents concurrent processes from consolidating the same scope. The cursor advances only to the highest episode actually processed.

A pass performs duplicate reconciliation, decay, stale-fact pruning, topic rebuilding, session-summary refresh, eligible raw-episode cleanup, history compaction, and optional wiki/snapshot output. Extraction happens once when the turn or session messages are captured, not again during consolidation.

## Storage and migration

The default database is `$HERMES_HOME/consolidating_memory.db`. Gateway identities default to a separately hashed database per user; agent and global scopes are configurable. Wiki output is likewise separated by scope. Writes to shared Hermes `USER.md`/`MEMORY.md` files are refused from user- or agent-scoped stores. Startup migrations are additive and recorded in `schema_migrations`. The version-3 temporal migration adds the structured columns/index and conservatively classifies legacy facts without fabricating event dates. FTS tables are versioned and count-checked against their sources.

Episode retention removes FTS rows and episode links and clears trace source references, preventing dangling provenance.

## Export safety

Sensitive facts and matching artifacts are omitted recursively from wiki and portable JSON exports by default. This can be changed explicitly with `export_redact_sensitive: false`. Portable imports reconstruct memory state, evidence, and remapped relationships; verified SQLite backups preserve the complete, unredacted operational database.

Wiki pages are written to temporary files, flushed, and atomically replaced. Dynamic Markdown content is escaped. A manifest records generated paths. Later exports prune only stale files in that manifest, so manually created notes—even inside `topics/` or `sessions/`—remain untouched. Export to a filesystem root or directly to `HERMES_HOME` is rejected.

## Privacy and isolation

Default operation is entirely local. Sensitive candidates follow `deny`, `ask`, or `allow`; `ask` creates a durable approval item. Credentials are always denied unless `allow_credential_memory: true` is explicitly enabled. Optional SQLCipher encryption fails closed when unavailable. LLM and embedding clients are disabled until their own endpoint is explicitly configured, and repeated failures open a cooldown circuit. Even after an endpoint is configured, sensitive text is withheld unless `allow_sensitive_model_processing: true` is explicitly enabled; credential processing also requires `allow_credential_memory: true`. In strict non-thinking mode, the extractor does not fall back to `reasoning_content`: an empty visible response fails and enters the same durable retry path as other extraction failures. Guided onboarding adds a separate `local_only` provenance flag to approved profile entries. Those entries remain normally visible to local recall, while any matching result set bypasses the remote embedding endpoint. Rebuilt topic summaries inherit the flag.

Hermes initializes providers with an `agent_context`. Only the primary context can write. Cron, flush, and subagent instances can read recall context without letting system prompts or tool output become user memory. Message extraction accepts only user/human and assistant/AI roles; system, developer, and tool roles are ignored.

### Onboarding and scope reproduction

Onboarding builds normal memory objects rather than a parallel profile table. Stable deterministic keys are used for single-valued preferences and policies; facts use normal fingerprints and subject-state reconciliation; goals are deduplicated against pending intentions. The entire approved batch is applied inside one outer transaction with nested savepoints. A repeated identical plan is detected before each write and produces `unchanged` results without new evidence or history.

The native CLI can reproduce the provider's user/agent scope path from the configured base database, runtime platform, user ID, and optional agent identity. It joins the same scope components with the same separator, hashes them with SHA-256, and uses the same 24-character prefix. Raw identities therefore do not appear in filenames. Scope options are explicit because copying an owner's profile into every user scope would violate isolation.

Interactive input strips terminal byte-order markers, bounds answer and list sizes, rejects unknown template keys, and defaults to cancellation. Credential-shaped text is omitted from both the plan and store. Never-remember policies remain non-sensitive instructions even when they name categories such as medical or financial data; they do not contain the excluded information itself.

## Recovery

If Hermes is stopped normally, accepted and spooled writes drain. After an abnormal process kill, SQLite WAL recovery and durable-operation replay recover committed work. Retries use bounded exponential backoff; poison tasks move to a dead-letter state instead of looping forever. The `doctor` action checks integrity, FTS counts, dangling links, active/dead-letter queue depth, and size. The native `hermes consolidating_local` CLI supports reviewed profile onboarding, verified backup/restore, redacted JSON export/import, maintenance, and an explicit confirmed `retry-failed` recovery command.
