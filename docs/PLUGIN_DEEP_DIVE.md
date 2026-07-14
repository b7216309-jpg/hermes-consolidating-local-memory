# Plugin architecture and operations

## Runtime flow

Hermes calls `sync_turn` after a completed turn. The provider adds a bounded task containing the user and assistant text plus the current structured message roles/tool names. Critical overflow is spooled to SQLite and replayed; optional prefetch work may be dropped under pressure. Companion-memory writes are always durably spooled before execution. If an extraction model and endpoint are configured, the completed turn is sent to that model after privacy admission; otherwise automatic extraction is disabled.

Recall stays responsive: synchronous `prefetch` uses local FTS only. Hermes can call `queue_prefetch` after a turn to precompute an embedding-reranked result for the next turn. Cache entries expire and every mutation invalidates them.

Session IDs can rotate without process restart. `on_session_switch` changes the provider target immediately and records continuation lineage. Already queued writes retain their original explicit session ID.

Shutdown is drain-first. New tasks are rejected, a FIFO sentinel is appended, and accepted work completes in order. If the worker cannot exit within the bounded wait, queued tasks are moved to the durable spool and the connection remains open only until the in-flight call returns; the worker then closes it itself.

## Fact state model

Facts have normalized content, a fingerprint, topic, category, confidence, importance, salience, provenance, review scheduling, temporal validity, sensitivity, pin state, revision number, observation count, and belief score. Every observation retains its source role, reliability, session, timestamp, confidence, correction flag, and metadata.

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

The default database is `$HERMES_HOME/consolidating_memory.db`. Gateway identities default to a separately hashed database per user; agent and global scopes are configurable. Wiki output is likewise separated by scope. Writes to shared Hermes `USER.md`/`MEMORY.md` files are refused from user- or agent-scoped stores. Startup migrations are additive and recorded in `schema_migrations`. FTS tables are versioned and count-checked against their sources.

Episode retention removes FTS rows and episode links and clears trace source references, preventing dangling provenance.

## Export safety

Sensitive facts and matching artifacts are omitted recursively from wiki and portable JSON exports by default. This can be changed explicitly with `export_redact_sensitive: false`. Portable imports reconstruct memory state, evidence, and remapped relationships; verified SQLite backups preserve the complete, unredacted operational database.

Wiki pages are written to temporary files, flushed, and atomically replaced. Dynamic Markdown content is escaped. A manifest records generated paths. Later exports prune only stale files in that manifest, so manually created notes—even inside `topics/` or `sessions/`—remain untouched. Export to a filesystem root or directly to `HERMES_HOME` is rejected.

## Privacy and isolation

Default operation is entirely local. Sensitive candidates follow `deny`, `ask`, or `allow`; `ask` creates a durable approval item. Credentials are always denied unless `allow_credential_memory: true` is explicitly enabled. Optional SQLCipher encryption fails closed when unavailable. LLM and embedding clients are disabled until their own endpoint is explicitly configured, and repeated failures open a cooldown circuit. Even after an endpoint is configured, sensitive text is withheld unless `allow_sensitive_model_processing: true` is explicitly enabled; credential processing also requires `allow_credential_memory: true`.

Hermes initializes providers with an `agent_context`. Only the primary context can write. Cron, flush, and subagent instances can read recall context without letting system prompts or tool output become user memory. Message extraction accepts only user/human and assistant/AI roles; system, developer, and tool roles are ignored.

## Recovery

If Hermes is stopped normally, accepted and spooled writes drain. After an abnormal process kill, SQLite WAL recovery and durable-operation replay recover committed work. Retries use bounded exponential backoff; poison tasks move to a dead-letter state instead of looping forever. The `doctor` action checks integrity, FTS counts, dangling links, active/dead-letter queue depth, and size. The native `hermes consolidating_local` CLI supports verified backup/restore, redacted JSON export/import, maintenance, and an explicit confirmed `retry-failed` recovery command.
