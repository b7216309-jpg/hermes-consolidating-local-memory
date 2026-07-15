# consolidating_local 3.3

This directory is the installable Hermes memory-provider bundle. Install it as `$HERMES_HOME/plugins/consolidating_local/`; the repository-level [`install.py`](../../../install.py) performs an atomic update.

## Hermes integration

The provider implements the current lifecycle:

- `prefetch` returns fast FTS recall and consumes a bounded, expiring per-session cache.
- `queue_prefetch` optionally performs embedding reranking off the request path.
- `sync_turn(..., messages=...)` queues the completed turn, captures its episode, and optionally runs configured LLM extraction.
- `on_session_switch` rotates session state for new, resumed, branched, rewound, and compressed sessions.
- `on_memory_write(..., metadata=...)` mirrors committed built-in-memory writes with provenance and replacement information.
- `on_session_end` and `on_pre_compress` distill summaries before context disappears.
- `shutdown` stops accepting work, drains accepted FIFO tasks, and only then closes SQLite.
- `backup_paths` declares configured storage outside `HERMES_HOME`.

Hermes cron, flush, and subagent contexts may recall data but cannot mutate the store.

Shutdown waits up to `shutdown_timeout_seconds` (10 seconds by default). If an optional model call is still in flight, queued writes are moved to the durable SQLite spool and replayed when the worker recovers; Hermes is not held for the model timeout. The worker closes its database connection itself after the in-flight call returns.

## Data layers

1. Raw capture: `episodes`, `memory_traces`, bounded `working_memory`
2. Durable memory: evidence-backed facts, topics, summaries, preferences, policies, and journals
3. Brain-inspired systems: procedures, prospective intentions, autobiographical events, and weighted associations
4. Audit and consent: evidence observations, history, links, contradictions, sessions, and approvals
5. Operations: durable pending work, maintenance leases, migrations, backups, repair, and redacted exports

SQLite is canonical. Exported files are rebuildable.

Facts distinguish semantic state from time: `temporal_kind` identifies atemporal/current/event/scheduled/temporary meaning; `event_at` records when an event occurred or is planned; `valid_from` and `valid_until` bound state validity; precision, source timezone, and confidence preserve what was actually known. Creation/update/observation timestamps remain separate. Scheduled facts can age out of current-state search without deleting their linked timeline record.

## Tool

The `consolidating_memory` tool supports:

Alongside the original actions, v2 adds `explain`, `working`, `procedure`, `intention`, `timeline`, `approval`, `associate`, `merge`, `split`, `pin`, `doctor`, `maintain`, `backup`, and `export_json`.

Version 3.3.1 accepts automatic turn capture only from Hermes' authoritative inbound gateway hook
or a direct human CLI turn. Synthetic gateway work and background review harnesses are excluded
from recall warming, episodes, traces, working memory, fact extraction, and session summaries.
Hermes 0.18.2 discovers the provider and general hooks separately, so `install.py` also enables the
same package as a lifecycle observer; both module namespaces share one bounded origin ledger.

## Extraction and retrieval

- Automatic extraction is disabled unless both `llm_model` and `llm_base_url` are configured.
- Automatic extraction is LLM-only. There is no rule-based extractor, hybrid extractor, or fallback.
- `llm_disable_thinking: true` sends `chat_template_kwargs.enable_thinking=false` to compatible OpenAI-style chat endpoints and rejects reasoning-only responses.
- Extraction receives the current local ISO time, Unix time, and Hermes IANA timezone. Relative dates are resolved against that reference and missing time precision must not be invented.
- Explicit Hermes memory-tool writes remain immediate and do not require an extraction model.
- `retrieval_backend: fts` is the default.
- `retrieval_backend: hybrid` reranks FTS candidates using an explicitly configured embedding endpoint.

Both a model and base URL are required to enable either remote-capable client. Hermes' normal chat endpoint is never reused implicitly.

Recall begins with the localized current time and a timestamp contract. Memory lines distinguish event/validity time from recorded/updated age, and passed schedules are labeled as unconfirmed rather than completed.

## Important defaults

```yaml
plugins:
  consolidating-local-memory:
    db_path: $HERMES_HOME/consolidating_memory.db
    memory_scope: user
    sensitive_memory: ask
    allow_sensitive_model_processing: false
    conflict_policy: evidence
    queue_max_size: 256
    queue_max_attempts: 5
    shutdown_timeout_seconds: 10
    max_database_mb: 512
    retrieval_backend: fts
    llm_disable_thinking: false
    builtin_snapshot_sync_enabled: false
    wiki_export_enabled: false
```

`memory_scope: user` creates a different hashed database and wiki subdirectory for every gateway user. A local CLI invocation without a user identity retains the configured legacy database. Shared Hermes `USER.md`/`MEMORY.md` snapshot writes are refused for user- and agent-scoped stores. Set `database_encryption: true` only after installing the `encryption` extra and exporting `CONSOLIDATING_MEMORY_DB_KEY`; startup fails rather than silently opening an unencrypted database when either requirement is missing.

Credentials are rejected even when other sensitive memories use the approval inbox. Only set `allow_credential_memory: true` for an intentional exception; the credential then follows `sensitive_memory`, and database encryption is strongly recommended.

Configured LLM and embedding endpoints never receive text classified as sensitive by default. `allow_sensitive_model_processing: true` is a separate explicit opt-in; credentials still require `allow_credential_memory: true` as well. When model processing is blocked, the provider still captures the redacted episode, accepts explicit memory-tool writes, and uses local FTS recall.

Onboarding adds a second remote-processing boundary: every accepted profile item has `metadata.local_only=true`. The item remains normally visible to local FTS and Hermes context, but any hybrid candidate set containing it skips the remote embedding client. This does not hide recalled context from Hermes' active chat model.

For offline operations, stop Hermes and run:

```console
hermes consolidating_local onboard
hermes consolidating_local doctor
hermes consolidating_local backup /path/to/backup.db
hermes consolidating_local export /path/to/memory.json
hermes consolidating_local retry-failed --confirm
```

`onboard` performs a guided 17-question interview, renders the complete proposed profile, and requires explicit approval before writing. It classifies answers as facts, preferences, policies, procedures, or intentions. Credential-like answers are discarded. Identical reruns are no-ops and do not duplicate evidence/history. Use `onboard --template PATH` for a protected blank JSON answer file, `--answers PATH --preview-only` to inspect it, and `--answers PATH --yes` only after review. Add `--skip-sensitive` to omit health, financial, identity, and precise-location entries.

For `memory_scope: user`, pass `--scope-platform PLATFORM --scope-user-id ID` before the subcommand to derive the same hashed database path as a gateway conversation. Omit them for local/CLI memory. Agent scope also accepts `--scope-agent-identity`. Scope flags work with onboarding, doctor, backup, restore, export, import, retry, and maintenance. Raw identity values are used only for hashing and are not placed in database filenames.

Durable work uses exponential backoff and moves to a visible dead-letter queue after `queue_max_attempts`. `doctor` reports this as degraded; inspect its error details before using the explicit retry command because a failed task may already have completed part of its work. Portable JSON imports reconstruct durable memory state and relationships. SQLite backups remain complete and unredacted.

Pass `--db /path/to/scoped.db` before the subcommand when operating on a per-user or per-agent database. For an encrypted database, set `CONSOLIDATING_MEMORY_DB_KEY` before every admin command. Restore is integrity-checked into a temporary database and atomically replaces the destination only after verification. Stop Hermes before repair, restore, import, or maintenance. From a source checkout, the equivalent entry point is `python -m plugins.memory.consolidating_local.admin --db PATH ...`.

SQLite backups are consistent but intentionally unredacted. Agent tool calls default them under `HERMES_HOME`; writing one elsewhere requires `confirm=true`. Portable JSON and wiki exports remain the sharing-oriented, redacted formats.

See the [root README](../../../README.md) for installation and the [deep dive](../../../docs/PLUGIN_DEEP_DIVE.md) for schema and operational details.
