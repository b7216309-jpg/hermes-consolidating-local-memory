# consolidating_local

`consolidating_local` is a Hermes memory provider built around a layered local-memory model.

It is meant to complement Hermes, not replace it.

Hermes keeps:

- bounded built-in prompt memory
- full session history

This provider keeps:

- short-lived raw turn buffers for consolidation
- durable distilled memory
- contradiction-aware state updates
- provenance and history
- an optional compiled markdown wiki mirror

## Mental Model

The system has three layers:

1. Raw capture

- `episodes`
- `traces`

2. Distilled memory

- `facts`
- `summaries`
- `preferences`
- `policies`
- `topics`
- `contradictions`

3. Derived output

- compiled markdown wiki pages exported from SQLite

The important rule is:

- SQLite is the source of truth
- markdown is generated output

## What It Does

Hook behavior:

- `prefetch(query)`
  Recalls relevant memory before a turn

- `sync_turn(user, assistant)`
  Stores a bounded episode buffer and a trace row

- `on_turn_start(...)`
  Checks the consolidation gate

- `on_session_end(...)`
  Extracts durable memory, refreshes session summaries, and requests consolidation

- `on_pre_compress(...)`
  Salvages durable signal and writes a handoff summary

- `on_memory_write(...)`
  Mirrors Hermes memory writes into the provider

- `on_delegation(...)`
  Stores delegation outcomes as workflow memory

## Stored Objects

- `episodes`
  Short-lived raw turn buffers

- `facts`
  Durable extracted memories

- `topics`
  Topic summaries rebuilt from active facts

- `memory_sessions`
  Session metadata

- `memory_traces`
  Lightweight turn traces

- `memory_journals`
  Narrative notes

- `memory_summaries`
  Session, handoff, and derived summaries

- `memory_preferences`
  Stable user preferences

- `memory_policies`
  Workflow rules and operating constraints

- `memory_history`
  Append-only change log

- `memory_links`
  Typed provenance links

- `contradictions`
  Resolved assumption changes

## Tool

The provider exposes one tool:

```text
consolidating_memory
```

Actions:

- `search`
- `remember`
- `forget`
- `recent`
- `contradictions`
- `status`
- `consolidate`
- `journal`
- `distill`
- `history`
- `policy`
- `decay`
- `export`

Search scopes:

- `all`
- `facts`
- `topics`
- `episodes`
- `summaries`
- `journals`
- `preferences`
- `policies`

## Compiled Wiki Export

When enabled, the provider can export a markdown mirror of the current store.

Generated files include:

- `index.md`
- `topics/*.md`
- `sessions/*.md`
- `preferences/index.md`
- `policies/index.md`
- `contradictions/index.md`

This export is:

- deterministic
- safe to rerun
- suitable for Obsidian-style browsing
- cleaned up on rerun when old generated pages no longer belong

## Companion Desktop App

This provider also has a companion desktop control panel:

- [Hermes Memory Control](https://github.com/b7216309-jpg/hermes-memory-control)

The app is useful when you want to:

- inspect facts, topics, preferences, and contradictions without querying SQLite by hand
- edit the plugin config from a UI instead of editing YAML manually
- browse the compiled wiki export in-app
- visualize the memory graph and spot bad clusters or noisy facts

Integration boundary:

- this provider writes and reads the canonical SQLite store
- the desktop app reads that same store and edits related operator-facing files
- the app does not replace consolidation logic or become a second source of truth

See [../../../docs/HERMES_MEMORY_CONTROL.md](../../../docs/HERMES_MEMORY_CONTROL.md) for the end-to-end workflow.

## Install Into Hermes

Copy this folder into your Hermes checkout:

```text
plugins/memory/consolidating_local/
```

Then configure Hermes:

```yaml
memory:
  provider: consolidating_local

plugins:
  consolidating-local-memory:
    db_path: $HERMES_HOME/consolidating_memory.db
    min_hours: 24
    min_sessions: 5
    scan_cooldown_seconds: 600
    prefetch_limit: 8
    max_topic_facts: 5
    topic_summary_chars: 650
    session_summary_chars: 900
    prune_after_days: 90
    episode_body_retention_hours: 24
    decay_half_life_days: 90
    decay_min_salience: 0.15
    wiki_export_enabled: false
    wiki_export_dir: $HERMES_HOME/consolidating_memory_wiki
    wiki_export_on_consolidate: true
    wiki_export_session_limit: 50
    wiki_export_topic_limit: 100
    extractor_backend: hybrid
    retrieval_backend: fts
    llm_model: ""
    llm_base_url: ""
    llm_timeout_seconds: 45
    llm_max_input_chars: 4000
    embedding_model: ""
    embedding_base_url: ""
    embedding_timeout_seconds: 20
    embedding_candidate_limit: 16
```

## Backends

Extraction:

- `heuristic`
- `hybrid`
- `llm`

Retrieval:

- `fts`
- `hybrid`

If a local LLM or embedding endpoint is unavailable, the provider falls back safely to heuristic extraction or plain FTS.

## Current Strengths

- Local-first operation
- Exclusive-state contradiction handling
- Session-aware summaries and handoff notes
- Stable preference and policy memory
- Append-only history and typed provenance
- Optional compiled wiki mirror

## Development

Syntax check:

```bash
python -m compileall plugins/memory/consolidating_local
```

Run tests:

```bash
python -m unittest discover -s tests -v
```
