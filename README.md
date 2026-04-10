IT IS NOT WORKING, BETTER WITHOUT THAN WITH THIS. 







# Hermes Consolidating Local Memory

`consolidating_local` is a local-first memory provider for Hermes.

It adds a durable SQLite-backed memory layer on top of Hermes' built-in bounded memory, with consolidation, contradiction handling, topic rebuilding, spaced review, and optional markdown export.

This repository contains the provider itself, its storage layer, the consolidation pipeline, and the supporting documentation.

## What This Plugin Adds

Hermes already has:

- bounded prompt memory in `USER.md` and `MEMORY.md`
- full session history

This provider adds:

- short-lived raw turn buffers for consolidation
- durable facts with salience, confidence, and provenance
- preferences and policies as first-class memory objects
- contradiction-aware updates for exclusive subjects
- topic summaries rebuilt from active facts
- session, handoff, and derived summaries
- spaced review and salience decay
- optional compiled wiki export

The goal is not to replace Hermes memory. The goal is to give Hermes a deeper local memory system that can distill, organize, and retrieve information over time.

## How It Works

The provider integrates into Hermes through five hooks:

- `on_turn_start`
- `on_session_end`
- `on_pre_compress`
- `on_memory_write`
- `on_delegation`

At runtime, the provider:

1. captures turns, sessions, and delegation outcomes
2. writes structured memory into SQLite
3. processes background tasks through a FIFO worker thread
4. extracts and normalizes candidate facts
5. deduplicates and resolves exclusive-state conflicts
6. rebuilds topics and summaries
7. applies decay and review scheduling
8. optionally syncs current winners back into `USER.md` and `MEMORY.md`
9. optionally exports a compiled markdown wiki

The worker executes tasks sequentially, which keeps hook behavior non-blocking while preserving a predictable consolidation flow.

## Storage Model

The canonical store is a single SQLite database, usually:

```text
$HERMES_HOME/consolidating_memory.db
```

Core tables include:

- `facts`
- `topics`
- `topic_membership`
- `episodes`
- `memory_sessions`
- `memory_traces`
- `memory_journals`
- `memory_summaries`
- `memory_preferences`
- `memory_policies`
- `memory_history`
- `memory_links`
- `contradictions`
- `provider_state`
- `consolidation_runs`

The store also creates FTS5 virtual tables for fast recall across facts, topics, episodes, summaries, journals, preferences, policies, and traces.

Important rule:

- SQLite is the source of truth
- generated markdown is derived output

## Memory Layers

The plugin is easiest to understand as three layers:

1. Raw capture
   `episodes` and `memory_traces`
2. Distilled memory
   `facts`, `topics`, `summaries`, `preferences`, `policies`, and `contradictions`
3. Derived output
   compiled markdown pages exported from SQLite

That layering is what allows the provider to keep recent conversational detail without treating every raw turn as durable memory.

## Retrieval and Extraction

Extraction backends:

- `heuristic`
- `hybrid`
- `llm`

Retrieval backends:

- `fts`
- `hybrid`

In practice:

- FTS gives fast local keyword recall
- hybrid retrieval can rerank candidates with embeddings
- hybrid or LLM extraction can turn noisy session text into normalized durable facts

If local LLM or embedding endpoints are unavailable, the provider can fall back to simpler local behavior instead of failing completely.

## Tool Surface

The provider exposes one tool:

```text
consolidating_memory
```

Supported actions:

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
- `review`
- `decay`
- `export`

These actions cover both operator workflows and direct memory manipulation, including forced consolidation, policy capture, history inspection, and spaced-review advancement.

## Install Into Hermes

Copy the plugin into your Hermes checkout:

```text
plugins/memory/consolidating_local/
```

Then enable it in Hermes:

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
    reconsolidation_window_hours: 6
    review_intervals_days: 1,3,7,14,30
    decay_min_salience: 0.15
    builtin_snapshot_sync_enabled: false
    wiki_export_enabled: false
    wiki_export_dir: $HERMES_HOME/consolidating_memory_wiki
    wiki_export_on_consolidate: true
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

## Hermes Memory Control

For an easier UI to inspect and manage the memory store, use:

- [Hermes Memory Control](https://github.com/b7216309-jpg/hermes-memory-control)

That companion app is useful when you want to:

- browse facts, topics, preferences, policies, and contradictions without querying SQLite by hand
- inspect the current provider state and consolidation results
- edit plugin configuration from a UI instead of editing YAML manually
- browse the compiled wiki export
- visualize the memory graph and spot noisy clusters or bad links

Integration boundary:

- this plugin owns the memory logic and the SQLite schema
- Hermes Memory Control is the operator UI over that same store
- the app does not replace consolidation or become a second memory backend

See [docs/HERMES_MEMORY_CONTROL.md](docs/HERMES_MEMORY_CONTROL.md) for the relationship between the provider and the desktop app.

## Repository Layout

- `plugins/memory/consolidating_local/`
  Provider implementation, plugin manifest, storage layer, consolidation engine, LLM wrapper, and wiki export
- `docs/PLUGIN_DEEP_DIVE.md`
  Internal architecture, lifecycle, schema, and flow details
- `docs/HERMES_MEMORY_CONTROL.md`
  Integration notes for the companion UI application

## Documentation

- Plugin quick reference: [plugins/memory/consolidating_local/README.md](plugins/memory/consolidating_local/README.md)
- Internal deep dive: [docs/PLUGIN_DEEP_DIVE.md](docs/PLUGIN_DEEP_DIVE.md)
- Companion app integration: [docs/HERMES_MEMORY_CONTROL.md](docs/HERMES_MEMORY_CONTROL.md)

## Core Principle

- SQLite is the source of truth
- generated markdown is derived output
