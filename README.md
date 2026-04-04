# Hermes Consolidating Local Memory

`consolidating_local` is a local-first memory provider for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

It is built for a simple goal: keep long-running agent memory useful without turning the system into a fragile black box.

The provider keeps SQLite as the source of truth, stores raw turns only long enough to distill them, maintains contradiction-aware durable memory, and can now export a compiled markdown wiki mirror for Obsidian-style browsing.

## What This Plugin Tries To Do

Hermes already gives you:

- bounded always-on memory through `MEMORY.md` and `USER.md`
- session storage and search

This plugin adds a second layer on top:

- short-lived local buffers for raw turns
- durable facts extracted from conversation
- stable preferences and workflow policies
- per-session traces, journals, handoff notes, and summaries
- contradiction tracking when exclusive state changes
- provenance links and append-only history
- topic summaries for fast recall
- an optional compiled markdown mirror generated from the memory store

The design principle is:

- Hermes owns transcript history
- this plugin owns distilled long-term memory
- the wiki mirror is generated output, not the canonical database

## Current Status

The current version is `0.3.0`.

Implemented today:

- layered SQLite memory model
- session-aware consolidation
- contradiction-aware exclusive facts
- optional OpenAI-compatible extraction
- optional OpenAI-compatible embedding reranking
- compiled wiki export

Planned next:

- memory health checks and linting
- report generation into the wiki mirror
- a separate external-source knowledge lane

## Architecture

The memory flow is:

1. Capture

- `sync_turn` stores bounded raw episode buffers
- turn traces are written immediately

2. Distill

- session end extracts durable facts
- preferences and policies are promoted into first-class memory
- session and handoff summaries are refreshed

3. Consolidate

- topic summaries are rebuilt from active facts
- contradictions and supersession chains are recorded
- low-value memory decays over time
- old raw episode buffers are pruned

4. Recall

- prefetch injects the most useful summaries and facts back into the prompt
- search uses FTS by default and can rerank with embeddings

5. Export

- the current store can be rendered as a compiled markdown wiki

## Core Features

- Local SQLite storage with no required cloud dependency
- Layered memory objects for sessions, traces, journals, summaries, preferences, policies, facts, topics, history, and links
- Contradiction-aware updates for exclusive facts such as response style, shell, current location, favorites, diet, allergies, and similar profile state
- Append-only history for important memory mutations
- Typed provenance links such as `captured_in`, `derived_from_episode`, `summarizes`, `supports`, `contradicts`, `supersedes`, and `recalls`
- Salience plus decay so low-value memories fade without wiping durable ones
- Optional local LLM extraction through an OpenAI-compatible endpoint
- Optional hybrid retrieval through an OpenAI-compatible `/embeddings` endpoint
- Deterministic compiled wiki export for browsing outside the agent

## Compiled Wiki Mirror

When enabled, the plugin can render a markdown mirror of the memory store.

Suggested export tree:

```text
consolidating_memory_wiki/
  index.md
  topics/
    user-preferences.md
    project-delivery.md
  sessions/
    test-session-<hash>.md
  preferences/
    index.md
  policies/
    index.md
  contradictions/
    index.md
```

The mirror is:

- generated from SQLite
- safe to regenerate
- intended for browsing in Obsidian or any markdown viewer
- not the authoritative store

The exporter is deterministic and prunes stale generated pages on rerun.

## Tool Interface

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
- `decay`
- `export`

Supported search scopes:

- `all`
- `facts`
- `topics`
- `episodes`
- `summaries`
- `journals`
- `preferences`
- `policies`

## Data Model

Main stored objects:

- `episodes`
  Short-lived turn buffers used only for consolidation and recovery windows

- `facts`
  Durable extracted memory

- `topics`
  Summaries built from the strongest active facts in each topic bucket

- `memory_sessions`
  Session metadata and summary state

- `memory_traces`
  Lightweight turn-by-turn traces

- `memory_journals`
  Narrative notes

- `memory_summaries`
  Session summaries, handoff summaries, and derived summaries

- `memory_preferences`
  Durable user preferences and stable profile memory

- `memory_policies`
  Workflow rules and operating constraints

- `memory_history`
  Append-only mutation log

- `memory_links`
  Typed provenance and backlink graph

- `contradictions`
  Resolved assumption changes

## Installation

Copy the plugin into your Hermes checkout:

```text
plugins/memory/consolidating_local/
```

Then activate it in your Hermes profile config:

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

Extraction backends:

- `heuristic`
  Rule-based extraction only

- `hybrid`
  Heuristics plus local LLM extraction when available

- `llm`
  Local LLM extraction only

Retrieval backends:

- `fts`
  SQLite full-text search only

- `hybrid`
  FTS candidate generation plus embedding reranking

## Local Model Support

The plugin can call OpenAI-compatible local endpoints for both extraction and embeddings.

Resolution order for model settings:

1. plugin config
2. Hermes `model.default` and `model.base_url`

If your extraction endpoint needs auth:

```bash
export CONSOLIDATING_MEMORY_LLM_API_KEY=...
```

If your embedding endpoint uses separate credentials:

```bash
export CONSOLIDATING_MEMORY_EMBEDDING_API_KEY=...
```

## Prompt Recall Strategy

Prefetch is intentionally structured.

The current order is:

1. relevant summaries
2. active preferences and workflow rules
3. direct fact matches
4. recent journal notes
5. changed assumptions

This keeps recall compact and makes contradiction changes visible.

## Repository Layout

```text
docs/
  V3_COMPILED_WIKI_MIRROR.md
plugins/memory/consolidating_local/
  __init__.py
  consolidator.py
  llm_client.py
  plugin.yaml
  README.md
  store.py
  wiki_export.py
tests/
  test_consolidating_local.py
```

## Development

Syntax check:

```bash
python -m compileall plugins/memory/consolidating_local
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

## Privacy

The plugin is local-first, but it stores memory.

Do not publish your live runtime data such as:

- `~/.hermes/config.yaml`
- `~/.hermes/.env`
- `~/.hermes/consolidating_memory.db`
- exported compiled wiki directories containing private memories

The code is safe to publish.
Your live memory data may not be.

## Design Direction

The current direction is inspired by a useful pattern:

- collect raw data
- compile it into a more navigable form
- let the agent query and improve the compiled layer over time

For this plugin, that means:

- SQLite stays canonical
- markdown stays derived
- memory remains constrained and auditable

The detailed next-step design lives in [docs/V3_COMPILED_WIKI_MIRROR.md](docs/V3_COMPILED_WIKI_MIRROR.md).

## License

MIT
