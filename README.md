# Hermes Consolidating Local Memory

`consolidating_local` is a local-first memory provider plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

It adds a layered long-term memory loop on top of Hermes' built-in `MEMORY.md`, `USER.md`, and session history:

- bounded episode buffers in SQLite for short-lived raw turn capture
- durable fact extraction with contradiction-aware exclusive state updates
- per-session traces, journals, and summaries
- durable preferences and workflow policies
- append-only history plus typed provenance links
- topic summaries for fast prompt injection
- optional OpenAI-compatible local extraction and embedding backends

The plugin is designed for self-hosted setups. It works in pure heuristic mode, can reuse your Hermes local model config in `hybrid` mode for better extraction quality, and keeps Hermes session storage as the source of truth for full transcripts.

## Features

- Local SQLite storage with no required cloud dependency
- Layered memory objects for sessions, traces, journals, summaries, preferences, policies, facts, topics, history, and links
- Background consolidation gated by time and session count
- Atomic fact extraction for:
  - user preferences
  - favorites, likes, and dislikes
  - personal profile details
  - notable life events
  - environment details
  - workflow rules
  - common project signals
- Contradiction-aware updates for exclusive facts such as:
  - favorite things
  - like/dislike flips
  - current location or hometown
  - diet and allergy facts
  - response style
  - shell
  - OS
  - SSH port
  - Docker sudo behavior
- Session-end and pre-compression distillation into handoff and session summaries
- Prompt prefetch ordered as: relevant summaries, active preferences and workflow rules, direct fact matches, recent journal notes, changed assumptions
- Append-only history plus provenance links such as `captured_in`, `summarizes`, `contradicts`, `supersedes`, and `recalls`
- Salience tracking with configurable decay
- Optional hybrid retrieval that reranks FTS candidates with local embeddings
- Safe fallback to heuristics or plain FTS when local model calls are unavailable

## Repository Layout

```text
plugins/memory/consolidating_local/
  __init__.py
  consolidator.py
  llm_client.py
  plugin.yaml
  README.md
  store.py
```

This layout matches the location Hermes expects for built-in memory providers, so you can copy the folder directly into a Hermes checkout.

## Installation

Copy the plugin into your Hermes repo:

```bash
cp -a plugins/memory/consolidating_local ~/.hermes/hermes-agent/plugins/memory/
```

Then enable it in `~/.hermes/config.yaml`:

```yaml
memory:
  provider: consolidating_local

plugins:
  consolidating-local-memory:
    extractor_backend: hybrid
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
    llm_timeout_seconds: 45
    llm_max_input_chars: 4000
    retrieval_backend: fts  # or hybrid
    llm_model: ""      # blank = use Hermes model.default
    llm_base_url: ""   # blank = use Hermes model.base_url
    embedding_model: ""      # blank = use llm_model/model.default
    embedding_base_url: ""   # blank = use llm_base_url/model.base_url
    embedding_timeout_seconds: 20
    embedding_candidate_limit: 16
```

## Extraction Backends

Three extraction modes are supported:

- `heuristic`
  Uses only the built-in rule-based extractor.

- `hybrid`
  Uses heuristics plus a local model if available. This is the recommended default.

- `llm`
  Uses only the local model path. This is less robust and usually not necessary.

## Local Model Support

The plugin can call OpenAI-compatible local endpoints for extraction and optional hybrid retrieval.

It looks for model settings in this order:

1. `plugins.consolidating-local-memory.llm_model` / `llm_base_url`
2. Hermes `model.default` / `model.base_url`

If the endpoint requires auth, set:

```bash
export CONSOLIDATING_MEMORY_LLM_API_KEY=...
```

If embeddings use separate credentials, set:

```bash
export CONSOLIDATING_MEMORY_EMBEDDING_API_KEY=...
```

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

## What Gets Stored

The SQLite store keeps layered memory objects:

- `episodes`
  Short-lived turn buffers used for consolidation, not long-term transcript storage

- `facts`
  Durable extracted memories

- `topics`
  Summaries assembled from high-value facts

- `memory_sessions`
  Session metadata and summary pointers

- `memory_traces`
  Lightweight turn-by-turn traces tied back to the session and source episode

- `memory_journals`
  Narrative notes and operator-authored observations

- `memory_summaries`
  Session, handoff, and derived summaries

- `memory_preferences`
  Durable user preferences and stable profile-style memory

- `memory_policies`
  High-salience workflow rules and operating constraints

- `memory_history`
  Append-only change log for facts, summaries, preferences, policies, and contradiction updates

- `memory_links`
  Typed provenance links between sessions, episodes, facts, summaries, and recalls

- `contradictions`
  Resolution logs for stale or replaced assumptions

## Development

Quick syntax check:

```bash
python -m py_compile plugins/memory/consolidating_local/*.py
```

Hermes-side smoke testing can be done by importing the provider from a real Hermes checkout and forcing a consolidation run.

## License

MIT
