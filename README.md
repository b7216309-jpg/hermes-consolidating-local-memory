# Hermes Consolidating Local Memory

`consolidating_local` is a local-first memory provider plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

It adds a longer-lived memory loop on top of Hermes' built-in `MEMORY.md` and `USER.md`:

- episodic turn storage in SQLite
- durable fact extraction
- topic summaries for fast prompt injection
- contradiction tracking when newer facts replace stale assumptions
- optional local-LLM-assisted extraction through an OpenAI-compatible endpoint

The plugin is designed to stay practical for self-hosted setups. It works in pure heuristic mode, but it can also reuse your existing Hermes local model config in `hybrid` mode for better extraction quality.

## Features

- Local SQLite storage with no required cloud dependency
- Background consolidation gated by time and session count
- Atomic fact extraction for:
  - user preferences
  - environment details
  - workflow rules
  - common project signals
- Contradiction-aware updates for exclusive facts such as:
  - response style
  - shell
  - OS
  - SSH port
  - Docker sudo behavior
- Prompt prefetch with topic summaries, direct fact matches, and recent assumption changes
- Optional OpenAI-compatible local model support
- Safe fallback to heuristics if the local model call or JSON parse fails

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
    prune_after_days: 90
    llm_timeout_seconds: 45
    llm_max_input_chars: 4000
    llm_model: ""      # blank = use Hermes model.default
    llm_base_url: ""   # blank = use Hermes model.base_url
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

The plugin can call an OpenAI-compatible local endpoint for extraction.

It looks for model settings in this order:

1. `plugins.consolidating-local-memory.llm_model` / `llm_base_url`
2. Hermes `model.default` / `model.base_url`

If the endpoint requires auth, set:

```bash
export CONSOLIDATING_MEMORY_LLM_API_KEY=...
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

## What Gets Stored

The SQLite store keeps four main record types:

- `episodes`
  Raw turn-level observations

- `facts`
  Durable extracted memories

- `topics`
  Summaries assembled from high-value facts

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
