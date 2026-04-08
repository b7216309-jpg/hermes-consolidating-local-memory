# Hermes Memory Control Integration

This document explains how the main plugin repo and the companion desktop app fit together.

Related repos:

- Plugin repo: [hermes-consolidating-local-memory](https://github.com/b7216309-jpg/hermes-consolidating-local-memory)
- Desktop control panel: [hermes-memory-control](https://github.com/b7216309-jpg/hermes-memory-control)

## Purpose

The two repos are meant to feel like one system with a clean split of responsibilities:

- this repo owns the memory provider, SQLite schema, consolidation logic, and wiki export
- Hermes Memory Control owns the operator UI for browsing, editing, and visualizing the same memory store

The app is not a separate memory backend. It is a control surface for the provider defined here.

## Shared Integration Points

The repos meet at three concrete files or directories inside a Hermes home:

1. `config.yaml`
   The app reads and writes the `plugins.consolidating-local-memory` block.

2. `consolidating_memory.db`
   The provider writes the SQLite database and the app inspects or edits its tables.

3. `consolidating_memory_wiki/`
   The provider generates the compiled wiki mirror and the app renders it for browsing.

Important boundary:

- SQLite is the source of truth
- the wiki export is derived output
- the app should be treated as an operator tool, not as a second consolidation engine

## Practical Workflow

Typical setup:

1. Install the plugin from this repo into your Hermes checkout.
2. Enable `consolidating_local` in Hermes `config.yaml`.
3. Run Hermes until the provider creates `consolidating_memory.db`.
4. Install and launch Hermes Memory Control on Windows.
5. Point the app at your Hermes home directory, for example `\\wsl$\\Ubuntu\\home\\user\\.hermes`.

Typical operating loop:

1. Hermes conversations feed the provider hooks.
2. The provider consolidates memory into SQLite.
3. The desktop app reads that database for inspection and visualization.
4. If needed, the app edits config or individual rows.
5. Hermes keeps using the same database and updated config on later runs.

## What Each Side Owns

Provider-side responsibilities in this repo:

- lifecycle hooks such as `on_turn_start`, `on_session_end`, `on_pre_compress`, `on_memory_write`, and `on_delegation`
- extraction, deduplication, contradiction handling, salience decay, and spaced review
- topic rebuilding, summaries, provenance, and history
- snapshot sync into `USER.md` and `MEMORY.md`
- compiled wiki export
Desktop app responsibilities:

- dashboard for counts and last consolidation state
- table explorers for facts, topics, sessions, preferences, policies, and contradictions
- inline CRUD-style edits for selected entities
- config editor for plugin settings including LLM and embedding options
- wiki viewer
- 3D memory graph

## Why This Split Works

This split keeps the hard parts where they belong:

- the provider stays local-first and scriptable
- the app stays lightweight and focused on observability and operator control
- both repos can evolve independently as long as the config contract and SQLite schema stay compatible

It also makes debugging easier:

- if the memory is wrong, inspect provider logic here
- if the data is right but hard to inspect, use the app
- if the app shows unexpected data, check whether the provider wrote it or whether it was manually edited later

## Documentation Pointers

- Root overview: [../README.md](../README.md)
- Plugin quick reference: [../plugins/memory/consolidating_local/README.md](../plugins/memory/consolidating_local/README.md)
- Internal architecture: [PLUGIN_DEEP_DIVE.md](PLUGIN_DEEP_DIVE.md)

## Recommended Language

When describing the ecosystem in docs or releases, use wording like:

- "Hermes consolidating local memory plugin"
- "Hermes Memory Control companion desktop app"
- "SQLite is the source of truth; the desktop app is the operator UI"
