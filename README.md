# Hermes Consolidating Local Memory

This repository contains the `consolidating_local` memory provider for Hermes.

It adds a local SQLite-backed memory layer with:

- durable facts and summaries
- contradiction-aware updates
- preferences and policies
- salience decay and spaced review
- optional wiki export

## Repository Layout

- `plugins/memory/consolidating_local/`
  Provider implementation, storage layer, consolidation logic, and plugin manifest
- `docs/PLUGIN_DEEP_DIVE.md`
  Internal architecture and schema notes
- `docs/HERMES_MEMORY_CONTROL.md`
  Integration notes for the companion desktop app

## Quick Start

1. Install this plugin into your Hermes checkout.
2. Enable `consolidating_local` in Hermes `config.yaml`.
3. Run Hermes so the provider can initialize its SQLite store.
