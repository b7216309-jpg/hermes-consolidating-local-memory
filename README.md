# Hermes Consolidating Local Memory

`consolidating_local` is a local-first memory provider for [Hermes Agent](https://github.com/NousResearch/hermes-agent), plus a companion benchmark suite that compares Hermes built-in bounded memory against the addon under controlled conditions.

The current plugin version is `0.3.0`.

## What This Repo Contains

- a SQLite-backed Hermes memory provider at `plugins/memory/consolidating_local/`
- a comparative benchmark at `bench_compare/`
- a PDF report generator for benchmark results
- a runbook for rerunning the benchmark safely from a fresh session

The design goal is simple:

- keep long-running agent memory useful
- keep the storage model inspectable
- keep prompt injection compact
- keep benchmark token spend low and measurable

## Provider Overview

Hermes already gives you:

- bounded always-on memory through `MEMORY.md` and `USER.md`
- transcript and session history

This plugin adds a second layer on top:

- short-lived raw episode buffers for consolidation
- durable facts extracted from turns
- stable preferences and workflow policies
- journals, traces, session summaries, and handoff summaries
- contradiction tracking and supersession chains for exclusive state
- append-only history and typed provenance links
- topic summaries and retrieval-aware prompt prefetch
- optional compiled markdown wiki export
- optional sync of current winners back into Hermes `USER.md` and `MEMORY.md`

The intended split is:

- Hermes owns transcript history and bounded built-in memory
- the plugin owns distilled long-term memory
- the wiki mirror is derived output, not the canonical store

## Current Capabilities

Implemented today:

- layered SQLite memory store
- background consolidation with session/hour gates
- contradiction-aware exclusive facts
- append-only mutation history
- typed provenance links
- salience and decay
- spaced review scheduling
- built-in snapshot sync back to `USER.md` and `MEMORY.md`
- retrieval modes for current-state, summary, workflow, history, and provenance queries
- optional OpenAI-compatible extraction
- optional OpenAI-compatible embedding reranking
- compiled markdown wiki export
- Python-accessible `get_context(...)` and `ConsolidatingLocalProvider` alias for direct integration

Important current behavior:

- broad subjectless `get_context()` queries now fall back to a merged current-memory snapshot instead of returning an empty block
- this fixed the earlier DIM-5 prefetch audit issue in the benchmark

## Architecture

The main flow is:

1. Capture

- `sync_turn` stores bounded raw episode buffers
- turn traces are recorded immediately

2. Distill

- session-end extraction promotes durable facts
- preferences and policies become first-class memory objects
- session and handoff summaries are refreshed

3. Consolidate

- topic summaries are rebuilt from active facts
- contradictions and supersession chains are recorded
- low-value memory decays over time
- raw episode bodies can be pruned aggressively after distill

4. Recall

- prefetch builds a compact recall block from facts, summaries, preferences, policies, journals, and provenance
- retrieval defaults to SQLite FTS and can optionally rerank with embeddings
- retrieval behavior changes by mode: current-state, summary, workflow, history, provenance, and review

5. Export

- the current store can be rendered as a compiled markdown wiki mirror

## Hooks and Prompt Behavior

The plugin currently registers these Hermes hooks:

- `on_turn_start`
- `on_session_end`
- `on_pre_compress`
- `on_memory_write`
- `on_delegation`

Prompt-facing behavior includes:

- `system_prompt_block()` for provider status and usage hints
- `prefetch(query, session_id=...)` for retrieval-driven recall blocks
- `get_context(session_id=..., query="")` for a direct current/provenance context block
- `queue_prefetch(...)` for async prefetch warmup

Built-in bounded memory sync is also supported:

- when `builtin_snapshot_sync_enabled` is true, the plugin keeps Hermes `USER.md` and `MEMORY.md` aligned with the plugin's current winners within their configured character budgets

## Tool Interface

The provider exposes one Hermes tool:

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

Supported search scopes:

- `all`
- `facts`
- `topics`
- `episodes`
- `summaries`
- `journals`
- `preferences`
- `policies`

The `status` action currently reports:

- object counts
- consolidation plan state
- recent contradictions
- extraction and retrieval backend info
- review status
- wiki export state
- built-in snapshot sync metadata
- effective config

## Data Model

Main stored objects:

- `episodes`
  Short-lived turn buffers used for extraction and recovery windows

- `facts`
  Durable extracted memory

- `topics`
  Summaries built from the strongest active facts in each topic bucket

- `memory_sessions`
  Session metadata and summary state

- `memory_traces`
  Lightweight turn traces

- `memory_journals`
  Narrative notes

- `memory_summaries`
  Session summaries, handoff summaries, and derived summaries

- `memory_preferences`
  Durable preferences and stable user-profile state

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

Then activate it in your Hermes config:

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
    reconsolidation_window_hours: 6
    review_intervals_days: "1,3,7,14,30"
    decay_half_life_days: 90
    decay_min_salience: 0.15
    builtin_snapshot_sync_enabled: true
    builtin_memory_dir: $HERMES_HOME/memories
    builtin_snapshot_user_chars: 1375
    builtin_snapshot_memory_chars: 2200
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
  LLM extraction only

Retrieval backends:

- `fts`
  SQLite full-text search only

- `hybrid`
  FTS candidate generation plus embedding reranking

## Local Model Support

The plugin can call OpenAI-compatible endpoints for both extraction and embeddings.

Resolution order for model settings:

1. plugin config
2. Hermes `model.default`
3. Hermes `model.base_url`

If your extraction endpoint needs auth:

```bash
export CONSOLIDATING_MEMORY_LLM_API_KEY=...
```

If your embedding endpoint uses separate credentials:

```bash
export CONSOLIDATING_MEMORY_EMBEDDING_API_KEY=...
```

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

## Benchmark Suite

The repo includes a benchmark package that compares:

- `baseline`
  Hermes built-in bounded memory only

- `addon`
  Hermes built-in bounded memory plus `consolidating_local`

The benchmark is designed to minimize token spend:

- seeding is done with direct file writes or provider API calls
- structural inspection is done from files, provider APIs, and SQLite
- LLM calls are used only for recall dimensions
- recall prompts are batched into one model call per system per dimension

The repo now also includes a second benchmark mode:

- a low-token structural benchmark
  Direct seeding and storage inspection, with LLM calls only for recall checks

- a complete real benchmark
  Real multi-turn `AIAgent` conversations for seeding and evaluation, plus LLM-backed addon extraction and hybrid retrieval

### Safety Rules

The benchmark helpers intentionally refuse to use live `~/.hermes` as a benchmark home.

Current safety behavior:

- fresh temporary `HERMES_HOME` directories are created for each benchmark system
- `ensure_fresh_home(...)` refuses to touch `~/.hermes`
- WSL support is available for using a real Hermes runtime while still keeping benchmark state in temp homes

### WSL Runtime Support

The benchmark can use a Hermes installation that lives in WSL2.

Current WSL support includes:

- reading model/provider seed config from WSL `~/.hermes`
- copying `auth.json` and `.env` into temp benchmark homes
- resolving runtime via the same Hermes route logic used by the CLI before constructing `AIAgent`
- running recall subprocesses through WSL Python

This matters because direct raw `AIAgent(model=...)` initialization can misroute Codex/OpenAI-style providers even when normal Hermes CLI usage works.

The complete real benchmark uses the same resolved Hermes runtime route for agent chats, but the addon's LLM extraction and embedding backends still require a real OpenAI-compatible backend.

Important limitation:

- ChatGPT Codex-style runtime URLs such as `chatgpt.com/backend-api/codex` are valid for Hermes agent chats but are not valid plugin backends for `consolidating_local`, because the plugin currently talks to `/chat/completions` and `/embeddings`
- for the full benchmark, use `--addon-llm-base-url` and `--addon-embedding-base-url` with an OpenAI-compatible endpoint if your normal Hermes runtime resolves to Codex

### Benchmark Dimensions

The benchmark currently implements:

- `DIM-1`
  Retention at scale

- `DIM-2`
  Overflow behavior

- `DIM-3`
  Cross-session recall quality

- `DIM-4`
  Contradiction handling

- `DIM-5`
  Prefetch context injection audit

- `DIM-6`
  Long-term salience

- `DIM-7`
  Recall precision under noise

### Complete Real Benchmark

The repo also includes a high-token, end-to-end benchmark focused on realistic memory behavior instead of token minimization.

Current design:

- seeding is done through real multi-turn `AIAgent` conversations
- baseline relies on Hermes built-in memory behavior during those chats
- addon uses the same real chats plus provider hook-style turn sync and session finalization
- addon defaults to `extractor_backend=llm` and `retrieval_backend=hybrid`
- evaluation prompts are separate real agent calls and scored locally from JSON responses

The current real dimensions are:

- `REAL-1`
  Natural acquisition and retention recall after real conversations

- `REAL-2`
  Correction handling and stale-value suppression

- `REAL-3`
  Task grounding accuracy on operational facts

- `REAL-4`
  Useful recall under conversational noise

- `REAL-5`
  Subject-change recall for facts that changed over time

The full benchmark fails fast if the addon would silently fall back away from LLM extraction or hybrid retrieval because the configured backend is missing or incompatible.

### Current DIM-5 Status

DIM-5 is resolved.

The provider now returns a non-empty context block for broad audit queries through `get_context(...)`, and the benchmark measures unique direct fact-bearing context lines for relevance instead of accidentally grading duplicated summary/provenance text as noise.

### Benchmark CLI

You can run the benchmark through:

- `python bench_compare.py`
- `python -m bench_compare`

For the complete real benchmark:

- `python bench_compare_full.py`
- `python -m bench_compare.full_main`

Example with WSL-backed runtime:

```powershell
python bench_compare.py `
  --model gpt-5.4 `
  --scale-facts 50 `
  --overflow-facts 200 `
  --dims DIM-1,DIM-2,DIM-3,DIM-4,DIM-5,DIM-6,DIM-7 `
  --output .\artifacts\benchmark\bench_results_YYYYMMDDTHHMMSSZ.json `
  --hermes-home-baseline .\tmp_baseline `
  --hermes-home-addon .\tmp_addon `
  --timeout 300 `
  --use-wsl `
  --wsl-distro Ubuntu `
  --wsl-hermes-root "~/.hermes/hermes-agent"
```

Important CLI options:

- `--model`
- `--scale-facts`
- `--overflow-facts`
- `--dims`
- `--output`
- `--hermes-home-baseline`
- `--hermes-home-addon`
- `--timeout`
- addon config overrides such as `--extractor-backend`, `--retrieval-backend`, `--prefetch-limit`, and decay/consolidation tuning flags
- WSL flags `--use-wsl`, `--wsl-distro`, `--wsl-hermes-root`, and `--wsl-python`

Important CLI options for the complete real benchmark:

- `--seed-batch-size`
- `--addon-llm-model`
- `--addon-llm-base-url`
- `--addon-embedding-model`
- `--addon-embedding-base-url`

Example complete real benchmark run:

```powershell
python bench_compare_full.py `
  --model gpt-5.4 `
  --dims REAL-1,REAL-2,REAL-3,REAL-4,REAL-5 `
  --output .\artifacts\benchmark\bench_results_full_YYYYMMDDTHHMMSSZ.json `
  --hermes-home-baseline .\tmp_baseline_real `
  --hermes-home-addon .\tmp_addon_real `
  --timeout 900 `
  --seed-batch-size 5 `
  --addon-llm-model "gpt-4o-mini" `
  --addon-llm-base-url "https://api.openai.com/v1" `
  --addon-embedding-model "text-embedding-3-large" `
  --addon-embedding-base-url "https://api.openai.com/v1" `
  --use-wsl `
  --wsl-distro Ubuntu `
  --wsl-hermes-root "~/.hermes/hermes-agent"
```

Set backend auth through environment variables when needed:

- `CONSOLIDATING_MEMORY_LLM_API_KEY`
- `CONSOLIDATING_MEMORY_EMBEDDING_API_KEY`

If Hermes or the provider cannot be imported, the benchmark exits with code `2`.

### Benchmark Outputs

The benchmark produces:

- a structured JSON file with metadata, per-dimension records, and weighted summary scoring
- a rich table on stdout when `rich` is available
- a plain text table otherwise

Current summary scoring is weighted:

- `DIM-3` has the highest weight because it measures real recall quality
- `DIM-5` has a smaller weight because it is more of an injection audit than a direct recall outcome
- `DIM-6` is addon-only and scored accordingly

The complete real benchmark also produces weighted summary scoring, but it emphasizes:

- `REAL-1` natural recall quality most heavily
- correction accuracy and stale suppression in `REAL-2`
- task usability in `REAL-3`
- noise resistance in `REAL-4`
- temporal change awareness in `REAL-5`

### PDF Report Generator

The benchmark includes a PDF renderer at [`bench_compare/report_pdf.py`](bench_compare/report_pdf.py).

It currently generates:

- an executive summary
- overview charts
- a methodology page
- one explanation page per dimension
- an appendix score table

Example:

```powershell
python -m bench_compare.report_pdf `
  --input .\artifacts\benchmark\bench_results_YYYYMMDDTHHMMSSZ.json `
  --output .\artifacts\benchmark\Hermes_Memory_Comparative_Report_YYYYMMDD.pdf
```

Generated benchmark artifacts are intended to live under `artifacts/benchmark/`, which is git-ignored.

## Repository Layout

```text
BENCHMARK_RUNBOOK.md
bench_compare.py
bench_compare_full.py
bench_compare/
  __main__.py
  full_main.py
  full_scenarios.py
  report.py
  report_pdf.py
  systems.py
  dims/
  utils/
plugins/memory/consolidating_local/
  __init__.py
  consolidator.py
  llm_client.py
  plugin.yaml
  store.py
  wiki_export.py
docs/
  V3_COMPILED_WIKI_MIRROR.md
```

## Development

Syntax check the plugin:

```bash
python -m compileall plugins/memory/consolidating_local
```

Syntax check the benchmark:

```bash
python -m compileall bench_compare bench_compare.py
```

Show benchmark CLI help:

```bash
python bench_compare.py --help
python bench_compare_full.py --help
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

Full rerun procedure:

- [BENCHMARK_RUNBOOK.md](BENCHMARK_RUNBOOK.md)

## Privacy

The provider is local-first, but it stores memory.

Do not publish your live runtime data such as:

- `~/.hermes/config.yaml`
- `~/.hermes/.env`
- `~/.hermes/auth.json`
- `~/.hermes/consolidating_memory.db`
- exported wiki directories containing private memories

The code is safe to publish.
Your live memory data probably is not.

## Design Direction

The current direction is:

- keep SQLite canonical
- keep markdown derived
- keep prompt injection compact
- keep memory auditable
- let the agent improve a compiled layer over time without hiding the state model

The longer-form design notes live in [docs/V3_COMPILED_WIKI_MIRROR.md](docs/V3_COMPILED_WIKI_MIRROR.md).

## License

MIT
