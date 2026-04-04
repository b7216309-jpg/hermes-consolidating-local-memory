# V3 Compiled Wiki Mirror

## Goal

Level up `consolidating_local` by adding a derived markdown knowledge surface on top of the current SQLite-first memory provider.

The key idea is:

- SQLite remains the authoritative memory store
- markdown becomes a compiled mirror, not the source of truth
- the provider can export, lint, and enrich this mirror for both humans and LLMs

This takes the best part of the Karpathy workflow without losing the strengths we already have:

- constrained writes
- contradiction-aware state
- append-only history
- typed provenance links
- local-first operation

## Why This Fits Our System

The current provider already has the right internal shape:

- raw capture: `episodes`, `traces`
- compiled memory: `facts`, `summaries`, `preferences`, `policies`, `topics`
- provenance: `memory_links`
- audit trail: `memory_history`
- retrieval: FTS plus optional hybrid reranking

What is missing is a durable, inspectable, file-based surface that can:

- be browsed in Obsidian or any markdown tool
- accumulate reports and derived outputs
- expose backlinks and provenance clearly
- let the LLM operate on a "compiled wiki" instead of only JSON tool output

## Non-Goals

V3 should not:

- replace SQLite with markdown files
- replace Hermes session storage
- become a full arbitrary graph database
- let the LLM freely rewrite authoritative memory without constraints
- try to ingest every external data type in the first slice

## Product Shape

`consolidating_local` becomes a two-layer system:

1. Authoritative layer

- SQLite store
- all mutation logic
- contradiction resolution
- history and links
- retrieval and decay

2. Compiled mirror layer

- markdown wiki generated from SQLite
- index pages
- topic pages
- session pages
- preference and policy pages
- contradiction and health reports

The mirror is disposable and regenerable.

## V3 Features

### 1. Wiki Export

Add a new exporter that renders the current memory graph into markdown files.

Suggested output tree:

```text
compiled_wiki/
  index.md
  topics/
    user-preferences.md
    project-delivery.md
  sessions/
    test-session.md
  preferences/
    index.md
  policies/
    index.md
  contradictions/
    index.md
  reports/
    latest-health-check.md
```

Every page should be generated from existing memory objects and links.

The exporter should be:

- idempotent
- deterministic
- safe to rerun after every consolidation
- able to prune stale generated pages

### 2. Backlinks And Provenance Rendering

The wiki mirror should render:

- supporting facts for each topic
- session summaries and linked traces
- which facts support a preference
- which facts contradict or supersede older facts
- recent recall events where useful

Examples:

- topic pages show linked facts and source sessions
- session pages show summaries, facts, journals, traces, preferences, and policies touched in that session
- contradiction pages show old assumption -> new assumption transitions

### 3. Memory Health Checks

Add a linter that audits the store and writes a markdown report.

Checks should include:

- duplicate or near-duplicate active facts
- active facts missing expected provenance
- summaries with no source refs
- topic pages with stale or missing supporting links
- preferences or policies missing session capture metadata
- inconsistent exclusive-state chains
- orphaned links
- inactive records still referenced by active pages
- high-value sections missing a compiled page

The report should include:

- findings
- severity
- suggested fixes
- counts by issue type

### 4. Report Output As First-Class Memory Artifact

Let the agent write outputs back into the mirror as files, not only chat text.

Examples:

- handoff note
- research brief
- memory audit
- contradiction digest
- session recap

These reports should live in `compiled_wiki/reports/`.

### 5. Optional External Source Lane

Do not mix external research documents directly into conversational memory.

Instead, add a later lane for external knowledge sources with a separate model:

- `raw_sources/` for imported markdown, notes, repo snapshots, or metadata
- compiled source summaries in `compiled_wiki/sources/`
- explicit links from source-derived knowledge to memory pages

This should arrive after the wiki export and linting layers are stable.

## V3 Tool Surface

Keep the single-tool approach and extend `consolidating_memory`.

### New actions

- `export`
  Render or refresh the compiled wiki mirror.

- `lint`
  Run health checks and optionally write a markdown report.

- `report`
  Write a generated markdown artifact into the wiki report directory.

### Existing actions to extend

- `status`
  Include wiki export status, last export time, export root, last lint time, and latest report path.

- `distill`
  Optionally write the distilled output to a wiki file.

## Config Additions

Suggested new config keys:

- `wiki_export_enabled`
- `wiki_export_dir`
- `wiki_export_on_consolidate`
- `wiki_export_session_limit`
- `wiki_export_topic_limit`
- `wiki_export_include_history`
- `wiki_report_dir`
- `wiki_lint_on_consolidate`
- `wiki_lint_max_findings`

Defaults:

- export disabled by default
- no runtime dependency on Obsidian
- all paths local and configurable

## File Layout Changes

Suggested new modules:

- `plugins/memory/consolidating_local/wiki_export.py`
- `plugins/memory/consolidating_local/health_checks.py`

Responsibilities:

- `wiki_export.py`
  Render pages, indexes, backlinks, and cleanup stale generated files.

- `health_checks.py`
  Run audits, score findings, and generate markdown reports.

This keeps `store.py` authoritative, `__init__.py` orchestration-focused, and avoids overloading the provider entrypoint.

## Export Rules

### Index page

`index.md` should summarize:

- latest session summaries
- top topics
- active preferences
- active policies
- recent contradictions
- latest reports

### Topic pages

Each topic page should include:

- title
- summary
- strongest supporting facts
- related sessions
- linked contradictions
- backlinks to preferences and policies when relevant

### Session pages

Each session page should include:

- session label and timestamps
- session summary
- traces
- journals
- facts extracted from the session
- preferences and policies captured in the session

### Preference and policy indexes

These should render stable high-salience memory in one place for fast browsing.

### Contradiction index

This should render a changelog-style view:

- subject key
- replaced assumption
- winning assumption
- resolution time

## Health Check Rules

Initial lint rules should be deterministic and not require an LLM.

That means:

- SQL and heuristic checks first
- optional LLM explanation later

This keeps linting local, fast, and predictable.

## Rollout Plan

### Phase 1. Compiled wiki mirror

Build only from existing store tables.

Deliver:

- exporter module
- `export` action
- `status` export metadata
- file rendering for index, topics, sessions, preferences, policies, contradictions
- cleanup of stale generated files

This is the highest-leverage slice.

### Phase 2. Memory health checks

Deliver:

- health check module
- `lint` action
- markdown report output
- status metadata for last lint run

### Phase 3. File-backed reports

Deliver:

- `report` action
- ability for `distill` to also write markdown artifacts
- report index page

### Phase 4. External source lane

Deliver:

- raw source directory support
- source summary pages
- links from source summaries to memory pages

This should be separate from core conversational memory consolidation.

## Testing Plan

### Export tests

- export writes expected page tree
- repeated export is idempotent
- stale generated files are pruned
- topic and session pages include backlinks
- exported contradictions reflect current store state

### Health check tests

- duplicate facts are flagged
- missing provenance is flagged
- stale links are flagged
- valid data produces an empty or low-noise report

### Integration tests

- consolidation followed by export produces updated session and topic pages
- preference and policy capture appears in the wiki mirror
- report generation creates markdown files in the expected location

## First Implementation Slice

The first build should be intentionally narrow:

- no external source ingest yet
- no LLM-authored freeform wiki edits
- no new canonical storage backend

Implement only:

- `wiki_export.py`
- `export` action
- config for export root and auto-export on consolidation
- deterministic markdown rendering from existing tables and links

If Phase 1 lands well, the plugin immediately becomes more inspectable, more Obsidian-friendly, and much more useful for long-running agent workflows.

## Recommendation

This V3 direction is worth doing.

The best fit is not "turn the plugin into a wiki."
The best fit is:

- keep the provider as a local memory engine
- add a compiled markdown mirror
- add health checks and report generation
- later add a separate external-source knowledge lane

That gives us the strongest parts of the Karpathy workflow without sacrificing correctness, locality, or control.
