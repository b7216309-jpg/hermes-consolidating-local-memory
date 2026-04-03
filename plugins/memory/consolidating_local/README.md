# consolidating_local

`consolidating_local` is a Hermes memory provider scaffold built around a practical long-term memory pattern:

- keep a tiny always-on memory surface
- store raw turn history separately
- periodically consolidate it into durable facts and topic summaries
- retrieve only the relevant slices back into the prompt

This version is local-first and dependency-light:

- SQLite backend
- background worker for non-blocking turn sync
- episodic store (`episodes`)
- durable facts (`facts`)
- merged topic summaries (`topics`)
- contradiction log (`contradictions`)
- auto-consolidation gate (`min_hours` + `min_sessions`)

## What it does

Hermes already has built-in `MEMORY.md` and `USER.md`. This provider runs alongside them and adds:

- `prefetch(query)`: injects relevant topic summaries and matching facts before each turn
- `sync_turn(user, assistant)`: appends each turn to episodic memory
- `on_turn_start(...)`: checks whether background consolidation should run
- `on_session_end(...)`: extracts more durable signal and requests consolidation
- `on_pre_compress(...)`: salvages useful facts before compression drops raw context
- `on_memory_write(...)`: mirrors built-in Hermes memory writes into the provider

## Install into Hermes

Copy this directory into your Hermes checkout:

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
    prune_after_days: 90
    extractor_backend: hybrid
    llm_model: ""          # blank = use Hermes model.default
    llm_base_url: ""       # blank = use Hermes model.base_url
    llm_timeout_seconds: 45
    llm_max_input_chars: 4000
```

If your Hermes profile already uses a local OpenAI-compatible endpoint, `hybrid` mode will automatically reuse it. In your case, that means it can reuse the existing Qwen setup from `~/.hermes/config.yaml`.

## Tool

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

## Current behavior

The consolidation pass is heuristic, not LLM-driven, but it is now more structured than the first scaffold:

- it extracts atomic facts for user preferences, environment details, workflow rules, and common project signals
- it carries subject/value metadata for exclusive facts like response style, shell, OS, SSH port, and Docker sudo behavior
- it logs contradiction resolutions when a newer fact replaces an older assumption
- it buckets facts into coarse topics
- it supersedes older facts with the same short signature or the same exclusive subject state
- it prunes low-value extracted facts after a retention window
- it rebuilds concise topic summaries from the strongest active facts

That gives you a practical memory loop without adding a cloud memory service.

When `extractor_backend` is `hybrid` or `llm`, the provider also tries an OpenAI-compatible local model for extraction and falls back to heuristics if the model call or JSON parse fails.

## Good next upgrades

- replace heuristic extraction with an LLM-backed summarizer
- add entity linking and contradiction scoring
- add per-workspace topic partitions
- add richer retrieval ranking than SQLite FTS + recency + importance
- emit a "changed assumptions" block during prefetch
