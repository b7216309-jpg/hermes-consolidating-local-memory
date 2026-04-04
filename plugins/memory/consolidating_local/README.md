# consolidating_local

`consolidating_local` is a Hermes memory provider scaffold built around a practical layered long-term memory pattern:

- keep a tiny always-on memory surface
- keep only short-lived raw turn buffers locally
- distill sessions into durable facts, preferences, policies, and summaries
- periodically consolidate them into topic summaries and contradiction-aware state
- retrieve only the relevant slices back into the prompt

This version is local-first and dependency-light:

- SQLite backend
- background worker for non-blocking turn sync
- bounded episodic buffers (`episodes`)
- session records, traces, journals, and summaries
- durable facts, preferences, and policies
- merged topic summaries (`topics`)
- contradiction log plus append-only history and provenance links
- auto-consolidation gate (`min_hours` + `min_sessions`)
- optional hybrid retrieval with OpenAI-compatible embeddings

## What it does

Hermes already has built-in `MEMORY.md` and `USER.md`. This provider runs alongside them and adds:

- `prefetch(query)`: injects relevant topic summaries and matching facts before each turn
- `sync_turn(user, assistant)`: appends each turn to episodic memory and a lightweight trace row
- `on_turn_start(...)`: checks whether background consolidation should run
- `on_session_end(...)`: extracts durable signal, updates preferences, creates a session summary, and requests consolidation
- `on_pre_compress(...)`: salvages useful facts before compression drops raw context and writes a handoff summary
- `on_memory_write(...)`: mirrors built-in Hermes memory writes into the provider and upgrades user memory into preferences when appropriate
- `consolidating_memory`: one richer tool for search, journaling, distillation, history, policies, decay, and fact writes

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
    session_summary_chars: 900
    prune_after_days: 90
    episode_body_retention_hours: 24
    decay_half_life_days: 90
    decay_min_salience: 0.15
    extractor_backend: hybrid
    retrieval_backend: fts
    llm_model: ""          # blank = use Hermes model.default
    llm_base_url: ""       # blank = use Hermes model.base_url
    llm_timeout_seconds: 45
    llm_max_input_chars: 4000
    embedding_model: ""          # blank = use llm_model/model.default
    embedding_base_url: ""       # blank = use llm_base_url/model.base_url
    embedding_timeout_seconds: 20
    embedding_candidate_limit: 16
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
- `journal`
- `distill`
- `history`
- `policy`
- `decay`

`search.scope` supports:

- `all`
- `facts`
- `topics`
- `episodes`
- `summaries`
- `journals`
- `preferences`
- `policies`

## Current behavior

The consolidation loop is still local-first, but it is now much more structured than the first scaffold:

- it extracts atomic facts for user preferences, favorites, likes/dislikes, personal details, life events, environment details, workflow rules, and common project signals
- it carries subject/value metadata for exclusive facts like response style, shell, OS, SSH port, and Docker sudo behavior
- it tracks user-specific contradiction cases such as like -> dislike, favorite changes, location changes, and diet/profile updates
- it logs contradiction resolutions when a newer fact replaces an older assumption, and records matching history plus provenance links
- it stores bounded raw turns as episode buffers, then prunes those buffers after successful consolidation
- it creates memory sessions, turn traces, journals, session summaries, and handoff summaries
- it upgrades durable user-profile signals into first-class preferences and workflow rules into policies
- it buckets facts into coarse topics and rebuilds concise topic summaries from the strongest active facts
- it applies salience decay so low-value traces, journals, summaries, topics, and facts fade out over time without erasing high-salience preferences and policies
- it can rerank FTS candidates with local embeddings when `retrieval_backend: hybrid` is configured and available
- it renders prefetch in a fixed order: summaries, preferences/policies, direct fact matches, recent journals, and changed assumptions

That gives you a practical memory loop without adding a cloud memory service.

When `extractor_backend` is `hybrid` or `llm`, the provider also tries an OpenAI-compatible local model for extraction and falls back to heuristics if the model call or JSON parse fails.

When `retrieval_backend` is `hybrid`, the provider also tries an OpenAI-compatible `/embeddings` endpoint for reranking and falls back to plain FTS if embeddings are unavailable.
