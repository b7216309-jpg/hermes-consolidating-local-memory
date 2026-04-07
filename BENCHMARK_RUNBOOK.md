# Hermes Memory Benchmark Runbook

This runbook captures the exact procedure used to run the Hermes memory
comparison benchmark from a fresh Codex session on this machine.

It is written for the current environment:

- Windows workspace at `C:\Users\Aezaror\Desktop\TESTUNKNOW2`
- Hermes installed in WSL2 at `~/.hermes/hermes-agent`
- live Hermes home at `~/.hermes`
- benchmark code in `bench_compare/`

## Goal

Run either benchmark mode while:

- using the installed Hermes runtime from WSL2
- keeping benchmark data in temporary benchmark homes
- optionally wiping live Hermes memory before and after the run
- preserving live Hermes auth and config

The repo now ships three benchmark modes:

- low-token structural benchmark
  `bench_compare.py`

- complete real benchmark
  `bench_compare_full.py`

- LongMemEval real benchmark
  `bench_compare_longmemeval.py`

## Important Findings

The recall dimensions must not instantiate raw `run_agent.AIAgent(model=...)`
without resolving runtime first.

Reason:

- `hermes` CLI works because it resolves provider, base URL, API key, and API
  mode before building the agent
- raw `AIAgent` initialization does not reliably carry over the resolved route
  metadata for `openai-codex`
- this caused false benchmark failures even though normal Hermes worked

The benchmark was fixed in:

- [llm.py](C:/Users/Aezaror/Desktop/TESTUNKNOW2/bench_compare/utils/llm.py)

That file now resolves Hermes runtime inside the recall subprocess using
`hermes_cli.runtime_provider.resolve_runtime_provider(...)` before building
`AIAgent`.

Second important finding:

- the complete real benchmark can now use the installed Hermes Codex backend for addon extraction
- this works because the plugin LLM client supports Codex `/responses` in addition to OpenAI-compatible `/chat/completions`
- embeddings are still a separate issue: if there is no live `/embeddings` backend, run the complete real benchmark with `--retrieval-backend fts`

Third important finding:

- the repo now has a LongMemEval-based real benchmark runner that replays timestamped LongMemEval haystack sessions into Hermes instead of trying to cram the entire history into one live chat
- each historical LongMemEval session is replayed as its own Hermes session, which keeps addon session boundaries meaningful and keeps the run practical on larger datasets
- answer scoring follows the LongMemEval task-specific yes/no judge prompts for knowledge updates, temporal reasoning, multi-session reasoning, preferences, and abstention

## Files That Matter

- [bench_compare.py](C:/Users/Aezaror/Desktop/TESTUNKNOW2/bench_compare.py)
- [bench_compare_full.py](C:/Users/Aezaror/Desktop/TESTUNKNOW2/bench_compare_full.py)
- [bench_compare_longmemeval.py](C:/Users/Aezaror/Desktop/TESTUNKNOW2/bench_compare_longmemeval.py)
- [__main__.py](C:/Users/Aezaror/Desktop/TESTUNKNOW2/bench_compare/__main__.py)
- [full_main.py](C:/Users/Aezaror/Desktop/TESTUNKNOW2/bench_compare/full_main.py)
- [longmemeval_main.py](C:/Users/Aezaror/Desktop/TESTUNKNOW2/bench_compare/longmemeval_main.py)
- [llm.py](C:/Users/Aezaror/Desktop/TESTUNKNOW2/bench_compare/utils/llm.py)
- [longmemeval.py](C:/Users/Aezaror/Desktop/TESTUNKNOW2/bench_compare/utils/longmemeval.py)
- [wsl.py](C:/Users/Aezaror/Desktop/TESTUNKNOW2/bench_compare/utils/wsl.py)
- [systems.py](C:/Users/Aezaror/Desktop/TESTUNKNOW2/bench_compare/systems.py)
- [report_pdf.py](C:/Users/Aezaror/Desktop/TESTUNKNOW2/bench_compare/report_pdf.py)
- [llm_client.py](C:/Users/Aezaror/Desktop/TESTUNKNOW2/plugins/memory/consolidating_local/llm_client.py)

## Live Hermes Memory Surfaces

If the user asks to clear live Hermes memory, wipe these and only these:

- `~/.hermes/memories/MEMORY.md`
- `~/.hermes/memories/USER.md`
- `~/.hermes/consolidating_memory.db`
- `~/.hermes/consolidating_memory.db-shm`
- `~/.hermes/consolidating_memory.db-wal`
- `~/.hermes/sessions/*`
- `~/.hermes/state.db`
- `~/.hermes/state.db-shm`
- `~/.hermes/state.db-wal`

Do not wipe:

- `~/.hermes/auth.json`
- `~/.hermes/.env`
- `~/.hermes/config.yaml`

## Standard Procedure

### 1. Verify Hermes runtime from WSL

Run:

```powershell
wsl.exe -d Ubuntu bash -lc "cd ~/.hermes/hermes-agent && .venv/bin/python - <<'PY'
from hermes_cli.auth import resolve_codex_runtime_credentials
from hermes_cli.codex_models import get_codex_model_ids
creds = resolve_codex_runtime_credentials()
print(creds.get('base_url'))
print(get_codex_model_ids(creds.get('api_key'))[:10])
PY"
```

Expected:

- base URL should be `https://chatgpt.com/backend-api/codex`
- model list should include `gpt-5.4`

### 2. Optionally wipe live Hermes memory before the run

Use this exact PowerShell block when the user explicitly wants live memory
cleared:

```powershell
$targets = @(
  '\\wsl$\Ubuntu\home\aezaror\.hermes\consolidating_memory.db',
  '\\wsl$\Ubuntu\home\aezaror\.hermes\consolidating_memory.db-shm',
  '\\wsl$\Ubuntu\home\aezaror\.hermes\consolidating_memory.db-wal',
  '\\wsl$\Ubuntu\home\aezaror\.hermes\state.db',
  '\\wsl$\Ubuntu\home\aezaror\.hermes\state.db-shm',
  '\\wsl$\Ubuntu\home\aezaror\.hermes\state.db-wal'
)
foreach ($path in $targets) {
  if (Test-Path -LiteralPath $path) {
    Remove-Item -LiteralPath $path -Force -ErrorAction Stop
  }
}
$sessionDir = '\\wsl$\Ubuntu\home\aezaror\.hermes\sessions'
if (Test-Path -LiteralPath $sessionDir) {
  Get-ChildItem -LiteralPath $sessionDir -Force -ErrorAction Stop | Remove-Item -Force -Recurse -ErrorAction Stop
}
$memoryDir = '\\wsl$\Ubuntu\home\aezaror\.hermes\memories'
if (!(Test-Path -LiteralPath $memoryDir)) {
  New-Item -ItemType Directory -Path $memoryDir -Force | Out-Null
}
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText((Join-Path $memoryDir 'MEMORY.md'), '', $utf8NoBom)
[System.IO.File]::WriteAllText((Join-Path $memoryDir 'USER.md'), '', $utf8NoBom)
```

### 3. Run the low-token structural benchmark

Use temp benchmark homes, not `~/.hermes`:

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

Notes:

- The benchmark uses WSL Hermes only for runtime and recall calls.
- Baseline/addon data is stored in the temp homes above, not in live `~/.hermes`.
- The benchmark copies runtime auth/config seeds needed for the temp homes.
- `artifacts/benchmark/` is ignored by git, so generated JSON/PDF outputs stay out of repo status.

### 3b. Run the complete real benchmark

Use temp benchmark homes, not `~/.hermes`:

```powershell
python bench_compare_full.py `
  --model gpt-5.4 `
  --scale-facts 50 `
  --dims REAL-1,REAL-2,REAL-3,REAL-4,REAL-5 `
  --output .\artifacts\benchmark\bench_results_full_YYYYMMDDTHHMMSSZ.json `
  --hermes-home-baseline .\tmp_full_baseline `
  --hermes-home-addon .\tmp_full_addon `
  --timeout 900 `
  --seed-batch-size 5 `
  --addon-llm-model gpt-5.4 `
  --retrieval-backend fts `
  --use-wsl `
  --wsl-distro Ubuntu `
  --wsl-hermes-root "~/.hermes/hermes-agent"
```

Notes:

- this is the exact working path used for the final complete real benchmark run
- addon extraction uses the resolved Codex backend through `/responses`
- retrieval is set to `fts` because the current installation does not expose a live embeddings endpoint
- if a working embeddings endpoint becomes available, switch to `--retrieval-backend hybrid` and pass `--addon-embedding-model` plus `--addon-embedding-base-url`

### 3c. Run the LongMemEval real benchmark

Use temp benchmark homes, not `~/.hermes`:

```powershell
python bench_compare_longmemeval.py `
  --model gpt-5.4 `
  --dataset .\data\longmemeval_oracle.json `
  --output .\artifacts\benchmark\bench_results_longmemeval_YYYYMMDDTHHMMSSZ.json `
  --hermes-home-baseline .\tmp_lme_baseline `
  --hermes-home-addon .\tmp_lme_addon `
  --timeout 900 `
  --history-format json `
  --include-abstention all `
  --retrieval-backend fts `
  --use-wsl `
  --wsl-distro Ubuntu `
  --wsl-hermes-root "~/.hermes/hermes-agent"
```

Notes:

- this runner expects a LongMemEval JSON or JSONL dataset file downloaded separately
- it uses the same resolved WSL Hermes runtime route as the other benchmark runners
- addon extraction can use the resolved Codex backend through `/responses`
- retrieval should stay on `fts` unless a working embeddings backend is explicitly available
- for quick smoke tests, use `--limit 1` and optionally `--max-sessions-per-example 2`

### 4. Check the result file

The final result JSON will be the file passed via `--output`.

Good recent known-good example:

- `artifacts/benchmark/bench_results_report_source_20260405.json`

Good recent known-good complete real example:

- `artifacts/benchmark/bench_results_full_20260405T202253Z_clean.json`

LongMemEval runner smoke example:

- synthetic 1-instance smoke run passed on `2026-04-05` with `--limit 1`
- baseline and addon both answered the simple knowledge-update sample correctly
- total usage on that smoke run was `8` LLM calls and roughly `5k` estimated tokens

Key values from that run:

- `DIM-1`: baseline `36/50`, addon `50/50`
- `DIM-2`: baseline `38 kept`, addon `200 kept`
- `DIM-3`: baseline `F1=0.79`, addon `F1=0.68`
- `DIM-4`: baseline `0/10`, addon `10/10`
- `DIM-5`: baseline `2710ch rel=1.00`, addon `2941ch rel=1.00`
- `DIM-6`: addon `PASS hi-lo=0.48`
- `DIM-7`: baseline `SNR=0.39`, addon `SNR=0.90`

Key values from the complete real run:

- `REAL-1`: baseline `F1=0.36`, addon `F1=0.92`
- `REAL-2`: baseline `acc=1.00`, addon `acc=0.90`
- `REAL-3`: tie at `0.90`
- `REAL-4`: baseline `SNR=0.50`, addon `SNR=0.87`
- `REAL-5`: tie at `F1=1.00`
- overall winner: addon

### 4b. Render the PDF report

Use the fresh JSON result as the input:

```powershell
python -m bench_compare.report_pdf `
  --input .\artifacts\benchmark\bench_results_report_source_20260405.json `
  --output .\artifacts\benchmark\Hermes_Memory_Comparative_Report_20260405.pdf
```

This produces a multi-page PDF with:

- executive summary
- overview graphs
- one explanation page per dimension
- compact appendix table

For the complete real benchmark:

```powershell
python -m bench_compare.report_pdf `
  --input .\artifacts\benchmark\bench_results_full_20260405T202253Z_clean.json `
  --output .\artifacts\benchmark\Hermes_Complete_Real_Benchmark_Report_20260405.pdf
```

### 5. Wipe live Hermes memory again after the run

If the user requested “clear before and after”, run the same wipe block again.

### 6. Verify live memory is empty

Run:

```powershell
Get-ChildItem -LiteralPath '\\wsl$\Ubuntu\home\aezaror\.hermes\memories' -Force | Select-Object FullName,Length
Get-ChildItem -LiteralPath '\\wsl$\Ubuntu\home\aezaror\.hermes\sessions' -Force -ErrorAction SilentlyContinue | Select-Object FullName,Length
Get-Item -LiteralPath '\\wsl$\Ubuntu\home\aezaror\.hermes\consolidating_memory.db','\\wsl$\Ubuntu\home\aezaror\.hermes\state.db' -ErrorAction SilentlyContinue | Select-Object FullName,Length
```

Expected when fully wiped:

- `MEMORY.md` length `0`
- `USER.md` length `0`
- no session files
- no `consolidating_memory.db`
- no `state.db`

## Fast Partial Runs

### Recall-only debug

Use this when debugging model/provider issues:

```powershell
python bench_compare.py `
  --model gpt-5.4 `
  --dims DIM-3,DIM-7 `
  --output .\artifacts\benchmark\bench_results_dim37_fixed.json `
  --hermes-home-baseline .\tmp_baseline `
  --hermes-home-addon .\tmp_addon `
  --timeout 300 `
  --use-wsl `
  --wsl-distro Ubuntu `
  --wsl-hermes-root "~/.hermes/hermes-agent"
```

### Prefetch-only debug

Use this when checking why addon context injection is empty:

```powershell
python bench_compare.py `
  --model gpt-5.4 `
  --dims DIM-5 `
  --output .\artifacts\benchmark\bench_results_dim5.json `
  --hermes-home-baseline .\tmp_baseline `
  --hermes-home-addon .\tmp_addon `
  --timeout 300 `
  --use-wsl `
  --wsl-distro Ubuntu `
  --wsl-hermes-root "~/.hermes/hermes-agent"
```

### Complete-real smoke run

Use this before the expensive full run:

```powershell
python bench_compare_full.py `
  --model gpt-5.4 `
  --scale-facts 10 `
  --dims REAL-1 `
  --output .\artifacts\benchmark\bench_results_full_smoke.json `
  --hermes-home-baseline .\tmp_full_smoke_baseline `
  --hermes-home-addon .\tmp_full_smoke_addon `
  --timeout 600 `
  --seed-batch-size 5 `
  --addon-llm-model gpt-5.4 `
  --retrieval-backend fts `
  --use-wsl `
  --wsl-distro Ubuntu `
  --wsl-hermes-root "~/.hermes/hermes-agent"
```

## Current DIM-5 Status

`DIM-5` is resolved.

Root cause that was fixed:

- broad subjectless audit queries were entering addon provenance mode
- retrieval found no direct lexical match
- `provider.get_context(...)` returned an empty string instead of a current-state fallback

Current behavior:

- addon `get_context()` now falls back to a merged current-memory snapshot for empty
  subjectless provenance or current-state queries
- the benchmark now measures direct fact-bearing context lines for relevance
- the latest full run shows `DIM-5`: baseline `2710ch rel=1.00`, addon `2941ch rel=1.00`

If this regresses in the future, inspect:

- `bench_compare/dims/dim5_prefetch.py`
- `bench_compare/report_pdf.py`
- `plugins/memory/consolidating_local/__init__.py`

## Current Complete-Real Backend Status

What works on this machine today:

- Hermes runtime resolves to `openai-codex`
- addon extraction can use that Codex backend through `/responses`
- complete real benchmark runs successfully with `--addon-llm-model gpt-5.4`

What does not currently work:

- the old local OpenAI-style base URL in `~/.hermes/.env` at `http://172.18.16.1:8080/v1` times out
- no working embeddings endpoint is currently available from the installed setup

Current recommendation:

- run the complete real benchmark with `--retrieval-backend fts`
- only switch to hybrid retrieval once a real embedding endpoint is confirmed working

## Sanity Check Commands Used During Debugging

### Verify explicit Codex route works

This was the proof that Hermes config was fine and raw benchmark agent setup was
the real problem:

```powershell
@'
import importlib
from hermes_cli.auth import resolve_codex_runtime_credentials
AIAgent = importlib.import_module('run_agent').AIAgent
creds = resolve_codex_runtime_credentials()
agent = AIAgent(
    model='gpt-5.4',
    provider='openai-codex',
    base_url=creds['base_url'],
    api_key=creds['api_key'],
    api_mode='codex_responses',
    skip_context_files=True,
    session_id='diag-explicit-codex',
    quiet_mode=True,
)
if hasattr(agent, 'tools'):
    agent.tools = []
if hasattr(agent, 'valid_tool_names'):
    agent.valid_tool_names = []
print(agent.chat('Respond ONLY with a JSON array containing one string: "ok"'))
'@ | wsl.exe -d Ubuntu bash -lc "cd ~/.hermes/hermes-agent && HERMES_HOME=~/.hermes .venv/bin/python -"
```

Expected:

- returns `["ok"]`

## Kept Final Artifacts

Keep only the latest final result pair for each benchmark:

- `artifacts/benchmark/bench_results_report_source_20260405.json`
- `artifacts/benchmark/Hermes_Memory_Comparative_Report_20260405.pdf`
- `artifacts/benchmark/bench_results_full_20260405T202253Z_clean.json`
- `artifacts/benchmark/Hermes_Complete_Real_Benchmark_Report_20260405.pdf`

LongMemEval runs are usually larger and more situational, so keep only the final JSON or PDF you actually intend to compare or publish.

Superseded smoke, rerun, and intermediate result files can be removed.

## Suggested Future Session Prompt

If a later session needs to rerun this benchmark, say:

```text
Follow BENCHMARK_RUNBOOK.md exactly. Use the installed WSL Hermes runtime,
keep benchmark state in temp homes, and run either:
- the low-token structural benchmark, or
- the complete real benchmark with gpt-5.4, Codex-backed extraction, and retrieval_backend fts unless a live embeddings endpoint is available.
- the LongMemEval real benchmark with a downloaded dataset file, json history replay, and retrieval_backend fts unless a live embeddings endpoint is available.
```
