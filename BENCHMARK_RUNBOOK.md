# Hermes Memory Benchmark Runbook

This runbook captures the exact procedure used to run the Hermes memory
comparison benchmark from a fresh Codex session on this machine.

It is written for the current environment:

- Windows workspace at `C:\Users\Aezaror\Desktop\TESTUNKNOW2`
- Hermes installed in WSL2 at `~/.hermes/hermes-agent`
- live Hermes home at `~/.hermes`
- benchmark code in `bench_compare/`

## Goal

Run the baseline vs addon comparison benchmark while:

- using the installed Hermes runtime from WSL2
- keeping benchmark data in temporary benchmark homes
- optionally wiping live Hermes memory before and after the run
- preserving live Hermes auth and config

## Important Finding

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

## Files That Matter

- [bench_compare.py](C:/Users/Aezaror/Desktop/TESTUNKNOW2/bench_compare.py)
- [__main__.py](C:/Users/Aezaror/Desktop/TESTUNKNOW2/bench_compare/__main__.py)
- [llm.py](C:/Users/Aezaror/Desktop/TESTUNKNOW2/bench_compare/utils/llm.py)
- [wsl.py](C:/Users/Aezaror/Desktop/TESTUNKNOW2/bench_compare/utils/wsl.py)
- [systems.py](C:/Users/Aezaror/Desktop/TESTUNKNOW2/bench_compare/systems.py)

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

### 3. Run the full benchmark

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

### 4. Check the result file

The final result JSON will be the file passed via `--output`.

Good recent known-good example:

- `artifacts/benchmark/bench_results_report_source_20260405.json`

Key values from that run:

- `DIM-1`: baseline `36/50`, addon `50/50`
- `DIM-2`: baseline `38 kept`, addon `200 kept`
- `DIM-3`: baseline `F1=0.79`, addon `F1=0.68`
- `DIM-4`: baseline `0/10`, addon `10/10`
- `DIM-5`: baseline `2710ch rel=1.00`, addon `2941ch rel=1.00`
- `DIM-6`: addon `PASS hi-lo=0.48`
- `DIM-7`: baseline `SNR=0.39`, addon `SNR=0.90`

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

## Suggested Future Session Prompt

If a later session needs to rerun this benchmark, say:

```text
Follow BENCHMARK_RUNBOOK.md exactly. Use the installed WSL Hermes runtime,
run the full benchmark with gpt-5.4, and wipe live Hermes memory before and
after the run.
```
