from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Mapping

from bench_compare.utils.wsl import default_wsl_python, run_wsl_python, windows_to_wsl_path


def estimate_tokens_from_chars(*texts: str) -> int:
    total_chars = sum(len(text or "") for text in texts)
    return int(round(total_chars / 3.8))


def parse_json_array(response_text: str) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    raw = str(response_text or "").strip()
    if not raw:
        return [], ["Empty response from model."]
    try:
        parsed = json.loads(raw)
        return _normalize_array(parsed), errors
    except Exception as exc:
        errors.append(f"Direct JSON parse failed: {exc}")
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
            return _normalize_array(parsed), errors
        except Exception as exc:
            errors.append(f"Embedded JSON parse failed: {exc}")
    items = [line.strip().lstrip("-*").strip().strip('"') for line in raw.splitlines()]
    items = [item for item in items if item]
    if items:
        errors.append("Fell back to line-based parsing.")
        return items, errors
    return [], errors


def parse_json_object(response_text: str) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    raw = str(response_text or "").strip()
    if not raw:
        return {}, ["Empty response from model."]
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed, errors
        errors.append(f"Expected JSON object, got {type(parsed).__name__}.")
    except Exception as exc:
        errors.append(f"Direct JSON parse failed: {exc}")
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
            if isinstance(parsed, dict):
                return parsed, errors
            errors.append(f"Embedded JSON object parse returned {type(parsed).__name__}.")
        except Exception as exc:
            errors.append(f"Embedded JSON parse failed: {exc}")
    return {}, errors


def validate_agent_runtime(*, repo_root: Path, wsl_settings: Mapping[str, Any] | None = None) -> None:
    script = textwrap.dedent(
        """
        import importlib
        errors = []
        for name in ("hermes.agent", "run_agent"):
            try:
                module = importlib.import_module(name)
                getattr(module, "AIAgent")
                print("OK")
                raise SystemExit(0)
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
        print(" | ".join(errors))
        raise SystemExit(2)
        """
    ).strip()
    if wsl_settings and wsl_settings.get("enabled"):
        proc = _run_wsl_script(
            script=script,
            payload="",
            repo_root=repo_root,
            hermes_home=Path(str(wsl_settings["hermes_home"])),
            wsl_settings=wsl_settings,
            timeout_seconds=30,
        )
    else:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(repo_root),
            timeout=30,
            env=os.environ.copy(),
        )
    if proc.returncode != 0:
        message = proc.stdout.strip() or proc.stderr.strip() or "Unable to import Hermes agent runtime."
        raise RuntimeError(message)


def run_agent_recall(
    *,
    repo_root: Path,
    hermes_home: Path,
    model: str,
    query: str,
    timeout_seconds: int,
    addon: bool,
    provider_config: Mapping[str, Any] | None = None,
    wsl_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "repo_root": str(repo_root),
        "hermes_home": str(hermes_home),
        "model": model,
        "query": query,
        "addon": bool(addon),
        "provider_config": dict(provider_config or {}),
    }
    script = _recall_script()
    if wsl_settings and wsl_settings.get("enabled"):
        proc = _run_wsl_script(
            script=script,
            payload=json.dumps(payload),
            repo_root=repo_root,
            hermes_home=hermes_home,
            wsl_settings=wsl_settings,
            timeout_seconds=timeout_seconds,
        )
    else:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(repo_root),
            timeout=timeout_seconds,
            env={**os.environ, "HERMES_HOME": str(hermes_home)},
        )
    if proc.returncode != 0:
        message = proc.stderr.strip() or proc.stdout.strip() or f"Recall subprocess exited with code {proc.returncode}."
        raise RuntimeError(message)
    stdout_text = str(proc.stdout or "").strip()
    if not stdout_text:
        stderr_text = str(proc.stderr or "").strip()
        raise RuntimeError(stderr_text or "Recall subprocess produced no stdout.")
    try:
        parsed = json.loads(stdout_text)
    except Exception as exc:
        stderr_text = str(proc.stderr or "").strip()
        details = f"{exc}. stdout={stdout_text[:400]!r}"
        if stderr_text:
            details += f" stderr={stderr_text[:400]!r}"
        raise RuntimeError(f"Failed to parse recall subprocess output: {details}") from exc
    recalled_items, parse_errors = parse_json_array(parsed.get("response_text", ""))
    call_error = str(parsed.get("call_error") or "").strip()
    agent_logs = str(parsed.get("agent_logs") or "").strip()
    if call_error:
        parse_errors.append(f"Recall call failed: {call_error}")
    elif not recalled_items and not str(parsed.get("response_text") or "").strip() and agent_logs:
        parse_errors.append(f"Recall call produced no response. Logs: {_summarize_logs(agent_logs)}")
    system_prompt_chars = int(parsed.get("system_prompt_chars") or 0)
    context_chars = int(parsed.get("context_chars") or 0)
    prompt_chars = int(parsed.get("prompt_chars") or 0)
    tokens_estimated = estimate_tokens_from_chars(
        "x" * system_prompt_chars,
        "x" * prompt_chars,
        str(parsed.get("response_text") or ""),
    )
    return {
        "response_text": str(parsed.get("response_text") or ""),
        "recalled_items": recalled_items,
        "parse_errors": parse_errors,
        "system_prompt_chars": system_prompt_chars,
        "context_chars": context_chars,
        "prompt_chars": prompt_chars,
        "tokens_estimated": tokens_estimated,
    }


def run_agent_conversation(
    *,
    repo_root: Path,
    hermes_home: Path,
    model: str,
    prompts: list[str],
    timeout_seconds: int,
    addon: bool,
    provider_config: Mapping[str, Any] | None = None,
    wsl_settings: Mapping[str, Any] | None = None,
    session_id: str = "bench-session",
    sync_provider_turns: bool = False,
    finalize_provider_session: bool = False,
) -> dict[str, Any]:
    payload = {
        "repo_root": str(repo_root),
        "hermes_home": str(hermes_home),
        "model": model,
        "addon": bool(addon),
        "provider_config": dict(provider_config or {}),
        "prompts": [str(prompt or "") for prompt in prompts],
        "session_id": str(session_id or "bench-session"),
        "sync_provider_turns": bool(sync_provider_turns),
        "finalize_provider_session": bool(finalize_provider_session),
    }
    script = _conversation_script()
    if wsl_settings and wsl_settings.get("enabled"):
        proc = _run_wsl_script(
            script=script,
            payload=json.dumps(payload),
            repo_root=repo_root,
            hermes_home=hermes_home,
            wsl_settings=wsl_settings,
            timeout_seconds=timeout_seconds,
        )
    else:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(repo_root),
            timeout=timeout_seconds,
            env={**os.environ, "HERMES_HOME": str(hermes_home)},
        )
    if proc.returncode != 0:
        message = proc.stderr.strip() or proc.stdout.strip() or f"Conversation subprocess exited with code {proc.returncode}."
        raise RuntimeError(message)
    stdout_text = str(proc.stdout or "").strip()
    if not stdout_text:
        stderr_text = str(proc.stderr or "").strip()
        raise RuntimeError(stderr_text or "Conversation subprocess produced no stdout.")
    try:
        parsed = json.loads(stdout_text)
    except Exception as exc:
        stderr_text = str(proc.stderr or "").strip()
        details = f"{exc}. stdout={stdout_text[:400]!r}"
        if stderr_text:
            details += f" stderr={stderr_text[:400]!r}"
        raise RuntimeError(f"Failed to parse conversation subprocess output: {details}") from exc
    responses = [str(item or "") for item in parsed.get("responses") or []]
    prompt_chars = sum(len(prompt or "") for prompt in prompts)
    response_chars = sum(len(item or "") for item in responses)
    system_prompt_chars = int(parsed.get("system_prompt_chars") or 0)
    tokens_estimated = estimate_tokens_from_chars(
        "x" * system_prompt_chars,
        "x" * prompt_chars,
        "x" * response_chars,
    )
    return {
        "responses": responses,
        "system_prompt_chars": system_prompt_chars,
        "prompt_chars": prompt_chars,
        "response_chars": response_chars,
        "tokens_estimated": tokens_estimated,
        "call_error": str(parsed.get("call_error") or ""),
        "agent_logs": str(parsed.get("agent_logs") or ""),
    }


def run_agent_prompt(
    *,
    repo_root: Path,
    hermes_home: Path,
    model: str,
    query: str,
    timeout_seconds: int,
    addon: bool,
    provider_config: Mapping[str, Any] | None = None,
    wsl_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "repo_root": str(repo_root),
        "hermes_home": str(hermes_home),
        "model": model,
        "query": query,
        "addon": bool(addon),
        "provider_config": dict(provider_config or {}),
    }
    script = _recall_script()
    if wsl_settings and wsl_settings.get("enabled"):
        proc = _run_wsl_script(
            script=script,
            payload=json.dumps(payload),
            repo_root=repo_root,
            hermes_home=hermes_home,
            wsl_settings=wsl_settings,
            timeout_seconds=timeout_seconds,
        )
    else:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(repo_root),
            timeout=timeout_seconds,
            env={**os.environ, "HERMES_HOME": str(hermes_home)},
        )
    if proc.returncode != 0:
        message = proc.stderr.strip() or proc.stdout.strip() or f"Prompt subprocess exited with code {proc.returncode}."
        raise RuntimeError(message)
    stdout_text = str(proc.stdout or "").strip()
    if not stdout_text:
        stderr_text = str(proc.stderr or "").strip()
        raise RuntimeError(stderr_text or "Prompt subprocess produced no stdout.")
    try:
        parsed = json.loads(stdout_text)
    except Exception as exc:
        stderr_text = str(proc.stderr or "").strip()
        details = f"{exc}. stdout={stdout_text[:400]!r}"
        if stderr_text:
            details += f" stderr={stderr_text[:400]!r}"
        raise RuntimeError(f"Failed to parse prompt subprocess output: {details}") from exc
    system_prompt_chars = int(parsed.get("system_prompt_chars") or 0)
    context_chars = int(parsed.get("context_chars") or 0)
    prompt_chars = int(parsed.get("prompt_chars") or 0)
    response_text = str(parsed.get("response_text") or "")
    tokens_estimated = estimate_tokens_from_chars(
        "x" * system_prompt_chars,
        "x" * prompt_chars,
        response_text,
    )
    return {
        "response_text": response_text,
        "call_error": str(parsed.get("call_error") or ""),
        "agent_logs": str(parsed.get("agent_logs") or ""),
        "system_prompt_chars": system_prompt_chars,
        "context_chars": context_chars,
        "prompt_chars": prompt_chars,
        "tokens_estimated": tokens_estimated,
    }


def _run_wsl_script(
    *,
    script: str,
    payload: str,
    repo_root: Path,
    hermes_home: Path,
    wsl_settings: Mapping[str, Any],
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    hermes_repo_root = str(wsl_settings.get("hermes_repo_root") or "~/.hermes/hermes-agent")
    python_bin = str(wsl_settings.get("python_bin") or default_wsl_python(hermes_repo_root))
    wsl_repo_root = windows_to_wsl_path(repo_root)
    wsl_home = windows_to_wsl_path(hermes_home)
    extra_env = {"BENCH_REPO_ROOT": wsl_repo_root}
    for env_name in (
        "CONSOLIDATING_MEMORY_LLM_API_KEY",
        "CONSOLIDATING_MEMORY_EMBEDDING_API_KEY",
    ):
        env_value = str(os.environ.get(env_name) or "").strip()
        if env_value:
            extra_env[env_name] = env_value
    return run_wsl_python(
        script=script,
        payload=payload,
        hermes_home=wsl_home,
        hermes_repo_root=hermes_repo_root,
        python_bin=python_bin,
        distro=str(wsl_settings.get("distro") or ""),
        extra_env=extra_env,
        timeout_seconds=timeout_seconds,
    )


def _recall_script() -> str:
    return textwrap.dedent(
        """
        import importlib
        import importlib.util
        import contextlib
        import io
        import json
        import os
        import sys
        from pathlib import Path

        payload = json.loads(sys.stdin.read())
        repo_root = Path(os.environ.get("BENCH_REPO_ROOT") or payload["repo_root"]).resolve()
        hermes_home = Path(os.environ.get("HERMES_HOME") or payload["hermes_home"]).resolve()
        sys.path.insert(0, str(repo_root))
        sys.path.insert(0, os.getcwd())

        def import_ai_agent():
            errors = []
            for module_name in ("hermes.agent", "run_agent"):
                try:
                    module = importlib.import_module(module_name)
                    return getattr(module, "AIAgent")
                except Exception as exc:
                    errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
            raise RuntimeError("Unable to import AIAgent. " + " | ".join(errors))

        def resolve_agent_runtime() -> dict:
            try:
                runtime_module = importlib.import_module("hermes_cli.runtime_provider")
                resolve_runtime_provider = getattr(runtime_module, "resolve_runtime_provider")
            except Exception as exc:
                raise RuntimeError(f"Unable to import Hermes runtime provider resolver: {type(exc).__name__}: {exc}")

            runtime = resolve_runtime_provider(
                requested=None,
                explicit_api_key=None,
                explicit_base_url=None,
            )
            if not isinstance(runtime, dict):
                raise RuntimeError(f"Unexpected runtime provider result: {type(runtime).__name__}")
            return runtime

        def load_provider_class():
            plugin_init = repo_root / "plugins" / "memory" / "consolidating_local" / "__init__.py"
            if not plugin_init.exists():
                raise RuntimeError(f"Workspace plugin not found: {plugin_init}")
            module_name = "bench_workspace_consolidating_local"
            spec = importlib.util.spec_from_file_location(
                module_name,
                str(plugin_init),
                submodule_search_locations=[str(plugin_init.parent)],
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Unable to load workspace plugin spec from {plugin_init}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            provider_cls = getattr(module, "ConsolidatingLocalProvider", None) or getattr(
                module, "ConsolidatingLocalMemoryProvider", None
            )
            if provider_cls is None:
                raise RuntimeError("Workspace plugin does not export a provider class.")
            return provider_cls

        def provider_context(session_id: str, query: str) -> str:
            if not payload.get("addon"):
                return ""
            provider_cls = load_provider_class()
            provider = provider_cls(payload.get("provider_config") or {})
            provider.initialize(session_id=session_id, hermes_home=str(hermes_home))
            try:
                getter = getattr(provider, "get_context", None)
                if callable(getter):
                    return str(getter(session_id=session_id, query=query) or "")
                return str(provider.prefetch(query, session_id=session_id) or "")
            finally:
                provider.shutdown()

        def extract_system_prompt(agent) -> str:
            for attr_name in ("_system_prompt", "_cached_system_prompt", "system_prompt"):
                value = getattr(agent, attr_name, "")
                if isinstance(value, str) and value:
                    return value
            for method_name in ("_build_system_prompt", "build_system_prompt"):
                method = getattr(agent, method_name, None)
                if not callable(method):
                    continue
                try:
                    result = method()
                except TypeError:
                    continue
                if isinstance(result, str):
                    return result
                if isinstance(result, (tuple, list)):
                    for item in result:
                        if isinstance(item, str):
                            return item
            return ""

        AIAgent = import_ai_agent()
        session_id = "bench-recall"
        runtime = resolve_agent_runtime()
        agent_kwargs = {
            "model": payload["model"],
            "provider": runtime.get("provider"),
            "base_url": runtime.get("base_url"),
            "api_key": runtime.get("api_key"),
            "api_mode": runtime.get("api_mode"),
            "skip_context_files": True,
            "session_id": session_id,
            "quiet_mode": True,
        }
        try:
            agent = AIAgent(**agent_kwargs)
        except TypeError:
            agent_kwargs.pop("quiet_mode", None)
            try:
                agent = AIAgent(**agent_kwargs)
            except TypeError:
                agent_kwargs.pop("skip_context_files", None)
                agent = AIAgent(**agent_kwargs)
        if hasattr(agent, "tools"):
            agent.tools = []
        if hasattr(agent, "valid_tool_names"):
            agent.valid_tool_names = []

        context = provider_context(session_id=session_id, query=payload["query"])
        prompt = payload["query"]
        if context:
            prompt = context + "\\n\\n" + payload["query"]

        response_text = ""
        call_error = ""
        log_buffer = io.StringIO()
        with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
            try:
                chat = getattr(agent, "chat", None)
                if callable(chat):
                    response_text = chat(prompt)
                else:
                    runner = getattr(agent, "run_conversation", None)
                    if not callable(runner):
                        raise RuntimeError("AIAgent has neither chat() nor run_conversation().")
                    response_text = runner(prompt)
            except Exception as exc:
                call_error = f"{type(exc).__name__}: {exc}"

        result = {
            "response_text": str(response_text or ""),
            "call_error": call_error,
            "agent_logs": log_buffer.getvalue(),
            "system_prompt_chars": len(extract_system_prompt(agent)),
            "context_chars": len(context),
            "prompt_chars": len(prompt),
        }
        print(json.dumps(result))
        """
    ).strip()


def _conversation_script() -> str:
    return textwrap.dedent(
        """
        import contextlib
        import importlib
        import importlib.util
        import io
        import json
        import os
        import sys
        from pathlib import Path

        payload = json.loads(sys.stdin.read())
        repo_root = Path(os.environ.get("BENCH_REPO_ROOT") or payload["repo_root"]).resolve()
        hermes_home = Path(os.environ.get("HERMES_HOME") or payload["hermes_home"]).resolve()
        sys.path.insert(0, str(repo_root))
        sys.path.insert(0, os.getcwd())

        def import_ai_agent():
            errors = []
            for module_name in ("hermes.agent", "run_agent"):
                try:
                    module = importlib.import_module(module_name)
                    return getattr(module, "AIAgent")
                except Exception as exc:
                    errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
            raise RuntimeError("Unable to import AIAgent. " + " | ".join(errors))

        def resolve_agent_runtime() -> dict:
            runtime_module = importlib.import_module("hermes_cli.runtime_provider")
            resolve_runtime_provider = getattr(runtime_module, "resolve_runtime_provider")
            runtime = resolve_runtime_provider(
                requested=None,
                explicit_api_key=None,
                explicit_base_url=None,
            )
            if not isinstance(runtime, dict):
                raise RuntimeError(f"Unexpected runtime provider result: {type(runtime).__name__}")
            return runtime

        def extract_system_prompt(agent) -> str:
            for attr_name in ("_system_prompt", "_cached_system_prompt", "system_prompt"):
                value = getattr(agent, attr_name, "")
                if isinstance(value, str) and value:
                    return value
            for method_name in ("_build_system_prompt", "build_system_prompt"):
                method = getattr(agent, method_name, None)
                if not callable(method):
                    continue
                try:
                    result = method()
                except TypeError:
                    continue
                if isinstance(result, str):
                    return result
                if isinstance(result, (tuple, list)):
                    for item in result:
                        if isinstance(item, str):
                            return item
            return ""

        def load_provider_class():
            plugin_init = repo_root / "plugins" / "memory" / "consolidating_local" / "__init__.py"
            if not plugin_init.exists():
                raise RuntimeError(f"Workspace plugin not found: {plugin_init}")
            module_name = "bench_workspace_consolidating_local"
            spec = importlib.util.spec_from_file_location(
                module_name,
                str(plugin_init),
                submodule_search_locations=[str(plugin_init.parent)],
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Unable to load workspace plugin spec from {plugin_init}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            provider_cls = getattr(module, "ConsolidatingLocalProvider", None) or getattr(
                module, "ConsolidatingLocalMemoryProvider", None
            )
            if provider_cls is None:
                raise RuntimeError("Workspace plugin does not export a provider class.")
            return provider_cls

        runtime = resolve_agent_runtime()
        AIAgent = import_ai_agent()
        session_id = str(payload.get("session_id") or "bench-session")
        agent_kwargs = {
            "model": payload["model"],
            "provider": runtime.get("provider"),
            "base_url": runtime.get("base_url"),
            "api_key": runtime.get("api_key"),
            "api_mode": runtime.get("api_mode"),
            "skip_context_files": True,
            "session_id": session_id,
            "quiet_mode": True,
        }
        try:
            agent = AIAgent(**agent_kwargs)
        except TypeError:
            agent_kwargs.pop("quiet_mode", None)
            try:
                agent = AIAgent(**agent_kwargs)
            except TypeError:
                agent_kwargs.pop("skip_context_files", None)
                agent = AIAgent(**agent_kwargs)

        provider = None
        messages = []
        if payload.get("addon") and payload.get("sync_provider_turns"):
            provider_cls = load_provider_class()
            provider = provider_cls(payload.get("provider_config") or {})
            provider.initialize(session_id=session_id, hermes_home=str(hermes_home))

        responses = []
        call_error = ""
        log_buffer = io.StringIO()
        with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
            try:
                for prompt in payload.get("prompts") or []:
                    user_prompt = str(prompt or "")
                    reply = ""
                    chat = getattr(agent, "chat", None)
                    if callable(chat):
                        reply = chat(user_prompt)
                    else:
                        runner = getattr(agent, "run_conversation", None)
                        if not callable(runner):
                            raise RuntimeError("AIAgent has neither chat() nor run_conversation().")
                        reply = runner(user_prompt)
                    reply_text = str(reply or "")
                    responses.append(reply_text)
                    messages.append({"role": "user", "content": user_prompt})
                    messages.append({"role": "assistant", "content": reply_text})
                    if provider is not None:
                        provider.sync_turn(
                            user_content=user_prompt,
                            assistant_content=reply_text,
                            session_id=session_id,
                        )
                if provider is not None and payload.get("finalize_provider_session"):
                    provider.on_session_end(messages)
            except Exception as exc:
                call_error = f"{type(exc).__name__}: {exc}"
            finally:
                if provider is not None:
                    provider.shutdown()

        result = {
            "responses": responses,
            "call_error": call_error,
            "agent_logs": log_buffer.getvalue(),
            "system_prompt_chars": len(extract_system_prompt(agent)),
        }
        print(json.dumps(result))
        """
    ).strip()


def _normalize_array(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"Expected JSON array, got {type(value).__name__}")
    return [str(item).strip() for item in value if str(item).strip()]


def _summarize_logs(logs: str, limit: int = 320) -> str:
    pieces = [line.strip() for line in str(logs or "").splitlines() if line.strip()]
    if not pieces:
        return ""
    summary = " | ".join(pieces[-6:])
    if len(summary) > limit:
        return summary[: limit - 3] + "..."
    return summary
