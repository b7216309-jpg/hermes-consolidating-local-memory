from __future__ import annotations

import os
import shlex
import subprocess
import json
from functools import lru_cache
from pathlib import Path
from typing import Any


def windows_to_wsl_path(path: str | Path) -> str:
    raw = str(path)
    if raw.startswith("/"):
        return raw
    resolved = Path(raw).expanduser().resolve()
    drive = resolved.drive.rstrip(":").lower()
    tail = resolved.as_posix().split(":/", 1)[1] if ":/" in resolved.as_posix() else resolved.as_posix()
    return f"/mnt/{drive}/{tail}"


def default_wsl_python(hermes_root: str) -> str:
    clean = hermes_root.rstrip("/")
    return f"{clean}/.venv/bin/python"


def build_wsl_command(command: str, *, distro: str = "") -> list[str]:
    args = ["wsl.exe"]
    if distro:
        args.extend(["-d", distro])
    args.extend(["bash", "-lc", command])
    return args


@lru_cache(maxsize=8)
def detect_wsl_home(distro: str = "") -> str:
    proc = subprocess.run(
        build_wsl_command('printf "%s" "$HOME"', distro=distro),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
        env=os.environ.copy(),
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(proc.stderr.strip() or "Unable to detect WSL home directory.")
    return proc.stdout.strip()


def run_wsl_python(
    *,
    script: str,
    payload: str,
    hermes_home: str,
    hermes_repo_root: str,
    python_bin: str,
    distro: str = "",
    extra_env: dict[str, str] | None = None,
    timeout_seconds: int = 180,
) -> subprocess.CompletedProcess[str]:
    home = detect_wsl_home(distro)
    hermes_repo_root = expand_wsl_user_path(hermes_repo_root, home=home)
    python_bin = expand_wsl_user_path(python_bin, home=home)
    env_parts = [f"export HERMES_HOME={shlex.quote(hermes_home)}"]
    for key, value in (extra_env or {}).items():
        env_parts.append(f"export {key}={shlex.quote(value)}")
    env_parts.append(f"cd {shlex.quote(hermes_repo_root)}")
    env_parts.append(f"{shlex.quote(python_bin)} -c {shlex.quote(script)}")
    command = " && ".join(env_parts)
    return subprocess.run(
        build_wsl_command(command, distro=distro),
        input=payload,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        env=os.environ.copy(),
    )


def expand_wsl_user_path(path: str, *, home: str) -> str:
    if path.startswith("~/"):
        return f"{home}/{path[2:]}"
    return path


def wsl_to_unc_path(path: str, *, distro: str) -> str:
    if not path.startswith("/"):
        raise ValueError(f"Expected absolute WSL path, got: {path}")
    tail = path.lstrip("/").replace("/", "\\")
    return f"\\\\wsl$\\{distro}\\{tail}"


def collect_wsl_runtime_seed(distro: str = "Ubuntu") -> dict[str, Any]:
    home = detect_wsl_home(distro)
    hermes_home = f"{home}/.hermes"
    config_path = Path(wsl_to_unc_path(f"{hermes_home}/config.yaml", distro=distro))
    auth_path = Path(wsl_to_unc_path(f"{hermes_home}/auth.json", distro=distro))
    env_path = Path(wsl_to_unc_path(f"{hermes_home}/.env", distro=distro))
    model_config = {"default": "", "provider": "", "base_url": ""}
    if config_path.exists():
        in_model = False
        for raw_line in config_path.read_text(encoding="utf-8").splitlines():
            if raw_line.startswith("model:"):
                in_model = True
                continue
            if in_model and not raw_line.startswith("  "):
                break
            if in_model and ":" in raw_line:
                key, value = raw_line.strip().split(":", 1)
                if key in model_config:
                    model_config[key] = value.strip().strip('"').strip("'")
    return {
        "source_home": hermes_home,
        "source_config_path": str(config_path),
        "source_auth_path": str(auth_path),
        "source_env_path": str(env_path),
        "model_config": model_config,
    }


def collect_wsl_runtime_credentials(
    *,
    distro: str = "Ubuntu",
    hermes_repo_root: str = "~/.hermes/hermes-agent",
    python_bin: str = "",
) -> dict[str, str]:
    home = detect_wsl_home(distro)
    hermes_home = f"{home}/.hermes"
    clean_python = python_bin or default_wsl_python(hermes_repo_root)
    script = """
import json
from hermes_cli.runtime_provider import resolve_runtime_provider
runtime = resolve_runtime_provider(
    requested=None,
    explicit_api_key=None,
    explicit_base_url=None,
)
print(json.dumps({
    "provider": str(runtime.get("provider") or ""),
    "base_url": str(runtime.get("base_url") or ""),
    "api_key": str(runtime.get("api_key") or ""),
    "api_mode": str(runtime.get("api_mode") or ""),
}))
""".strip()
    proc = run_wsl_python(
        script=script,
        payload="",
        hermes_home=hermes_home,
        hermes_repo_root=hermes_repo_root,
        python_bin=clean_python,
        distro=distro,
        timeout_seconds=30,
    )
    if proc.returncode != 0:
        message = proc.stderr.strip() or proc.stdout.strip() or "Unable to resolve WSL runtime credentials."
        raise RuntimeError(message)
    try:
        data = json.loads(proc.stdout.strip())
    except Exception as exc:
        raise RuntimeError(f"Unable to parse WSL runtime credentials: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected WSL runtime credentials payload: {type(data).__name__}")
    return {key: str(data.get(key) or "") for key in ("provider", "base_url", "api_key", "api_mode")}
