from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List

from .store import MemoryStore, normalize_whitespace, slugify

PREFERENCE_HINTS = (
    "prefer",
    "prefers",
    "do not",
    "don't",
    "avoid",
    "always",
    "never",
    "dislike",
    "likes",
    "keep responses",
    "be concise",
    "bullet",
    "paragraph",
)

PROJECT_HINTS = (
    "project",
    "repo",
    "repository",
    "service",
    "database",
    "deployment",
    "deploy",
    "ci",
    "pipeline",
    "stack",
    "backend",
    "frontend",
    "tests",
)

ENVIRONMENT_HINTS = (
    "ubuntu",
    "debian",
    "macos",
    "windows",
    "wsl",
    "docker",
    "podman",
    "python",
    "node",
    "shell",
    "editor",
    "ssh",
    "port",
    "timezone",
)

WORKFLOW_HINTS = (
    "must",
    "should",
    "remember",
    "needs",
    "required",
    "workflow",
    "convention",
    "run tests",
)

STYLE_VALUE_MAP = {
    "concise": "concise",
    "brief": "concise",
    "short": "concise",
    "terse": "concise",
    "detailed": "detailed",
    "verbose": "detailed",
    "long-form": "detailed",
    "long form": "detailed",
}

SHELL_VALUES = ("bash", "zsh", "fish", "powershell")

EDITOR_ALIASES = {
    "vs code": "VS Code",
    "vscode": "VS Code",
    "visual studio code": "VS Code",
    "neovim": "Neovim",
    "vim": "Vim",
    "emacs": "Emacs",
    "zed": "Zed",
    "pycharm": "PyCharm",
    "intellij": "IntelliJ",
    "jetbrains": "JetBrains IDE",
}

OS_PATTERNS = [
    ("ubuntu", "Ubuntu"),
    ("debian", "Debian"),
    ("fedora", "Fedora"),
    ("arch linux", "Arch Linux"),
    ("macos", "macOS"),
    ("windows", "Windows"),
    ("linux", "Linux"),
]

TECH_PATTERNS = [
    (r"\bfastapi\b", "Project uses FastAPI.", "project-stack"),
    (r"\bdjango\b", "Project uses Django.", "project-stack"),
    (r"\bflask\b", "Project uses Flask.", "project-stack"),
    (r"\breact\b", "Project uses React.", "project-stack"),
    (r"\bvue\b", "Project uses Vue.", "project-stack"),
    (r"\btypescript\b", "Project uses TypeScript.", "project-stack"),
    (r"\bjavascript\b", "Project uses JavaScript.", "project-stack"),
    (r"\bpython\b", "Project uses Python.", "project-stack"),
    (r"\bgo\b|\bgolang\b", "Project uses Go.", "project-stack"),
    (r"\bpostgres(?:ql)?\b", "Project uses PostgreSQL.", "project-data"),
    (r"\bmysql\b", "Project uses MySQL.", "project-data"),
    (r"\bsqlite\b", "Project uses SQLite.", "project-data"),
    (r"\bredis\b", "Project uses Redis.", "project-data"),
    (r"\bdocker compose\b", "Project uses Docker Compose.", "project-delivery"),
    (r"\bkubernetes\b|\bk8s\b", "Project uses Kubernetes.", "project-delivery"),
]


def absolutize_relative_dates(text: str, ref_ts: float | None = None) -> str:
    if not text:
        return ""
    anchor = datetime.fromtimestamp(ref_ts or time.time())
    replacements = {
        "today": anchor.strftime("%Y-%m-%d"),
        "yesterday": (anchor - timedelta(days=1)).strftime("%Y-%m-%d"),
        "tomorrow": (anchor + timedelta(days=1)).strftime("%Y-%m-%d"),
        "last week": f"week of {(anchor - timedelta(days=7)).strftime('%Y-%m-%d')}",
        "next week": f"week of {(anchor + timedelta(days=7)).strftime('%Y-%m-%d')}",
    }
    updated = text
    for needle, replacement in replacements.items():
        updated = re.sub(rf"\b{re.escape(needle)}\b", replacement, updated, flags=re.IGNORECASE)
    return updated


def normalize_sentence(text: str) -> str:
    clean = normalize_whitespace(text)
    if not clean:
        return ""
    clean = clean.rstrip(" .;")
    if clean and clean[-1] not in ".!?":
        clean += "."
    return clean


def build_candidate(
    *,
    content: str,
    category: str,
    topic: str,
    importance: int,
    confidence: float,
    source_role: str,
    subject_key: str = "",
    value_key: str = "",
    exclusive: bool = False,
    polarity: int = 1,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    merged_metadata = dict(metadata or {})
    merged_metadata["source_role"] = source_role
    if subject_key:
        merged_metadata["subject_key"] = subject_key
    if value_key:
        merged_metadata["value_key"] = value_key
    if exclusive:
        merged_metadata["exclusive"] = True
    if polarity != 1:
        merged_metadata["polarity"] = polarity
    return {
        "content": normalize_sentence(content),
        "category": category,
        "topic": topic,
        "importance": importance,
        "confidence": confidence,
        "metadata": merged_metadata,
    }


def clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return default
    return max(low, min(high, parsed))


def clamp_float(value: Any, low: float, high: float, default: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    return max(low, min(high, parsed))


def infer_category(sentence: str) -> str:
    lower = sentence.lower()
    if any(hint in lower for hint in PREFERENCE_HINTS):
        return "user_pref"
    if any(hint in lower for hint in PROJECT_HINTS):
        return "project"
    if any(hint in lower for hint in ENVIRONMENT_HINTS):
        return "environment"
    if any(hint in lower for hint in WORKFLOW_HINTS):
        return "workflow"
    return "general"


def infer_topic(sentence: str, category: str, subject_key: str = "") -> str:
    if subject_key.startswith("user:"):
        return "user-profile"
    if subject_key.startswith("environment:"):
        return "environment"
    if subject_key.startswith("workflow:"):
        return "workflow-rules"
    if subject_key.startswith("project:"):
        suffix = subject_key.split(":", 1)[1]
        return slugify(f"project-{suffix}")
    lower = sentence.lower()
    if category == "user_pref":
        return "user-profile"
    if category == "workflow":
        return "workflow-rules"
    if category == "environment":
        return "environment"
    if category == "project":
        if "deploy" in lower or "ci" in lower or "test" in lower:
            return "project-delivery"
        if any(token in lower for token in ("postgres", "mysql", "sqlite", "redis", "database")):
            return "project-data"
        return "project-context"
    return "general"


def infer_importance(sentence: str, category: str, *, structured: bool = False) -> int:
    lower = sentence.lower()
    if any(token in lower for token in ("remember", "must", "never", "do not", "don't")):
        return 9
    if structured and category in {"user_pref", "workflow"}:
        return 8
    if category in {"user_pref", "workflow"}:
        return 7
    if category in {"project", "environment"}:
        return 6
    return 4


def split_candidate_segments(text: str) -> List[str]:
    cleaned = text.replace("\r", "\n")
    segments: List[str] = []
    for raw_line in cleaned.split("\n"):
        line = normalize_whitespace(re.sub(r"^[>\-*0-9.)\s]+", "", raw_line or ""))
        if not line:
            continue
        parts = re.split(r"(?<=[.!?;])\s+|\s+\bbut\b\s+|\s+\bhowever\b\s+", line, flags=re.IGNORECASE)
        for part in parts:
            part = normalize_whitespace(part)
            if not part:
                continue
            if len(part) < 8 or len(part) > 260:
                continue
            segments.append(part)
    return segments


def is_memory_worthy(sentence: str) -> bool:
    lower = sentence.lower()
    return any(
        hint in lower
        for hint in (PREFERENCE_HINTS + PROJECT_HINTS + ENVIRONMENT_HINTS + WORKFLOW_HINTS)
    )


def _extract_response_style(segment: str, source_role: str) -> List[Dict[str, Any]]:
    lower = segment.lower()
    match = re.search(
        r"\b(concise|brief|short|terse|detailed|verbose|long[- ]form)\b",
        lower,
    )
    if not match:
        return []
    if not (
        "response" in lower
        or "answer" in lower
        or "explanation" in lower
        or any(hint in lower for hint in ("prefer", "keep", "like", "want", "be concise"))
    ):
        return []
    raw_value = match.group(1)
    value = STYLE_VALUE_MAP.get(raw_value, raw_value)
    return [
        build_candidate(
            content=f"User prefers {value} responses",
            category="user_pref",
            topic="user-profile",
            importance=8,
            confidence=0.8 if source_role == "user" else 0.6,
            source_role=source_role,
            subject_key="user:response_style",
            value_key=value,
            exclusive=True,
        )
    ]


def _extract_answer_format(segment: str, source_role: str) -> List[Dict[str, Any]]:
    lower = segment.lower()
    if not any(hint in lower for hint in ("prefer", "likes", "like", "want")):
        return []
    if "bullet" in lower:
        return [
            build_candidate(
                content="User prefers bullet-point answers",
                category="user_pref",
                topic="user-profile",
                importance=7,
                confidence=0.8 if source_role == "user" else 0.6,
                source_role=source_role,
                subject_key="user:answer_format",
                value_key="bullet-points",
            )
        ]
    if "paragraph" in lower:
        return [
            build_candidate(
                content="User prefers paragraph-style answers",
                category="user_pref",
                topic="user-profile",
                importance=7,
                confidence=0.8 if source_role == "user" else 0.6,
                source_role=source_role,
                subject_key="user:answer_format",
                value_key="paragraphs",
            )
        ]
    return []


def _extract_user_role(segment: str, source_role: str) -> List[Dict[str, Any]]:
    match = re.search(
        r"\b(?:i am|i'm|the user is|user is)\s+(?:a|an)\s+([a-z][a-z0-9 /+-]{2,40})\b",
        segment,
        flags=re.IGNORECASE,
    )
    if not match:
        return []
    role = normalize_whitespace(match.group(1))
    return [
        build_candidate(
            content=f"User role: {role}",
            category="user_pref",
            topic="user-profile",
            importance=7,
            confidence=0.85 if source_role == "user" else 0.6,
            source_role=source_role,
            subject_key="user:role",
            value_key=slugify(role),
            exclusive=True,
        )
    ]


def _extract_timezone(segment: str, source_role: str) -> List[Dict[str, Any]]:
    match = re.search(
        r"\b(?:timezone(?: is)?|i(?:'m| am) in|we(?:'re| are) in)\s+([A-Za-z]+(?:/[A-Za-z_]+)?|UTC[+-]?\d{0,2}|CET|CEST|PST|EST|EDT|BST)\b",
        segment,
        flags=re.IGNORECASE,
    )
    if not match:
        return []
    value = normalize_whitespace(match.group(1))
    return [
        build_candidate(
            content=f"User timezone: {value}",
            category="user_pref",
            topic="user-profile",
            importance=7,
            confidence=0.85 if source_role == "user" else 0.6,
            source_role=source_role,
            subject_key="user:timezone",
            value_key=slugify(value),
            exclusive=True,
        )
    ]


def _extract_shell(segment: str, source_role: str) -> List[Dict[str, Any]]:
    lower = segment.lower()
    if "shell" not in lower:
        return []
    for shell in SHELL_VALUES:
        if re.search(rf"\b{re.escape(shell)}\b", lower):
            return [
                build_candidate(
                    content=f"Environment shell is {shell}",
                    category="environment",
                    topic="environment",
                    importance=7,
                    confidence=0.8 if source_role == "user" else 0.6,
                    source_role=source_role,
                    subject_key="environment:shell",
                    value_key=shell,
                    exclusive=True,
                )
            ]
    return []


def _extract_editor(segment: str, source_role: str) -> List[Dict[str, Any]]:
    lower = segment.lower()
    if "editor" not in lower and not re.search(r"\b(?:i use|using)\b", lower):
        return []
    for alias, label in EDITOR_ALIASES.items():
        if alias in lower:
            return [
                build_candidate(
                    content=f"Environment editor is {label}",
                    category="environment",
                    topic="environment",
                    importance=6,
                    confidence=0.8 if source_role == "user" else 0.6,
                    source_role=source_role,
                    subject_key="environment:editor",
                    value_key=slugify(label),
                )
            ]
    return []


def _extract_operating_system(segment: str, source_role: str) -> List[Dict[str, Any]]:
    lower = segment.lower()
    candidates: List[Dict[str, Any]] = []
    if "wsl2" in lower or "wsl 2" in lower or "wsl" in lower:
        candidates.append(
            build_candidate(
                content="Environment uses WSL2",
                category="environment",
                topic="environment",
                importance=6,
                confidence=0.8 if source_role == "user" else 0.6,
                source_role=source_role,
                subject_key="environment:wsl",
                value_key="wsl2",
            )
        )
    for needle, label in OS_PATTERNS:
        if needle not in lower:
            continue
        version_match = re.search(
            rf"{re.escape(needle)}\s+([0-9]+(?:\.[0-9]+)*)",
            lower,
        )
        value = label if not version_match else f"{label} {version_match.group(1)}"
        candidates.append(
            build_candidate(
                content=f"Environment runs {value}",
                category="environment",
                topic="environment",
                importance=7,
                confidence=0.8 if source_role == "user" else 0.6,
                source_role=source_role,
                subject_key="environment:os",
                value_key=slugify(value),
                exclusive=True,
            )
        )
        break
    return candidates


def _extract_docker_rule(segment: str, source_role: str) -> List[Dict[str, Any]]:
    lower = segment.lower()
    if "docker" not in lower:
        return []
    if any(token in lower for token in ("do not use sudo", "don't use sudo", "no sudo")):
        return [
            build_candidate(
                content="Do not use sudo for Docker commands",
                category="workflow",
                topic="workflow-rules",
                importance=9,
                confidence=0.85 if source_role == "user" else 0.6,
                source_role=source_role,
                subject_key="workflow:docker_sudo",
                value_key="no-sudo",
                exclusive=True,
            )
        ]
    if "use sudo" in lower and "docker" in lower:
        return [
            build_candidate(
                content="Use sudo for Docker commands",
                category="workflow",
                topic="workflow-rules",
                importance=9,
                confidence=0.85 if source_role == "user" else 0.6,
                source_role=source_role,
                subject_key="workflow:docker_sudo",
                value_key="sudo-required",
                exclusive=True,
            )
        ]
    return []


def _extract_ssh_port(segment: str, source_role: str) -> List[Dict[str, Any]]:
    lower = segment.lower()
    if "port" not in lower:
        return []
    port_match = re.search(r"\bport\s+(\d{2,5})\b", lower)
    if not port_match:
        return []
    port = port_match.group(1)
    subject_key = "environment:ssh_port" if "ssh" in lower else "environment:service_port"
    label = "SSH uses port" if "ssh" in lower else "Primary service uses port"
    return [
        build_candidate(
            content=f"{label} {port}",
            category="environment",
            topic="environment",
            importance=8 if "ssh" in lower else 6,
            confidence=0.85 if source_role == "user" else 0.6,
            source_role=source_role,
            subject_key=subject_key,
            value_key=port,
            exclusive=True,
        )
    ]


def _extract_key_path(segment: str, source_role: str) -> List[Dict[str, Any]]:
    match = re.search(r"\bkey is at\s+([~/A-Za-z0-9._/\-]+)\b", segment, flags=re.IGNORECASE)
    if not match:
        return []
    path = match.group(1)
    return [
        build_candidate(
            content=f"SSH key path: {path}",
            category="environment",
            topic="environment",
            importance=6,
            confidence=0.85 if source_role == "user" else 0.55,
            source_role=source_role,
            subject_key="environment:ssh_key_path",
            value_key=slugify(path),
        )
    ]


def _extract_test_command(segment: str, source_role: str) -> List[Dict[str, Any]]:
    match = re.search(
        r"\brun tests?\s+with\s+[\"'`]?([^\"'`]+?)(?:[\"'`]|$)",
        segment,
        flags=re.IGNORECASE,
    )
    if not match:
        return []
    command = normalize_whitespace(match.group(1)).rstrip(".")
    return [
        build_candidate(
            content=f"Project tests run with {command}",
            category="project",
            topic="project-delivery",
            importance=8,
            confidence=0.85 if source_role == "user" else 0.6,
            source_role=source_role,
            subject_key="project:test_command",
            value_key=slugify(command),
            exclusive=True,
        )
    ]


def _extract_deploy_method(segment: str, source_role: str) -> List[Dict[str, Any]]:
    match = re.search(
        r"\bdeploy(?:s|ed)? with\s+([A-Za-z0-9 _./-]+?)(?:[.;,]|$)",
        segment,
        flags=re.IGNORECASE,
    )
    if not match:
        return []
    method = normalize_whitespace(match.group(1))
    return [
        build_candidate(
            content=f"Project deploys with {method}",
            category="project",
            topic="project-delivery",
            importance=7,
            confidence=0.8 if source_role == "user" else 0.6,
            source_role=source_role,
            subject_key="project:deploy_method",
            value_key=slugify(method),
        )
    ]


def _extract_database_identity(segment: str, source_role: str) -> List[Dict[str, Any]]:
    match = re.search(
        r"\b(?:database|db)\s+(?:is|:)\s*(postgres(?:ql)?|mysql|sqlite|redis)\b",
        segment,
        flags=re.IGNORECASE,
    )
    if not match:
        return []
    raw = match.group(1).lower()
    label = "PostgreSQL" if raw.startswith("postgres") else raw.upper() if raw == "db" else raw.capitalize()
    normalized = "postgresql" if raw.startswith("postgres") else raw
    return [
        build_candidate(
            content=f"Primary project database is {label}",
            category="project",
            topic="project-data",
            importance=8,
            confidence=0.85 if source_role == "user" else 0.6,
            source_role=source_role,
            subject_key="project:database",
            value_key=normalized,
            exclusive=True,
        )
    ]


def _extract_project_techs(segment: str, source_role: str) -> List[Dict[str, Any]]:
    lower = segment.lower()
    if not (
        any(hint in lower for hint in PROJECT_HINTS)
        or " uses " in f" {lower} "
        or " with " in f" {lower} "
    ):
        return []
    candidates: List[Dict[str, Any]] = []
    for pattern, content, topic in TECH_PATTERNS:
        if re.search(pattern, lower, flags=re.IGNORECASE):
            candidates.append(
                build_candidate(
                    content=content,
                    category="project",
                    topic=topic,
                    importance=6,
                    confidence=0.75 if source_role == "user" else 0.55,
                    source_role=source_role,
                )
            )
    if "redis" in lower and "cache" in lower:
        candidates.append(
            build_candidate(
                content="Project uses Redis for caching",
                category="project",
                topic="project-data",
                importance=7,
                confidence=0.8 if source_role == "user" else 0.55,
                source_role=source_role,
                subject_key="project:cache_backend",
                value_key="redis",
            )
        )
    return candidates


def _extract_generic_rule(segment: str, source_role: str) -> List[Dict[str, Any]]:
    lower = segment.lower()
    if not any(token in lower for token in ("remember", "must", "should", "always", "never", "do not", "don't")):
        return []
    category = infer_category(segment)
    return [
        build_candidate(
            content=segment,
            category=category,
            topic=infer_topic(segment, category),
            importance=infer_importance(segment, category, structured=True),
            confidence=0.75 if source_role == "user" else 0.55,
            source_role=source_role,
        )
    ]


def _structured_candidates(segment: str, source_role: str) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    extractors = (
        _extract_response_style,
        _extract_answer_format,
        _extract_user_role,
        _extract_timezone,
        _extract_shell,
        _extract_editor,
        _extract_operating_system,
        _extract_docker_rule,
        _extract_ssh_port,
        _extract_key_path,
        _extract_test_command,
        _extract_deploy_method,
        _extract_database_identity,
        _extract_project_techs,
        _extract_generic_rule,
    )
    for extractor in extractors:
        candidates.extend(extractor(segment, source_role))
    return candidates


def _dedupe_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for candidate in candidates:
        metadata = dict(candidate.get("metadata") or {})
        key = (
            candidate["content"].lower(),
            metadata.get("subject_key", ""),
            metadata.get("value_key", ""),
            metadata.get("polarity", 1),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def extract_candidate_facts_from_turn(
    *,
    user_content: str,
    assistant_content: str,
    created_at: float | None = None,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for source_role, text in (("user", user_content), ("assistant", assistant_content)):
        for segment in split_candidate_segments(text or ""):
            normalized = absolutize_relative_dates(segment, created_at)
            structured = _structured_candidates(normalized, source_role)
            if structured:
                candidates.extend(structured)
                continue
            if source_role == "assistant" and "remember" not in normalized.lower():
                continue
            if source_role == "user" and not is_memory_worthy(normalized):
                if not re.search(r"\b(i|my|we|our)\b", normalized.lower()):
                    continue
            category = infer_category(normalized)
            candidates.append(
                build_candidate(
                    content=normalized,
                    category=category,
                    topic=infer_topic(normalized, category),
                    importance=infer_importance(normalized, category),
                    confidence=0.7 if source_role == "user" else 0.55,
                    source_role=source_role,
                )
            )
    return _dedupe_candidates(candidates)


def extract_candidate_facts_from_messages(messages: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or message.get("type") or "")
        content = message.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(block.get("text", "")) for block in content if isinstance(block, dict))
        text = str(content)
        if not text:
            continue
        source_role = "assistant" if "assistant" in role else "user"
        if source_role == "assistant" and "remember" not in text.lower():
            continue
        candidates.extend(
            extract_candidate_facts_from_turn(
                user_content=text if source_role == "user" else "",
                assistant_content=text if source_role == "assistant" else "",
            )
        )
    return _dedupe_candidates(candidates)


def normalize_candidate_fact(raw: Dict[str, Any], *, source_role: str = "assistant") -> Dict[str, Any] | None:
    content = normalize_sentence(str(raw.get("content") or ""))
    if not content:
        return None
    category = str(raw.get("category") or infer_category(content)).strip().lower()
    if category not in {"user_pref", "project", "environment", "workflow", "general"}:
        category = infer_category(content)
    subject_key = normalize_whitespace(str(raw.get("subject_key") or ""))
    value_key = normalize_whitespace(str(raw.get("value_key") or ""))
    exclusive = bool(raw.get("exclusive")) if subject_key else False
    polarity_raw = raw.get("polarity", 1)
    polarity = -1 if str(polarity_raw).strip() in {"-1", "false", "neg"} else 1
    topic = str(raw.get("topic") or infer_topic(content, category, subject_key)).strip()
    if not topic:
        topic = infer_topic(content, category, subject_key)
    return build_candidate(
        content=content,
        category=category,
        topic=topic,
        importance=clamp_int(raw.get("importance"), 1, 10, infer_importance(content, category, structured=bool(subject_key))),
        confidence=clamp_float(raw.get("confidence"), 0.05, 1.0, 0.6 if source_role == "assistant" else 0.75),
        source_role=source_role,
        subject_key=subject_key,
        value_key=value_key,
        exclusive=exclusive,
        polarity=polarity,
        metadata=dict(raw.get("metadata") or {}),
    )


def build_consolidation_plan(store: MemoryStore, *, min_hours: int, min_sessions: int) -> Dict[str, Any]:
    last_at = float(store.get_state("last_consolidated_at", "0") or 0)
    last_episode_id = int(store.get_state("last_consolidated_episode_id", "0") or 0)
    hours_since = (time.time() - last_at) / 3600 if last_at else float("inf")
    pending_sessions = store.sessions_since_episode(last_episode_id)
    pending_episodes = store.latest_episode_id() - last_episode_id
    return {
        "last_consolidated_at": last_at,
        "last_consolidated_episode_id": last_episode_id,
        "hours_since_last": None if hours_since == float("inf") else round(hours_since, 2),
        "pending_sessions": int(max(pending_sessions, 0)),
        "pending_episodes": int(max(pending_episodes, 0)),
        "min_hours": int(min_hours),
        "min_sessions": int(min_sessions),
        "should_run": (hours_since == float("inf") or hours_since >= min_hours)
        and pending_sessions >= min_sessions,
    }


def run_consolidation(
    store: MemoryStore,
    *,
    min_hours: int,
    min_sessions: int,
    max_topic_facts: int,
    topic_summary_chars: int,
    prune_after_days: int,
    extractor=None,
    force: bool = False,
    reason: str = "auto",
) -> Dict[str, Any]:
    started_at = time.time()
    plan = build_consolidation_plan(store, min_hours=min_hours, min_sessions=min_sessions)
    if not force and not plan["should_run"]:
        plan["status"] = "skipped"
        return plan

    last_episode_id = int(plan["last_consolidated_episode_id"])
    episodes = store.episodes_since_episode(last_episode_id)
    facts_added = 0
    facts_updated = 0
    facts_superseded = 0
    contradictions_resolved = 0

    for episode in episodes:
        extract = extractor or extract_candidate_facts_from_turn
        for candidate in extract(
            user_content=str(episode.get("user_content", "")),
            assistant_content=str(episode.get("assistant_content", "")),
            created_at=float(episode.get("created_at") or started_at),
        ):
            result = store.upsert_fact(
                content=str(candidate["content"]),
                category=str(candidate["category"]),
                topic=str(candidate["topic"]),
                source="episode_extract",
                importance=int(candidate["importance"]),
                confidence=float(candidate["confidence"]),
                metadata=dict(candidate.get("metadata") or {}),
                observed_at=float(episode.get("created_at") or started_at),
            )
            if result["action"] == "inserted":
                facts_added += 1
            else:
                facts_updated += 1
            facts_superseded += len(result.get("superseded", []))
            contradictions_resolved += len(result.get("contradictions", []))

    pruned = store.prune_stale_facts(max_age_days=prune_after_days)
    topics_rebuilt = store.rebuild_topics(
        max_facts=max_topic_facts,
        max_chars=topic_summary_chars,
    )
    latest_episode_id = store.latest_episode_id()
    finished_at = time.time()

    stats = {
        "status": "completed",
        "reason": reason,
        "episodes_scanned": len(episodes),
        "facts_added": facts_added,
        "facts_updated": facts_updated,
        "facts_superseded": facts_superseded,
        "contradictions_resolved": contradictions_resolved,
        "facts_pruned": pruned,
        "topics_rebuilt": topics_rebuilt,
        "latest_episode_id": latest_episode_id,
        "counts": store.counts(),
        "duration_seconds": round(finished_at - started_at, 3),
    }
    store.record_consolidation(
        reason=reason,
        started_at=started_at,
        finished_at=finished_at,
        source_episode_id=latest_episode_id,
        stats=stats,
    )
    return stats
