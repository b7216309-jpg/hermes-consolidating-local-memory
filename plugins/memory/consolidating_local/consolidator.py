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

PERSONAL_DETAIL_HINTS = (
    "favorite",
    "vegetarian",
    "vegan",
    "pescatarian",
    "gluten-free",
    "gluten free",
    "allergic to",
    "pronouns",
    "i'm from",
    "i am from",
    "i live in",
    "i'm based in",
    "i am based in",
    "i grew up in",
    "single",
    "married",
    "engaged",
    "divorced",
    "widowed",
)

LIFE_EVENT_HINTS = (
    "birthday",
    "wedding",
    "married",
    "getting married",
    "engaged",
    "graduat",
    "vacation",
    "holiday",
    "trip",
    "flight",
    "move",
    "moving",
    "relocat",
    "baby",
    "pregnan",
    "anniversary",
    "funeral",
    "surgery",
    "appointment",
    "concert",
)

TIME_REFERENCE_HINTS = (
    "today",
    "yesterday",
    "tomorrow",
    "tonight",
    "this week",
    "next week",
    "last week",
    "this month",
    "next month",
    "last month",
    "this year",
    "next year",
    "last year",
    "this weekend",
    "next weekend",
    "last weekend",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)

RELATION_HINTS = (
    "my sister",
    "my brother",
    "my mom",
    "my mother",
    "my dad",
    "my father",
    "my wife",
    "my husband",
    "my partner",
    "my girlfriend",
    "my boyfriend",
    "my fiance",
    "my fiancee",
    "my son",
    "my daughter",
    "my kid",
    "my child",
    "my children",
    "my parents",
    "my family",
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
    if any(hint in lower for hint in PERSONAL_DETAIL_HINTS):
        return "user_pref"
    if any(hint in lower for hint in LIFE_EVENT_HINTS) and _has_time_reference(lower):
        return "general"
    if any(hint in lower for hint in PROJECT_HINTS):
        return "project"
    if any(hint in lower for hint in ENVIRONMENT_HINTS):
        return "environment"
    if any(hint in lower for hint in WORKFLOW_HINTS):
        return "workflow"
    return "general"


def infer_topic(sentence: str, category: str, subject_key: str = "") -> str:
    if subject_key in {"user:response_style", "user:answer_format"}:
        return "user-preferences"
    if subject_key.startswith("user:preference:") or subject_key.startswith("user:favorite:"):
        return "user-preferences"
    if subject_key.startswith("user:event:"):
        return "life-events"
    if subject_key in {
        "user:role",
        "user:timezone",
        "user:diet",
        "user:origin",
        "user:hometown",
        "user:location:current",
        "user:pronouns",
        "user:relationship_status",
    } or subject_key.startswith("user:allergy:"):
        return "personal-profile"
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
        for hint in (PREFERENCE_HINTS + PROJECT_HINTS + ENVIRONMENT_HINTS + WORKFLOW_HINTS + PERSONAL_DETAIL_HINTS)
    ) or (any(hint in lower for hint in LIFE_EVENT_HINTS) and _has_time_reference(lower))


def _has_time_reference(text: str) -> bool:
    lower = text.lower()
    if any(token in lower for token in TIME_REFERENCE_HINTS):
        return True
    return bool(
        re.search(
            r"\b\d{4}-\d{2}-\d{2}\b|\b(?:in|on)\s+\d{4}\b|\b(?:this|next|last)\s+(?:summer|winter|spring|fall|autumn|week|month|year|weekend)\b",
            lower,
            flags=re.IGNORECASE,
        )
    )


def _trim_phrase(text: str, *, split_lists: bool = False) -> str:
    clean = normalize_whitespace(text.strip(" .,!?:;\"'`"))
    clean = re.sub(r"\b(?:right now|currently|for now|these days|now)$", "", clean, flags=re.IGNORECASE)
    clean = normalize_whitespace(clean)
    if split_lists:
        return clean
    parts = re.split(r"\b(?:because|since|but|although|though|while|currently|with)\b", clean, maxsplit=1, flags=re.IGNORECASE)
    return normalize_whitespace(parts[0] if parts else clean)


def _normalize_preference_items(raw: str) -> List[str]:
    clean = _trim_phrase(raw, split_lists=True)
    if not clean:
        return []
    parts = re.split(r"\s*,\s*|\s+\band\b\s+|\s+\bor\b\s+", clean, flags=re.IGNORECASE)
    if len(parts) <= 1:
        parts = [clean]
    items: List[str] = []
    seen = set()
    for part in parts[:4]:
        item = normalize_whitespace(part.strip(" .,!?:;\"'`"))
        item = re.sub(r"^(?:a|an|the|my)\s+", "", item, flags=re.IGNORECASE)
        item = re.sub(r"\b(?:a lot|very much|too much)$", "", item, flags=re.IGNORECASE)
        item = normalize_whitespace(item)
        if not item or len(item) > 48:
            continue
        if any(
            token in item.lower()
            for token in (
                "response",
                "responses",
                "answer",
                "answers",
                "explanation",
                "explanations",
                "bullet",
                "bullets",
                "paragraph",
                "paragraphs",
                "format",
                "tone",
                "style",
                "markdown",
                "code block",
                "code blocks",
            )
        ):
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(item)
    return items


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


def _extract_favorite_things(segment: str, source_role: str) -> List[Dict[str, Any]]:
    match = re.search(
        r"\bmy\s+favorite\s+([a-z][a-z0-9 _/-]{1,24})\s+is\s+(.+?)(?:[.;,]|$)",
        segment,
        flags=re.IGNORECASE,
    )
    if not match:
        return []
    trait = normalize_whitespace(match.group(1))
    value = _trim_phrase(match.group(2))
    if not trait or not value:
        return []
    return [
        build_candidate(
            content=f"User's favorite {trait} is {value}",
            category="user_pref",
            topic="user-preferences",
            importance=7,
            confidence=0.85 if source_role == "user" else 0.6,
            source_role=source_role,
            subject_key=f"user:favorite:{slugify(trait)}",
            value_key=slugify(value),
            exclusive=True,
            metadata={"trait_label": trait, "value_label": value},
        )
    ]


def _extract_user_preferences(segment: str, source_role: str) -> List[Dict[str, Any]]:
    patterns = (
        (r"\bi\s+(?:really\s+|absolutely\s+|honestly\s+|pretty\s+much\s+|kind of\s+|kinda\s+)*(like|love|enjoy|prefer)\s+(.+?)(?:[.;,]|$)", 1, "likes"),
        (r"\bi\s+(?:really\s+|absolutely\s+|honestly\s+)*(dislike|hate|avoid|can't stand|cannot stand|do not like|don't like)\s+(.+?)(?:[.;,]|$)", -1, "dislikes"),
    )
    candidates: List[Dict[str, Any]] = []
    for pattern, polarity, verb in patterns:
        match = re.search(pattern, segment, flags=re.IGNORECASE)
        if not match:
            continue
        for item in _normalize_preference_items(match.group(2)):
            candidates.append(
                build_candidate(
                    content=f"User {verb} {item}",
                    category="user_pref",
                    topic="user-preferences",
                    importance=6 if polarity > 0 else 7,
                    confidence=0.85 if source_role == "user" else 0.6,
                    source_role=source_role,
                    subject_key=f"user:preference:{slugify(item)}",
                    value_key="like" if polarity > 0 else "dislike",
                    exclusive=True,
                    polarity=polarity,
                    metadata={"item_label": item},
                )
            )
        break
    return candidates


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


def _extract_personal_details(segment: str, source_role: str) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    confidence = 0.85 if source_role == "user" else 0.6

    match = re.search(r"\bmy\s+pronouns\s+are\s+([A-Za-z/ -]{2,24})\b", segment, flags=re.IGNORECASE)
    if match:
        pronouns = normalize_whitespace(match.group(1))
        candidates.append(
            build_candidate(
                content=f"User pronouns are {pronouns}",
                category="user_pref",
                topic="personal-profile",
                importance=8,
                confidence=confidence,
                source_role=source_role,
                subject_key="user:pronouns",
                value_key=slugify(pronouns),
                exclusive=True,
                metadata={"pronouns_label": pronouns},
            )
        )

    match = re.search(
        r"\b(?:i(?:'m| am)\s+based in|i live in)\s+(.+?)(?:[.;,]|$)",
        segment,
        flags=re.IGNORECASE,
    )
    if match:
        location = _trim_phrase(match.group(1))
        if location:
            candidates.append(
                build_candidate(
                    content=f"User lives in {location}",
                    category="user_pref",
                    topic="personal-profile",
                    importance=7,
                    confidence=confidence,
                    source_role=source_role,
                    subject_key="user:location:current",
                    value_key=slugify(location),
                    exclusive=True,
                    metadata={"location_label": location},
                )
            )

    match = re.search(r"\b(?:i(?:'m| am)\s+from)\s+(.+?)(?:[.;,]|$)", segment, flags=re.IGNORECASE)
    if match:
        origin = _trim_phrase(match.group(1))
        if origin:
            candidates.append(
                build_candidate(
                    content=f"User is from {origin}",
                    category="user_pref",
                    topic="personal-profile",
                    importance=7,
                    confidence=confidence,
                    source_role=source_role,
                    subject_key="user:origin",
                    value_key=slugify(origin),
                    exclusive=True,
                    metadata={"origin_label": origin},
                )
            )

    match = re.search(r"\bi grew up in\s+(.+?)(?:[.;,]|$)", segment, flags=re.IGNORECASE)
    if match:
        hometown = _trim_phrase(match.group(1))
        if hometown:
            candidates.append(
                build_candidate(
                    content=f"User grew up in {hometown}",
                    category="user_pref",
                    topic="personal-profile",
                    importance=7,
                    confidence=confidence,
                    source_role=source_role,
                    subject_key="user:hometown",
                    value_key=slugify(hometown),
                    exclusive=True,
                    metadata={"hometown_label": hometown},
                )
            )

    match = re.search(
        r"\bi(?:'m| am)\s+(vegetarian|vegan|pescatarian|gluten[- ]free|lactose intolerant)\b",
        segment,
        flags=re.IGNORECASE,
    )
    if match:
        diet = normalize_whitespace(match.group(1)).replace("  ", " ")
        candidates.append(
            build_candidate(
                content=f"User is {diet}",
                category="user_pref",
                topic="personal-profile",
                importance=8,
                confidence=confidence,
                source_role=source_role,
                subject_key="user:diet",
                value_key=slugify(diet),
                exclusive=True,
                metadata={"diet_label": diet},
            )
        )

    match = re.search(r"\bi(?:'m| am)\s+(single|married|engaged|divorced|widowed)\b", segment, flags=re.IGNORECASE)
    if match:
        status = normalize_whitespace(match.group(1))
        candidates.append(
            build_candidate(
                content=f"User is {status}",
                category="user_pref",
                topic="personal-profile",
                importance=7,
                confidence=confidence,
                source_role=source_role,
                subject_key="user:relationship_status",
                value_key=slugify(status),
                exclusive=True,
                metadata={"relationship_label": status},
            )
        )

    for pattern, polarity, label in (
        (r"\bi(?:'m| am)\s+allergic to\s+(.+?)(?:[.;,]|$)", 1, "allergic"),
        (r"\bi(?:'m| am)\s+not allergic to\s+(.+?)(?:[.;,]|$)", -1, "not allergic"),
    ):
        match = re.search(pattern, segment, flags=re.IGNORECASE)
        if not match:
            continue
        for item in _normalize_preference_items(match.group(1)):
            candidates.append(
                build_candidate(
                    content=f"User is {label} to {item}",
                    category="user_pref",
                    topic="personal-profile",
                    importance=8,
                    confidence=confidence,
                    source_role=source_role,
                    subject_key=f"user:allergy:{slugify(item)}",
                    value_key="allergic" if polarity > 0 else "not-allergic",
                    exclusive=True,
                    polarity=polarity,
                    metadata={"item_label": item},
                )
            )
        break

    return candidates


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


def _extract_life_event(segment: str, source_role: str) -> List[Dict[str, Any]]:
    lower = segment.lower()
    if not re.search(r"\b(i|my|we|our)\b", lower):
        return []
    if not any(hint in lower for hint in LIFE_EVENT_HINTS):
        return []
    if not (_has_time_reference(lower) or any(hint in lower for hint in RELATION_HINTS)):
        return []
    event_type = next((hint for hint in LIFE_EVENT_HINTS if hint in lower), "life-event")
    metadata: Dict[str, Any] = {"kind": "life_event", "event_type": slugify(event_type)}
    date_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", segment)
    if date_match:
        metadata["event_date"] = date_match.group(0)
    relation = next((hint for hint in RELATION_HINTS if hint in lower), "")
    if relation:
        metadata["relation_label"] = relation.replace("my ", "").strip()
    return [
        build_candidate(
            content=segment,
            category="general",
            topic="life-events",
            importance=8 if any(token in lower for token in ("birthday", "wedding", "married", "graduat", "surgery", "baby")) else 7,
            confidence=0.85 if source_role == "user" else 0.6,
            source_role=source_role,
            subject_key=f"user:event:{slugify(event_type)}",
            value_key=slugify(segment[:64]),
            metadata=metadata,
        )
    ]


def _structured_candidates(segment: str, source_role: str) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    extractors = (
        _extract_response_style,
        _extract_answer_format,
        _extract_favorite_things,
        _extract_user_preferences,
        _extract_personal_details,
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
        _extract_life_event,
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
    has_pending = pending_episodes > 0
    return {
        "last_consolidated_at": last_at,
        "last_consolidated_episode_id": last_episode_id,
        "hours_since_last": None if hours_since == float("inf") else round(hours_since, 2),
        "pending_sessions": int(max(pending_sessions, 0)),
        "pending_episodes": int(max(pending_episodes, 0)),
        "min_hours": int(min_hours),
        "min_sessions": int(min_sessions),
        "should_run": has_pending and (hours_since == float("inf") or hours_since >= min_hours)
        and pending_sessions >= min_sessions,
    }


def _build_session_summary_text(artifacts: Dict[str, Any], *, max_chars: int) -> str:
    parts: List[str] = []
    facts = [str(item.get("content") or "") for item in artifacts.get("facts", [])[:4] if item.get("content")]
    if facts:
        parts.append("Facts: " + "; ".join(facts))
    journals = [str(item.get("content") or "") for item in artifacts.get("journals", [])[:2] if item.get("content")]
    if journals:
        parts.append("Notes: " + " | ".join(journals))
    traces = [str(item.get("content") or "") for item in artifacts.get("traces", [])[:3] if item.get("content")]
    if traces:
        parts.append("Recent flow: " + " | ".join(traces))
    preferences = [str(item.get("content") or item.get("label") or "") for item in artifacts.get("preferences", [])[:2] if item.get("content") or item.get("label")]
    if preferences:
        parts.append("Preferences: " + " | ".join(preferences))
    policies = [str(item.get("content") or item.get("label") or "") for item in artifacts.get("policies", [])[:2] if item.get("content") or item.get("label")]
    if policies:
        parts.append("Policies: " + " | ".join(policies))
    summary = " ".join(part for part in parts if part).strip()
    return summary[:max_chars] if summary else ""


def run_consolidation(
    store: MemoryStore,
    *,
    min_hours: int,
    min_sessions: int,
    max_topic_facts: int,
    topic_summary_chars: int,
    prune_after_days: int,
    session_summary_chars: int = 900,
    episode_retention_hours: float = 24.0,
    decay_half_life_days: float = 90.0,
    decay_min_salience: float = 0.15,
    extractor=None,
    fact_writer=None,
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
    touched_sessions = set()

    for episode in episodes:
        extract = extractor or extract_candidate_facts_from_turn
        session_id = normalize_whitespace(str(episode.get("session_id") or ""))
        if session_id:
            touched_sessions.add(session_id)
        for candidate in extract(
            user_content=str(episode.get("user_content", "")),
            assistant_content=str(episode.get("assistant_content", "")),
            created_at=float(episode.get("created_at") or started_at),
        ):
            if fact_writer:
                result = fact_writer(candidate, episode)
            else:
                result = store.upsert_fact(
                    content=str(candidate["content"]),
                    category=str(candidate["category"]),
                    topic=str(candidate["topic"]),
                    source="episode_extract",
                    importance=int(candidate["importance"]),
                    confidence=float(candidate["confidence"]),
                    metadata=dict(candidate.get("metadata") or {}),
                    observed_at=float(episode.get("created_at") or started_at),
                    source_session_id=session_id,
                    history_reason="episode_extract",
                )
                fact_id = dict(result.get("fact") or {}).get("id")
                if fact_id is not None and episode.get("id") is not None:
                    store.add_link("fact", fact_id, "episode", int(episode["id"]), "derived_from_episode")
            if result["action"] == "inserted":
                facts_added += 1
            else:
                facts_updated += 1
            facts_superseded += len(result.get("superseded", []))
            contradictions_resolved += len(result.get("contradictions", []))

    pruned = store.prune_stale_facts(max_age_days=prune_after_days)
    decay_stats = store.apply_decay(
        half_life_days=decay_half_life_days,
        min_salience=decay_min_salience,
    )
    topics_rebuilt = store.rebuild_topics(
        max_facts=max_topic_facts,
        max_chars=topic_summary_chars,
    )
    session_summaries = 0
    for session_id in sorted(touched_sessions):
        artifacts = store.get_session_artifacts(session_id, limit=max(8, max_topic_facts * 2))
        summary = _build_session_summary_text(artifacts, max_chars=int(session_summary_chars))
        if not summary:
            continue
        refs: List[Dict[str, Any]] = []
        kind_map = {
            "facts": "fact",
            "journals": "journal",
            "traces": "trace",
            "episodes": "episode",
            "preferences": "preference",
            "policies": "policy",
        }
        for section in ("facts", "journals", "traces", "episodes", "preferences", "policies"):
            for item in artifacts.get(section, [])[:4]:
                if item.get("id") is None:
                    continue
                refs.append({"kind": kind_map[section], "id": item["id"]})
        store.upsert_summary(
            label="Session Summary",
            summary=summary,
            session_id=session_id,
            content=summary,
            summary_type="session",
            metadata={"source": "consolidation"},
            importance=8,
            salience=0.72,
            source_refs=refs,
            reason="consolidation_distill",
        )
        store.ensure_memory_session(session_id, summary=summary)
        session_summaries += 1
    latest_episode_id = store.latest_episode_id()
    episodes_pruned = store.purge_episode_buffers(
        retention_hours=episode_retention_hours,
        max_episode_id=latest_episode_id,
    )
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
        "session_summaries": session_summaries,
        "episodes_pruned": episodes_pruned,
        "decay": decay_stats,
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
