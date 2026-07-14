"""Guided, review-first bootstrap of a user's durable memory profile."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

from .store import MemoryStore, _looks_like_credential, fingerprint_text, normalize_text, normalize_whitespace, slugify


ONBOARDING_VERSION = 1
MAX_ANSWER_CHARS = 4000
MAX_LIST_ITEMS = 20
MAX_ITEM_CHARS = 500


@dataclass(frozen=True)
class OnboardingQuestion:
    key: str
    prompt: str
    hint: str


QUESTIONS = (
    OnboardingQuestion("preferred_name", "What should Hermes call you?", "Your preferred first name or handle."),
    OnboardingQuestion("pronouns", "What pronouns should Hermes use?", "Optional; for example: they/them."),
    OnboardingQuestion("timezone", "What is your timezone?", "For example: Europe/Paris."),
    OnboardingQuestion(
        "languages", "Which languages should Hermes use with you?", "Comma-separated, in preferred order."
    ),
    OnboardingQuestion("occupation", "What is your occupation or main role?", "Keep this broad if you prefer."),
    OnboardingQuestion(
        "broad_location",
        "What broad location is useful to remember?",
        "Country or region only; do not enter an address.",
    ),
    OnboardingQuestion(
        "response_style", "How should Hermes structure its answers?", "For example: concise, then technical details."
    ),
    OnboardingQuestion("response_tone", "What response tone do you prefer?", "For example: direct and collaborative."),
    OnboardingQuestion(
        "preferred_tools", "Which tools do you prefer?", "Comma-separated editors, shells, languages, etc."
    ),
    OnboardingQuestion(
        "technical_interests", "Which technical areas matter most to you?", "Comma-separated topics or specialties."
    ),
    OnboardingQuestion("active_projects", "Which projects are currently active?", "Comma-separated project names."),
    OnboardingQuestion("current_goals", "What are your current goals?", "Separate multiple goals with semicolons."),
    OnboardingQuestion(
        "approval_rules", "Which actions must Hermes ask before taking?", "Separate multiple rules with semicolons."
    ),
    OnboardingQuestion(
        "avoid_in_responses",
        "What should Hermes avoid in its responses?",
        "For example: excessive headings or repetition.",
    ),
    OnboardingQuestion(
        "never_remember",
        "What categories of information must never be remembered?",
        "Rules only; never enter a secret.",
    ),
    OnboardingQuestion(
        "recurring_workflow",
        "Describe one recurring workflow, if useful.",
        "Format: Workflow name | first step; second step; final step.",
    ),
    OnboardingQuestion(
        "additional_context", "Any other stable context Hermes should know?", "Avoid temporary details and secrets."
    ),
)
QUESTION_KEYS = frozenset(question.key for question in QUESTIONS)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        raise ValueError("Onboarding answers must be strings; use comma or semicolon separators for lists")
    clean = normalize_whitespace(str(value).lstrip("\ufeff"))
    if len(clean) > MAX_ANSWER_CHARS:
        raise ValueError(f"Onboarding answers cannot exceed {MAX_ANSWER_CHARS} characters")
    return clean


def _items(value: Any, separator: str) -> List[str]:
    if isinstance(value, (list, tuple)):
        raw_items = [_text(item) for item in value]
    else:
        raw_items = [_text(item) for item in _text(value).split(separator)]
    items: List[str] = []
    for item in raw_items:
        if not item:
            continue
        if len(item) > MAX_ITEM_CHARS:
            raise ValueError(f"Onboarding list items cannot exceed {MAX_ITEM_CHARS} characters")
        if item not in items:
            items.append(item)
    if len(items) > MAX_LIST_ITEMS:
        raise ValueError(f"Onboarding lists cannot exceed {MAX_LIST_ITEMS} items")
    return items


def classify_sensitivity(content: str, metadata: Dict[str, Any] | None = None) -> tuple[str, str]:
    """Mirror provider admission labels without calling a model or endpoint."""
    meta = dict(metadata or {})
    subject = normalize_text(str(meta.get("subject_key") or ""))
    text = normalize_text(content)
    combined = f"{subject} {text}"
    if _looks_like_credential(content) or re.search(
        r"\b(password|passphrase|api[_ -]?key|access[_ -]?token|private[_ -]?key|secret)\b", combined
    ):
        return "credential", "credential or secret material"
    if any(token in combined for token in ("health", "medical", "diagnosis", "medication", "surgery", "allerg")):
        return "health", "health information"
    if any(token in combined for token in ("financial", "bank", "iban", "credit card", "salary", "income", "debt")):
        return "financial", "financial information"
    if any(token in combined for token in ("date of birth", "dob", "passport", "social security", "national id")):
        return "identity", "identity information"
    if any(token in subject for token in ("address", "exact_location", "home_location")):
        return "location", "precise location"
    return "normal", ""


def load_answers(path: str | Path) -> Dict[str, str]:
    source = Path(path).expanduser().resolve()
    with open(source, encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Onboarding answer files must contain a JSON object")
    return validate_answers(payload)


def validate_answers(answers: Dict[str, Any]) -> Dict[str, str]:
    unknown = sorted(set(answers) - QUESTION_KEYS)
    if unknown:
        raise ValueError(f"Unknown onboarding answer keys: {', '.join(unknown)}")
    return {key: _text(value) for key, value in answers.items() if _text(value)}


def collect_answers(
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> Dict[str, str]:
    output_fn("Hermes memory onboarding")
    output_fn("Press Enter to skip any question. Do not enter passwords, API keys, tokens, or exact addresses.")
    answers: Dict[str, str] = {}
    for index, question in enumerate(QUESTIONS, start=1):
        output_fn(f"\n[{index}/{len(QUESTIONS)}] {question.prompt}")
        output_fn(f"  {question.hint}")
        try:
            value = _text(input_fn("> "))
        except EOFError:
            output_fn("\nOnboarding cancelled; no answers were stored.")
            return {}
        if value:
            answers[question.key] = value
    return answers


def _base_item(memory_type: str, key: str, content: str, *, label: str = "") -> Dict[str, Any]:
    return {
        "memory_type": memory_type,
        "key": key,
        "label": label or key.replace("_", " ").title(),
        "content": normalize_whitespace(content),
        "sensitivity": "normal",
        "metadata": {
            "onboarding_version": ONBOARDING_VERSION,
            "source_role": "user",
            "local_only": True,
        },
    }


def _fact(
    key: str,
    label: str,
    content: str,
    *,
    value: str,
    category: str = "profile",
    topic: str = "user-profile",
    importance: int = 8,
    exclusive: bool = True,
) -> Dict[str, Any]:
    item = _base_item("fact", key, content, label=label)
    item.update(
        {
            "category": category,
            "topic": topic,
            "importance": importance,
            "confidence": 1.0,
            "pinned": True,
        }
    )
    item["metadata"].update(
        {
            "subject_key": key,
            "value_key": normalize_text(value),
            "value_label": value,
            "exclusive": exclusive,
        }
    )
    return item


def _preference(key: str, label: str, value: str, content: str, *, importance: int = 8) -> Dict[str, Any]:
    item = _base_item("preference", key, content, label=label)
    item.update({"value": value, "importance": importance})
    item["metadata"].update({"subject_key": key})
    return item


def _policy(key: str, label: str, content: str) -> Dict[str, Any]:
    item = _base_item("policy", key, content, label=label)
    item.update({"importance": 10})
    item["metadata"].update({"subject_key": f"policy:{key}"})
    return item


def _append_classified(items: List[Dict[str, Any]], skipped: List[Dict[str, str]], item: Dict[str, Any]) -> None:
    sensitivity, reason = classify_sensitivity(str(item["content"]), dict(item.get("metadata") or {}))
    # A category-exclusion policy names sensitive classes but does not contain
    # the excluded data. Keep the rule visible unless it contains a real token.
    if (
        item["memory_type"] == "policy"
        and item["key"] == "onboarding-never-remember"
        and not _looks_like_credential(item["content"])
    ):
        sensitivity, reason = "normal", ""
    if sensitivity == "credential":
        skipped.append(
            {
                "key": str(item["key"]),
                "memory_type": str(item["memory_type"]),
                "reason": "Credential-like content is never accepted by onboarding",
            }
        )
        return
    item["sensitivity"] = sensitivity
    if reason:
        item["sensitivity_reason"] = reason
    items.append(item)


def build_onboarding_plan(answers: Dict[str, Any], *, skip_sensitive: bool = False) -> Dict[str, Any]:
    clean = validate_answers(answers)
    items: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []

    single_facts = (
        ("preferred_name", "user:name", "Preferred name", "The user's preferred name is {value}.", "profile", 10),
        ("pronouns", "user:pronouns", "Pronouns", "The user's pronouns are {value}.", "profile", 8),
        ("timezone", "environment:timezone", "Timezone", "The user's timezone is {value}.", "environment", 9),
        ("occupation", "user:occupation", "Occupation", "The user's occupation or main role is {value}.", "profile", 8),
        (
            "broad_location",
            "user:broad_location",
            "Broad location",
            "The user's broad location is {value}.",
            "profile",
            7,
        ),
    )
    for answer_key, subject_key, label, template, category, importance in single_facts:
        value = clean.get(answer_key, "")
        if value:
            _append_classified(
                items,
                skipped,
                _fact(
                    subject_key,
                    label,
                    template.format(value=value),
                    value=value,
                    category=category,
                    importance=importance,
                ),
            )

    languages = _items(clean.get("languages", ""), ",")
    if languages:
        value = ", ".join(languages)
        _append_classified(
            items,
            skipped,
            _preference(
                "user:preference:language",
                "Communication languages",
                value,
                f"Use these languages with the user, in preference order: {value}.",
                importance=9,
            ),
        )

    preference_fields = (
        (
            "response_style",
            "user:preference:response_style",
            "Response structure",
            "Structure responses this way: {value}",
            9,
        ),
        ("response_tone", "user:preference:response_tone", "Response tone", "Use this response tone: {value}", 8),
        ("preferred_tools", "user:preference:tools", "Preferred tools", "Prefer these tools: {value}", 8),
        (
            "avoid_in_responses",
            "user:preference:response_avoid",
            "Response elements to avoid",
            "Avoid these response patterns: {value}",
            9,
        ),
    )
    for answer_key, key, label, template, importance in preference_fields:
        value = clean.get(answer_key, "")
        if value:
            _append_classified(
                items,
                skipped,
                _preference(key, label, value, template.format(value=value), importance=importance),
            )

    for interest in _items(clean.get("technical_interests", ""), ","):
        key = f"user:technical_interest:{slugify(interest)}"
        _append_classified(
            items,
            skipped,
            _fact(
                key,
                "Technical interest",
                f"The user is interested in {interest}.",
                value=interest,
                category="interest",
                topic="technical-interests",
                importance=7,
                exclusive=False,
            ),
        )

    for project in _items(clean.get("active_projects", ""), ","):
        key = f"project:{slugify(project)}:active"
        _append_classified(
            items,
            skipped,
            _fact(
                key,
                "Active project",
                f"The user is actively working on {project}.",
                value=project,
                category="project",
                topic=f"project-{slugify(project)}",
                importance=8,
                exclusive=False,
            ),
        )

    for goal in _items(clean.get("current_goals", ""), ";"):
        item = _base_item("intention", f"goal-{fingerprint_text(goal)[:12]}", goal, label="Current goal")
        item.update({"importance": 8, "condition_text": ""})
        _append_classified(items, skipped, item)

    for index, rule in enumerate(_items(clean.get("approval_rules", ""), ";"), start=1):
        key = f"onboarding-approval-{fingerprint_text(rule)[:12]}"
        _append_classified(items, skipped, _policy(key, f"Approval rule {index}", rule))

    never_remember = clean.get("never_remember", "")
    if never_remember:
        _append_classified(
            items,
            skipped,
            _policy(
                "onboarding-never-remember",
                "Never remember",
                f"Never retain these categories of information: {never_remember}",
            ),
        )

    workflow = clean.get("recurring_workflow", "")
    if workflow:
        if "|" in workflow:
            label, raw_steps = (part.strip() for part in workflow.split("|", 1))
        else:
            label, raw_steps = "Recurring workflow", workflow
        steps = _items(raw_steps, ";")
        if not steps:
            skipped.append({"key": "recurring_workflow", "memory_type": "procedure", "reason": "No steps supplied"})
        else:
            item = _base_item("procedure", f"onboarding-{slugify(label)}", " ".join([label, *steps]), label=label)
            item.update({"steps": steps, "confidence": 1.0})
            _append_classified(items, skipped, item)

    additional = clean.get("additional_context", "")
    if additional:
        _append_classified(
            items,
            skipped,
            _fact(
                f"user:context:{fingerprint_text(additional)[:12]}",
                "Additional stable context",
                additional,
                value=additional,
                category="profile",
                topic="user-profile",
                importance=7,
                exclusive=False,
            ),
        )

    if skip_sensitive:
        safe_items: List[Dict[str, Any]] = []
        for item in items:
            if str(item.get("sensitivity") or "normal") == "normal":
                safe_items.append(item)
            else:
                skipped.append(
                    {
                        "key": str(item["key"]),
                        "memory_type": str(item["memory_type"]),
                        "reason": f"Skipped by --skip-sensitive ({item['sensitivity']})",
                    }
                )
        items = safe_items

    return {"version": ONBOARDING_VERSION, "items": items, "skipped": skipped}


def render_plan(plan: Dict[str, Any]) -> str:
    lines = ["\nProposed memory profile", "=" * 72]
    items = list(plan.get("items") or [])
    if not items:
        lines.append("No eligible memories were provided.")
    for index, item in enumerate(items, start=1):
        sensitivity = str(item.get("sensitivity") or "normal")
        privacy = "local-only" if bool(dict(item.get("metadata") or {}).get("local_only")) else "remote-eligible"
        lines.append(
            f"{index:>2}. [{item['memory_type']}] {item['label']} ({sensitivity}, {privacy})\n    {item['content']}"
        )
    skipped = list(plan.get("skipped") or [])
    if skipped:
        lines.append("\nSkipped entries")
        for entry in skipped:
            lines.append(f" - [{entry['memory_type']}] {entry['key']}: {entry['reason']}")
    lines.append("=" * 72)
    return "\n".join(lines)


def _existing_intention(store: MemoryStore, content: str) -> Dict[str, Any] | None:
    target = normalize_text(content)
    for row in store.list_intentions(limit=1000):
        if normalize_text(str(row.get("intention") or "")) == target:
            return row
    return None


def _is_same_onboarding_row(row: Dict[str, Any] | None) -> bool:
    metadata = dict((row or {}).get("metadata") or {})
    return bool(row) and int(metadata.get("onboarding_version") or 0) == ONBOARDING_VERSION


def apply_onboarding(store: MemoryStore, plan: Dict[str, Any]) -> Dict[str, Any]:
    items = list(plan.get("items") or [])
    counts: Dict[str, int] = {"facts": 0, "preferences": 0, "policies": 0, "procedures": 0, "intentions": 0}
    unchanged_counts: Dict[str, int] = {
        "facts": 0,
        "preferences": 0,
        "policies": 0,
        "procedures": 0,
        "intentions": 0,
    }
    results: List[Dict[str, Any]] = []
    with store.transaction():
        for item in items:
            memory_type = str(item["memory_type"])
            metadata = dict(item.get("metadata") or {})
            sensitivity = str(item.get("sensitivity") or "normal")
            status = "stored"
            if memory_type == "fact":
                row = store._fetchone(
                    "SELECT * FROM facts WHERE fingerprint = ?",
                    (fingerprint_text(str(item["content"])),),
                )
                if _is_same_onboarding_row(row):
                    status = "unchanged"
                    unchanged_counts["facts"] += 1
                else:
                    result = store.upsert_fact(
                        content=str(item["content"]),
                        category=str(item.get("category") or "profile"),
                        topic=str(item.get("topic") or "user-profile"),
                        source="onboarding",
                        importance=int(item.get("importance") or 8),
                        confidence=float(item.get("confidence") or 1.0),
                        metadata=metadata,
                        source_role="user",
                        reliability=1.0,
                        history_reason=f"onboarding-v{ONBOARDING_VERSION}",
                        sensitivity=sensitivity,
                        pinned=bool(item.get("pinned", True)),
                    )
                    row = dict(result.get("fact") or {})
                    counts["facts"] += 1
            elif memory_type == "preference":
                row = store._fetchone(
                    "SELECT * FROM memory_preferences WHERE preference_key = ?",
                    (str(item["key"]),),
                )
                same_value = normalize_text(str((row or {}).get("value") or "")) == normalize_text(
                    str(item.get("value") or item["content"])
                )
                same_content = normalize_text(str((row or {}).get("content") or "")) == normalize_text(
                    str(item["content"])
                )
                if _is_same_onboarding_row(row) and same_value and same_content:
                    status = "unchanged"
                    unchanged_counts["preferences"] += 1
                else:
                    row = store.upsert_preference(
                        key=str(item["key"]),
                        label=str(item["label"]),
                        value=str(item.get("value") or item["content"]),
                        content=str(item["content"]),
                        metadata=metadata,
                        importance=int(item.get("importance") or 8),
                        salience=0.95,
                        reason=f"onboarding-v{ONBOARDING_VERSION}",
                        sensitivity=sensitivity,
                    )
                    counts["preferences"] += 1
            elif memory_type == "policy":
                row = store._fetchone(
                    "SELECT * FROM memory_policies WHERE policy_key = ?",
                    (str(item["key"]),),
                )
                same_content = normalize_text(str((row or {}).get("content") or "")) == normalize_text(
                    str(item["content"])
                )
                if _is_same_onboarding_row(row) and same_content:
                    status = "unchanged"
                    unchanged_counts["policies"] += 1
                else:
                    row = store.upsert_policy(
                        key=str(item["key"]),
                        label=str(item["label"]),
                        content=str(item["content"]),
                        metadata=metadata,
                        importance=int(item.get("importance") or 10),
                        salience=1.0,
                        reason=f"onboarding-v{ONBOARDING_VERSION}",
                        sensitivity=sensitivity,
                    )
                    counts["policies"] += 1
            elif memory_type == "procedure":
                row = store._fetchone(
                    "SELECT * FROM memory_procedures WHERE procedure_key = ?",
                    (slugify(str(item["key"])),),
                )
                existing_steps = [normalize_text(str(step)) for step in list((row or {}).get("steps") or [])]
                proposed_steps = [normalize_text(str(step)) for step in list(item.get("steps") or [])]
                if _is_same_onboarding_row(row) and existing_steps == proposed_steps:
                    status = "unchanged"
                    unchanged_counts["procedures"] += 1
                else:
                    row = store.upsert_procedure(
                        procedure_key=str(item["key"]),
                        label=str(item["label"]),
                        steps=list(item.get("steps") or []),
                        confidence=float(item.get("confidence") or 1.0),
                        metadata=metadata,
                        sensitivity=sensitivity,
                    )
                    counts["procedures"] += 1
            elif memory_type == "intention":
                existing = _existing_intention(store, str(item["content"]))
                if existing:
                    row = existing
                    status = "unchanged"
                    unchanged_counts["intentions"] += 1
                else:
                    row = store.add_intention(
                        intention=str(item["content"]),
                        condition_text=str(item.get("condition_text") or ""),
                        importance=int(item.get("importance") or 8),
                        metadata=metadata,
                        sensitivity=sensitivity,
                    )
                    counts["intentions"] += 1
            else:
                raise ValueError(f"Unsupported onboarding memory type: {memory_type}")
            results.append(
                {"memory_type": memory_type, "id": (row or {}).get("id"), "key": item["key"], "status": status}
            )
    return {
        "processed": len(results),
        "stored": sum(counts.values()),
        "unchanged": sum(unchanged_counts.values()),
        "counts": counts,
        "unchanged_counts": unchanged_counts,
        "results": results,
    }


def run_onboarding(
    store: MemoryStore,
    answers: Dict[str, Any],
    *,
    preview_only: bool = False,
    assume_yes: bool = False,
    skip_sensitive: bool = False,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> Dict[str, Any]:
    plan = build_onboarding_plan(answers, skip_sensitive=skip_sensitive)
    output_fn(render_plan(plan))
    if preview_only:
        return {"status": "preview", "proposed": len(plan["items"]), "skipped": len(plan["skipped"])}
    if not plan["items"]:
        return {"status": "empty", "stored": 0, "skipped": len(plan["skipped"])}
    if not assume_yes:
        try:
            confirmed = input_fn("Store this profile in Hermes memory? [y/N] ")
        except EOFError:
            confirmed = ""
        if normalize_text(confirmed) not in {"y", "yes"}:
            return {"status": "cancelled", "stored": 0, "skipped": len(plan["skipped"])}
    result = apply_onboarding(store, plan)
    return {"status": "stored", "skipped": len(plan["skipped"]), **result}


def answer_template() -> Dict[str, str]:
    return {question.key: "" for question in QUESTIONS}


def write_answer_template(path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "x", encoding="utf-8") as handle:
        json.dump(answer_template(), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    try:
        destination.chmod(0o600)
    except OSError:
        pass
    return destination
