from __future__ import annotations

import random
from dataclasses import dataclass

DEFAULT_SEED = 42


@dataclass(frozen=True)
class FactSeed:
    text: str
    target: str
    category: str
    topic: str
    importance: int = 6
    memory_type: str = "fact"
    subject_key: str = ""
    value: str = ""
    useful: bool = True
    tier: str = ""
    observed_days_ago: int = 0

    def addon_args(self) -> dict[str, object]:
        args: dict[str, object] = {
            "action": "remember",
            "memory_type": self.memory_type,
            "content": self.text,
            "category": self.category,
            "topic": self.topic,
            "importance": self.importance,
        }
        if self.subject_key:
            args["subject_key"] = self.subject_key
        if self.value:
            args["value"] = self.value
        return args


@dataclass(frozen=True)
class SessionSeed:
    session_id: str
    facts: tuple[FactSeed, ...]


@dataclass(frozen=True)
class ContradictionPair:
    earlier: FactSeed
    later: FactSeed


def _fact(
    text: str,
    *,
    target: str,
    category: str,
    topic: str,
    importance: int = 6,
    memory_type: str = "fact",
    subject_key: str = "",
    value: str = "",
    useful: bool = True,
    tier: str = "",
    observed_days_ago: int = 0,
) -> FactSeed:
    return FactSeed(
        text=text,
        target=target,
        category=category,
        topic=topic,
        importance=importance,
        memory_type=memory_type,
        subject_key=subject_key,
        value=value,
        useful=useful,
        tier=tier,
        observed_days_ago=observed_days_ago,
    )


BASE_SCALE_FACTS: tuple[FactSeed, ...] = (
    _fact(
        "User role: MIG/MAG welder who automates shop paperwork with small Python scripts.",
        target="user",
        category="user_pref",
        topic="personal-profile",
        importance=8,
        subject_key="user:role",
        value="mig-mag-welder",
    ),
    _fact(
        "User lives in Cambrai and mostly works from a quiet workshop office near the floor.",
        target="user",
        category="user_pref",
        topic="personal-profile",
        importance=8,
        subject_key="user:location:current",
        value="cambrai",
    ),
    _fact(
        "User is from Lille and still travels there often for family weekends and errands.",
        target="user",
        category="user_pref",
        topic="personal-profile",
        importance=7,
        subject_key="user:origin",
        value="lille",
    ),
    _fact(
        "User's timezone is Europe/Paris and meetings should be scheduled against CET or CEST.",
        target="user",
        category="user_pref",
        topic="user-profile",
        importance=8,
        subject_key="user:timezone",
        value="europe-paris",
    ),
    _fact(
        "User prefers concise responses that front-load the answer before any extra explanation.",
        target="user",
        category="user_pref",
        topic="user-preferences",
        importance=8,
        subject_key="user:response_style",
        value="concise",
    ),
    _fact(
        "User prefers paragraph-style answers when tradeoffs need context and rationale together.",
        target="user",
        category="user_pref",
        topic="user-preferences",
        importance=7,
        subject_key="user:answer_format",
        value="paragraphs",
    ),
    _fact(
        "User likes Neovim with keyboard-first workflows and very few modal prompts or popups.",
        target="user",
        category="user_pref",
        topic="user-preferences",
        importance=7,
    ),
    _fact(
        "User dislikes verbose output that repeats obvious steps after the key result is already known.",
        target="user",
        category="user_pref",
        topic="user-preferences",
        importance=7,
    ),
    _fact(
        "User's favorite drink is espresso and coffee examples land better than tea analogies.",
        target="user",
        category="user_pref",
        topic="user-preferences",
        importance=6,
        subject_key="user:favorite:drink",
        value="espresso",
    ),
    _fact(
        "User pronouns are he/him and profile text should not default to plural pronouns.",
        target="user",
        category="user_pref",
        topic="personal-profile",
        importance=7,
        subject_key="user:pronouns",
        value="he-him",
    ),
    _fact(
        "User likes short commit messages that still mention the concrete subsystem or file family.",
        target="user",
        category="user_pref",
        topic="user-preferences",
        importance=6,
    ),
    _fact(
        "User dislikes emoji in commit messages because production history should stay plain and grepable.",
        target="user",
        category="user_pref",
        topic="user-preferences",
        importance=6,
    ),
    _fact(
        "User is allergic to peanuts and any food examples should avoid peanut butter shortcuts.",
        target="user",
        category="user_pref",
        topic="personal-profile",
        importance=8,
        subject_key="user:allergy:peanuts",
        value="allergic",
    ),
    _fact(
        "Environment runs Ubuntu 24.04 with Docker, Podman, and OpenSSH already installed.",
        target="memory",
        category="environment",
        topic="environment",
        importance=7,
        subject_key="environment:os",
        value="ubuntu-24-04",
    ),
    _fact(
        "Environment shell is zsh with aliases for pytest, docker compose, and journalctl tailing.",
        target="memory",
        category="environment",
        topic="environment",
        importance=7,
        subject_key="environment:shell",
        value="zsh",
    ),
    _fact(
        "Environment editor is Neovim with Treesitter, Telescope, and a Python linting preset.",
        target="memory",
        category="environment",
        topic="environment",
        importance=6,
        subject_key="environment:editor",
        value="neovim",
    ),
    _fact(
        "Environment uses WSL2 on the laptop, but long-running services live on a Linux workstation.",
        target="memory",
        category="environment",
        topic="environment",
        importance=6,
        subject_key="environment:wsl",
        value="wsl2",
    ),
    _fact(
        "SSH uses port 2222 on the staging box because the default port is filtered upstream.",
        target="memory",
        category="environment",
        topic="environment",
        importance=7,
        subject_key="environment:ssh_port",
        value="2222",
    ),
    _fact(
        "SSH key path: ~/.ssh/workstation_ed25519 and the staging host rejects the old RSA key.",
        target="memory",
        category="environment",
        topic="environment",
        importance=6,
        subject_key="environment:ssh_key_path",
        value="workstation-ed25519",
    ),
    _fact(
        "Project root is ~/code/myapi and the main HTTP service lives in apps/api for local runs.",
        target="memory",
        category="project",
        topic="project-context",
        importance=7,
    ),
    _fact(
        "Project uses Axum for the API layer and keeps request validation inside typed extractors.",
        target="memory",
        category="project",
        topic="project-stack",
        importance=6,
    ),
    _fact(
        "Project uses SQLx with offline query checking and migration files committed to version control.",
        target="memory",
        category="project",
        topic="project-data",
        importance=7,
    ),
    _fact(
        "Primary project database is PostgreSQL because staging and production must stay query-compatible.",
        target="memory",
        category="project",
        topic="project-data",
        importance=8,
        subject_key="project:database",
        value="postgresql",
    ),
    _fact(
        "Project uses Redis for caching and the cache namespace mirrors the API service name.",
        target="memory",
        category="project",
        topic="project-data",
        importance=7,
        subject_key="project:cache_backend",
        value="redis",
    ),
    _fact(
        "Project tests run with pytest -x so the first failing case stops the batch immediately.",
        target="memory",
        category="project",
        topic="project-delivery",
        importance=8,
        subject_key="project:test_command",
        value="pytest-x",
    ),
    _fact(
        "Project deploys with Docker Compose from ops/deploy and keeps per-host overrides in env files.",
        target="memory",
        category="project",
        topic="project-delivery",
        importance=8,
        subject_key="project:deploy_method",
        value="docker-compose",
    ),
    _fact(
        "Project CI runs on GitHub Actions with a separate lint matrix for Python and frontend jobs.",
        target="memory",
        category="project",
        topic="project-delivery",
        importance=6,
    ),
    _fact(
        "Project uses Python 3.12 for tooling even though the API service itself is implemented in Rust.",
        target="memory",
        category="project",
        topic="project-stack",
        importance=6,
    ),
    _fact(
        "Project uses FastAPI for an internal admin surface that shares auth helpers with the Rust API.",
        target="memory",
        category="project",
        topic="project-stack",
        importance=6,
    ),
    _fact(
        "Project uses React on the frontend and keeps component tests beside each feature directory.",
        target="memory",
        category="project",
        topic="project-stack",
        importance=6,
    ),
    _fact(
        "Project uses TypeScript with strict mode and rejects any commits that widen the tsconfig.",
        target="memory",
        category="project",
        topic="project-stack",
        importance=6,
    ),
    _fact(
        "Project database migrations run with Alembic for the admin surface and SQLx for the Rust API.",
        target="memory",
        category="project",
        topic="project-data",
        importance=6,
    ),
    _fact(
        "Staging API listens on port 8081 and the health endpoint is exposed without frontend auth.",
        target="memory",
        category="environment",
        topic="environment",
        importance=6,
        subject_key="environment:service_port",
        value="8081",
    ),
    _fact(
        "Do not use sudo for Docker commands because the user is already in the docker group.",
        target="memory",
        category="workflow",
        topic="workflow-rules",
        importance=9,
        subject_key="workflow:docker_sudo",
        value="no-sudo",
    ),
    _fact(
        "Use apply_patch for manual file edits so every change stays reviewable and easy to diff.",
        target="memory",
        category="workflow",
        topic="workflow-rules",
        importance=9,
        subject_key="workflow:manual_edits",
        value="apply-patch",
    ),
    _fact(
        "Never use git reset --hard unless the user explicitly asks for destructive history rewriting.",
        target="memory",
        category="workflow",
        topic="workflow-rules",
        importance=9,
        subject_key="workflow:git_safety",
        value="avoid-git-reset-hard",
    ),
    _fact(
        "Project formatting uses black with a stable line length so generated diffs stay compact.",
        target="memory",
        category="workflow",
        topic="workflow-rules",
        importance=6,
    ),
    _fact(
        "Project linting uses ruff with import sorting enabled and warnings promoted before releases.",
        target="memory",
        category="workflow",
        topic="workflow-rules",
        importance=6,
    ),
    _fact(
        "Project commits follow conventional commits because release notes are generated from git history.",
        target="memory",
        category="workflow",
        topic="workflow-rules",
        importance=6,
    ),
    _fact(
        "Feature branches use the prefix feature/ and hotfix branches use the prefix hotfix/.",
        target="memory",
        category="workflow",
        topic="workflow-rules",
        importance=6,
    ),
    _fact(
        "Project API base URL is http://localhost:8000 and auth cookies are scoped to localhost.",
        target="memory",
        category="project",
        topic="project-context",
        importance=6,
    ),
    _fact(
        "The dev database DSN uses localhost port 5432 and the test DSN uses localhost port 55432.",
        target="memory",
        category="environment",
        topic="environment",
        importance=6,
    ),
    _fact(
        "The worker queue is RabbitMQ and queue names are prefixed with myapi- for every environment.",
        target="memory",
        category="project",
        topic="project-delivery",
        importance=6,
    ),
    _fact(
        "Logs are stored in /var/log/myapi and logrotate keeps seven compressed daily archives.",
        target="memory",
        category="environment",
        topic="environment",
        importance=6,
    ),
    _fact(
        "Backups run every Sunday at 02:00 and restore drills are logged in docs/ops/backups.md.",
        target="memory",
        category="workflow",
        topic="workflow-rules",
        importance=6,
    ),
    _fact(
        "Release notes live in docs/releases and each shipped tag gets one short operator summary.",
        target="memory",
        category="project",
        topic="project-context",
        importance=6,
    ),
    _fact(
        "The frontend package manager is pnpm and lockfile churn should stay isolated from backend work.",
        target="memory",
        category="project",
        topic="project-stack",
        importance=6,
    ),
    _fact(
        "The backend virtualenv lives in .venv and local tooling should prefer uv over pip when possible.",
        target="memory",
        category="environment",
        topic="environment",
        importance=6,
    ),
    _fact(
        "The project default branch is main and release branches are cut only from reviewed merge commits.",
        target="memory",
        category="workflow",
        topic="workflow-rules",
        importance=6,
    ),
    _fact(
        "The integration test marker is slow and CI excludes it unless the nightly matrix is running.",
        target="memory",
        category="project",
        topic="project-delivery",
        importance=6,
    ),
    _fact(
        "The staging host is api-staging.internal and DNS there can lag after fresh certificate issuance.",
        target="memory",
        category="environment",
        topic="environment",
        importance=6,
    ),
)


def generate_scale_facts(count: int = 50) -> list[FactSeed]:
    facts = list(BASE_SCALE_FACTS)
    if count <= len(facts):
        return facts[:count]
    facts.extend(_synthetic_facts(start=len(facts) + 1, total=count - len(facts)))
    return facts[:count]


def generate_overflow_facts(count: int = 200) -> list[FactSeed]:
    return generate_scale_facts(count)


def generate_recall_sessions() -> list[SessionSeed]:
    base = generate_scale_facts(50)
    session_one = SessionSeed(
        session_id="bench-recall-s1",
        facts=tuple(base[0:10]),
    )
    session_two = SessionSeed(
        session_id="bench-recall-s2",
        facts=tuple(base[13:23]),
    )
    session_three = SessionSeed(
        session_id="bench-recall-s3",
        facts=tuple(base[32:42]),
    )
    return [session_one, session_two, session_three]


def generate_contradiction_pairs() -> list[ContradictionPair]:
    rows = [
        (
            ("Environment editor is Vim and most examples assume classic modal keybindings.", "vim"),
            ("Environment editor is Neovim and plugin-aware workflows are preferred now.", "neovim"),
            "environment:editor",
            "environment",
            "environment",
        ),
        (
            ("Environment shell is bash and old scripts still assume bash arrays.", "bash"),
            ("Environment shell is zsh and new shell snippets should target zsh first.", "zsh"),
            "environment:shell",
            "environment",
            "environment",
        ),
        (
            ("User lives in Paris and commutes to the workshop from the city center.", "paris"),
            ("User lives in Cambrai and local references should use northern France examples.", "cambrai"),
            "user:location:current",
            "user_pref",
            "personal-profile",
        ),
        (
            ("User's timezone is UTC and deadlines should be phrased in universal time.", "utc"),
            ("User's timezone is Europe/Paris and local scheduling should use CET or CEST.", "europe-paris"),
            "user:timezone",
            "user_pref",
            "user-profile",
        ),
        (
            ("Primary project database is SQLite for fast local prototyping.", "sqlite"),
            ("Primary project database is PostgreSQL for parity with staging and production.", "postgresql"),
            "project:database",
            "project",
            "project-data",
        ),
        (
            ("Project deploys with Docker Compose from the ops directory on the same host.", "docker-compose"),
            ("Project deploys with Nomad and jobs are updated through a release pipeline.", "nomad"),
            "project:deploy_method",
            "project",
            "project-delivery",
        ),
        (
            ("Use sudo for Docker commands because the socket is not group-accessible.", "sudo-required"),
            ("Do not use sudo for Docker commands because the user is already in the docker group.", "no-sudo"),
            "workflow:docker_sudo",
            "workflow",
            "workflow-rules",
        ),
        (
            ("Manual file edits use vim directly when changing one small line in place.", "vim"),
            ("Use apply_patch for manual file edits so every change stays reviewable.", "apply-patch"),
            "workflow:manual_edits",
            "workflow",
            "workflow-rules",
        ),
        (
            ("Project tests run with pytest -q to keep the terminal as quiet as possible.", "pytest-q"),
            ("Project tests run with pytest -x so failures stop immediately during validation.", "pytest-x"),
            "project:test_command",
            "project",
            "project-delivery",
        ),
        (
            ("Project uses Memcached for caching because no persistence is required there.", "memcached"),
            ("Project uses Redis for caching because the queue workers already depend on it.", "redis"),
            "project:cache_backend",
            "project",
            "project-data",
        ),
    ]
    pairs: list[ContradictionPair] = []
    for earlier, later, subject_key, category, topic in rows:
        target = "user" if subject_key.startswith("user:") else "memory"
        pairs.append(
            ContradictionPair(
                earlier=_fact(
                    earlier[0],
                    target=target,
                    category=category,
                    topic=topic,
                    importance=8,
                    subject_key=subject_key,
                    value=earlier[1],
                ),
                later=_fact(
                    later[0],
                    target=target,
                    category=category,
                    topic=topic,
                    importance=8,
                    subject_key=subject_key,
                    value=later[1],
                ),
            )
        )
    return pairs


def generate_signal_noise_facts() -> tuple[list[FactSeed], list[FactSeed]]:
    useful_facts = list(generate_scale_facts(20))
    noise_texts = [
        "Noise note 01: user asked about Python in a casual one-off conversation.",
        "Noise note 02: the sky is blue when the weather is clear.",
        "Noise note 03: user said hello at the start of the session.",
        "Noise note 04: rain is wet and puddles form on the ground.",
        "Noise note 05: user once asked for a summary of a traceback.",
        "Noise note 06: coffee can be hot and tea can be warm.",
        "Noise note 07: user asked what time it was during a short chat.",
        "Noise note 08: logs sometimes contain timestamps and service names.",
        "Noise note 09: the build printed a warning with no action needed.",
        "Noise note 10: a Python example mentioned dictionaries and lists.",
        "Noise note 11: user asked about Docker in a generic tutorial context.",
        "Noise note 12: terminals can display text in many colors.",
        "Noise note 13: user said thanks after the answer was complete.",
        "Noise note 14: the benchmark machine has files and directories.",
        "Noise note 15: a browser tab was opened during unrelated research.",
        "Noise note 16: a command finished successfully and returned zero.",
        "Noise note 17: the conversation included a general note about errors.",
        "Noise note 18: user once asked what JSON means in broad terms.",
        "Noise note 19: a session contained small talk before the task began.",
        "Noise note 20: the phrase hello world appears in many tutorials.",
    ]
    noise_facts = [
        _fact(text, target="memory", category="general", topic="general", importance=2, useful=False)
        for text in noise_texts
    ]
    return useful_facts, noise_facts


def generate_salience_facts() -> list[FactSeed]:
    facts: list[FactSeed] = []
    for index in range(30):
        facts.append(
            _fact(
                f"CRITICAL: workflow rule {index:02d} must always be remembered before making shell edits.",
                target="memory",
                category="workflow",
                topic="workflow-rules",
                importance=9,
                tier="high",
                observed_days_ago=90,
            )
        )
    for index in range(20):
        facts.append(
            _fact(
                f"Project fact {index:02d}: the service keeps one medium-priority convention for deployment docs.",
                target="memory",
                category="project",
                topic="project-context",
                importance=6,
                tier="medium",
                observed_days_ago=90,
            )
        )
    for index in range(20):
        facts.append(
            _fact(
                f"Environment fact {index:02d}: one medium-priority machine detail is worth retaining for support.",
                target="memory",
                category="environment",
                topic="environment",
                importance=5,
                tier="medium",
                observed_days_ago=90,
            )
        )
    for index in range(30):
        facts.append(
            _fact(
                f"By the way, low-priority note {index:02d} was mentioned casually with no follow-up value.",
                target="memory",
                category="general",
                topic="general",
                importance=3,
                tier="low",
                observed_days_ago=90,
            )
        )
    return facts


def _synthetic_facts(*, start: int, total: int) -> list[FactSeed]:
    rng = random.Random(DEFAULT_SEED)
    targets = ["memory", "memory", "memory", "user"]
    categories = ["project", "environment", "workflow", "user_pref"]
    topics = {
        "project": ["project-context", "project-stack", "project-data", "project-delivery"],
        "environment": ["environment"],
        "workflow": ["workflow-rules"],
        "user_pref": ["user-preferences", "personal-profile"],
    }
    templates = {
        "project": "Synthetic project fact {index:03d}: service shard {shard} keeps convention {detail} for repeatable delivery.",
        "environment": "Synthetic environment fact {index:03d}: host profile {shard} uses detail {detail} during local support work.",
        "workflow": "Synthetic workflow fact {index:03d}: operator checklist {shard} requires detail {detail} before release steps.",
        "user_pref": "Synthetic user fact {index:03d}: preference bundle {shard} keeps detail {detail} for future conversations.",
    }
    detail_pool = [
        "alpha-cache",
        "beta-index",
        "gamma-queue",
        "delta-lint",
        "omega-archive",
        "theta-report",
        "sigma-diff",
        "lambda-build",
    ]
    facts: list[FactSeed] = []
    for offset in range(total):
        index = start + offset
        target = targets[offset % len(targets)]
        category = categories[offset % len(categories)]
        topic = topics[category][offset % len(topics[category])]
        shard = chr(97 + (offset % 26))
        detail = detail_pool[rng.randrange(len(detail_pool))]
        text = templates[category].format(index=index, shard=shard, detail=detail)
        facts.append(
            _fact(
                text,
                target=target,
                category=category,
                topic=topic,
                importance=4 if category != "workflow" else 5,
            )
        )
    return facts
