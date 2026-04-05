from __future__ import annotations

import argparse
import importlib
import importlib.metadata
from pathlib import Path
from typing import Any

from bench_compare.report import build_summary, render_stdout, write_results
from bench_compare.systems import AddonSystem, BaselineSystem, BenchmarkContext, empty_record, resolve_runtime
from bench_compare.utils.hermes_home import DEFAULT_ADDON_CONFIG
from bench_compare.utils.llm import validate_agent_runtime
from bench_compare.utils.wsl import collect_wsl_runtime_seed

DIMENSION_MODULES = {
    "DIM-1": "dim1_retention",
    "DIM-2": "dim2_overflow",
    "DIM-3": "dim3_recall",
    "DIM-4": "dim4_contradictions",
    "DIM-5": "dim5_prefetch",
    "DIM-6": "dim6_salience",
    "DIM-7": "dim7_noise",
}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    try:
        runtime = resolve_runtime()
    except Exception as exc:
        print(f"Benchmark prerequisites unavailable: {exc}")
        return 2
    wsl_settings = build_wsl_settings(args)
    try:
        validate_agent_runtime(repo_root=repo_root, wsl_settings=wsl_settings)
    except Exception as exc:
        print(f"Benchmark prerequisites unavailable: {exc}")
        return 2

    addon_config = build_addon_config(args)
    runtime_seed = collect_runtime_seed(wsl_settings)
    baseline = BaselineSystem(
        hermes_home=Path(args.hermes_home_baseline),
        runtime=runtime,
        runtime_seed=runtime_seed,
    )
    addon = AddonSystem(
        hermes_home=Path(args.hermes_home_addon),
        runtime=runtime,
        addon_config=addon_config,
        runtime_seed=runtime_seed,
    )
    selected_dims = parse_dims(args.dims)
    context = BenchmarkContext(
        repo_root=repo_root,
        model=args.model,
        scale_facts=args.scale_facts,
        overflow_facts=args.overflow_facts,
        timeout_seconds=args.timeout,
        addon_config=addon_config,
        baseline=baseline,
        addon=addon,
        wsl_settings=wsl_settings,
    )

    dimensions: dict[str, dict[str, dict[str, Any]]] = {}
    try:
        for dim_id in selected_dims:
            dimensions[dim_id] = run_dimension(dim_id, context)
    finally:
        baseline.close()
        addon.close()

    total_llm_calls = sum(int(record.get("llm_calls_made") or 0) for systems in dimensions.values() for record in systems.values())
    total_tokens = sum(int(record.get("tokens_estimated") or 0) for systems in dimensions.values() for record in systems.values())
    summary = build_summary(dimensions)
    output_path = Path(args.output) if args.output else default_output_path()
    results = {
        "meta": {
            "model": args.model,
            "hermes_version": detect_hermes_version(),
            "plugin_version": detect_plugin_version(repo_root),
            "timestamp": output_path.stem.replace("bench_results_", ""),
            "total_llm_calls": total_llm_calls,
            "total_tokens_estimated": total_tokens,
            "hermes_home_baseline": str(Path(args.hermes_home_baseline).resolve()),
            "hermes_home_addon": str(Path(args.hermes_home_addon).resolve()),
        },
        "dimensions": dimensions,
        "summary": summary,
    }
    write_results(output_path, results)
    render_stdout(dimensions, total_llm_calls, total_tokens)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hermes memory benchmark: baseline vs consolidating_local")
    parser.add_argument("--model", required=True)
    parser.add_argument("--scale-facts", type=int, default=50)
    parser.add_argument("--overflow-facts", type=int, default=200)
    parser.add_argument("--dims", default=",".join(DIMENSION_MODULES.keys()))
    parser.add_argument("--output")
    parser.add_argument("--hermes-home-baseline", required=True)
    parser.add_argument("--hermes-home-addon", required=True)
    parser.add_argument("--timeout", type=int, default=180)

    parser.add_argument("--extractor-backend", default=DEFAULT_ADDON_CONFIG["extractor_backend"])
    parser.add_argument("--retrieval-backend", default=DEFAULT_ADDON_CONFIG["retrieval_backend"])
    parser.add_argument("--prefetch-limit", type=int, default=DEFAULT_ADDON_CONFIG["prefetch_limit"])
    parser.add_argument("--max-topic-facts", type=int, default=DEFAULT_ADDON_CONFIG["max_topic_facts"])
    parser.add_argument("--topic-summary-chars", type=int, default=DEFAULT_ADDON_CONFIG["topic_summary_chars"])
    parser.add_argument("--session-summary-chars", type=int, default=DEFAULT_ADDON_CONFIG["session_summary_chars"])
    parser.add_argument("--min-hours", type=float, default=DEFAULT_ADDON_CONFIG["min_hours"])
    parser.add_argument("--min-sessions", type=int, default=DEFAULT_ADDON_CONFIG["min_sessions"])
    parser.add_argument("--decay-half-life-days", type=float, default=DEFAULT_ADDON_CONFIG["decay_half_life_days"])
    parser.add_argument("--decay-min-salience", type=float, default=DEFAULT_ADDON_CONFIG["decay_min_salience"])
    parser.add_argument("--episode-body-retention-hours", type=float, default=DEFAULT_ADDON_CONFIG["episode_body_retention_hours"])
    parser.add_argument("--use-wsl", action="store_true")
    parser.add_argument("--wsl-distro", default="Ubuntu")
    parser.add_argument("--wsl-hermes-root", default="~/.hermes/hermes-agent")
    parser.add_argument("--wsl-python", default="")
    return parser.parse_args(argv)


def build_addon_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        **DEFAULT_ADDON_CONFIG,
        "extractor_backend": args.extractor_backend,
        "retrieval_backend": args.retrieval_backend,
        "prefetch_limit": args.prefetch_limit,
        "max_topic_facts": args.max_topic_facts,
        "topic_summary_chars": args.topic_summary_chars,
        "session_summary_chars": args.session_summary_chars,
        "min_hours": args.min_hours,
        "min_sessions": args.min_sessions,
        "decay_half_life_days": args.decay_half_life_days,
        "decay_min_salience": args.decay_min_salience,
        "episode_body_retention_hours": args.episode_body_retention_hours,
    }


def build_wsl_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "enabled": bool(args.use_wsl),
        "distro": args.wsl_distro,
        "hermes_repo_root": args.wsl_hermes_root,
        "python_bin": args.wsl_python,
        "hermes_home": args.hermes_home_baseline,
    }


def collect_runtime_seed(wsl_settings: dict[str, Any]) -> dict[str, Any]:
    if wsl_settings.get("enabled"):
        return collect_wsl_runtime_seed(str(wsl_settings.get("distro") or "Ubuntu"))
    return {}


def parse_dims(raw: str) -> list[str]:
    dims = [item.strip().upper() for item in str(raw or "").split(",") if item.strip()]
    unknown = [dim for dim in dims if dim not in DIMENSION_MODULES]
    if unknown:
        raise SystemExit(f"Unknown dimensions requested: {', '.join(unknown)}")
    return dims


def run_dimension(dim_id: str, context: BenchmarkContext) -> dict[str, dict[str, Any]]:
    module_name = DIMENSION_MODULES[dim_id]
    try:
        module = importlib.import_module(f"bench_compare.dims.{module_name}")
        return module.run(context)
    except Exception as exc:
        injected: list[str] = []
        baseline = empty_record(dim_id, "baseline", injected)
        addon = empty_record(dim_id, "addon", injected)
        baseline["errors"].append(f"{type(exc).__name__}: {exc}")
        addon["errors"].append(f"{type(exc).__name__}: {exc}")
        return {"baseline": baseline, "addon": addon}


def detect_hermes_version() -> str:
    for package_name in ("hermes-agent", "hermes_agent"):
        try:
            return importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def detect_plugin_version(repo_root: Path) -> str:
    plugin_yaml = repo_root / "plugins" / "memory" / "consolidating_local" / "plugin.yaml"
    if not plugin_yaml.exists():
        return "unknown"
    for line in plugin_yaml.read_text(encoding="utf-8").splitlines():
        if line.startswith("version:"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def default_output_path() -> Path:
    from time import strftime, gmtime

    stamp = strftime("%Y%m%dT%H%M%SZ", gmtime())
    return Path.cwd() / f"bench_results_{stamp}.json"


if __name__ == "__main__":
    raise SystemExit(main())
