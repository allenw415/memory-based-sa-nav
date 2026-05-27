from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

DEFAULT_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def load_dotenv(dotenv_path: str | Path) -> None:
    path = Path(dotenv_path)
    if not path.exists():
        return

    initial_keys = set(os.environ)
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value and ((value[0] == value[-1]) and value[0] in {"'", '"'}):
            value = value[1:-1]
        if key in initial_keys:
            continue
        os.environ[key] = value


@dataclass(frozen=True)
class ModelEnvironment:
    provider: str | None = None
    model_name: str | None = None
    api_key: str | None = None
    api_base: str | None = None
    api_kind: str | None = None
    request_timeout: float | None = None
    num_ctx: int | None = None
    temperature: float | None = None
    active_profile: str | None = None


def resolve_model_environment(
    *,
    default_model: str | None = None,
    default_api_base: str | None = None,
    default_api_kind: str | None = None,
    profile: str | None = None,
) -> ModelEnvironment:
    active_profile = (profile or os.environ.get("ST_NAV_ACTIVE_PROFILE") or os.environ.get("ST_NAV_PROFILE") or "").strip()

    def lookup(name: str, *legacy_names: str) -> str | None:
        candidates: list[str] = []
        if active_profile:
            normalized_profile = _normalize_profile_name(active_profile)
            candidates.append(f"ST_NAV_PROFILE_{normalized_profile}_{name}")
        candidates.append(f"ST_NAV_{name}")
        candidates.extend(legacy_names)
        for candidate in candidates:
            value = os.environ.get(candidate)
            if value is not None and value.strip():
                return value.strip()
        return None

    provider = lookup("MODEL_PROVIDER")
    api_key = lookup("API_KEY", "OPENAI_API_KEY")
    if api_key is None and provider in {"gemini", "gemini_api", "google_gemma_api"}:
        api_key = lookup("API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")
    api_base = lookup("API_BASE")
    if api_base is None and provider in {"gemini", "gemini_api", "google_gemma_api"}:
        api_base = DEFAULT_GEMINI_API_BASE

    return ModelEnvironment(
        provider=provider,
        model_name=lookup("MODEL_NAME") or default_model,
        api_key=api_key,
        api_base=api_base or default_api_base,
        api_kind=lookup("API_KIND") or default_api_kind,
        request_timeout=_parse_float(lookup("REQUEST_TIMEOUT")),
        num_ctx=_parse_int(lookup("NUM_CTX")),
        temperature=_parse_float(lookup("TEMPERATURE")),
        active_profile=active_profile or None,
    )



def resolve_task_num_ctx(
    task_name: str,
    *,
    explicit_num_ctx: int | None = None,
    fallback_num_ctx: int | None = None,
    default_num_ctx: int | None = None,
) -> int | None:
    if explicit_num_ctx is not None:
        return int(explicit_num_ctx)

    normalized_task_name = _normalize_profile_name(task_name)
    if normalized_task_name:
        task_value = _parse_int(os.environ.get(f"ST_NAV_{normalized_task_name}_NUM_CTX"))
        if task_value is not None:
            return task_value

    if fallback_num_ctx is not None:
        return int(fallback_num_ctx)
    if default_num_ctx is not None:
        return int(default_num_ctx)
    return None


def _normalize_profile_name(value: str) -> str:
    normalized = []
    for char in value.strip().upper():
        normalized.append(char if char.isalnum() else "_")
    return "".join(normalized).strip("_")


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
