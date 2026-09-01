"""Configuration, read exclusively from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

# Multilingual NLI model, works on German and English. Swappable via MODEL_ID,
# but the chosen model has to be baked into the image at build time.
MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

# The service targets German-language text, hence the German default template.
# Override per request or via DEFAULT_HYPOTHESIS_TEMPLATE.
DEFAULT_TEMPLATE = "Diese Nachricht betrifft {}."


class ConfigError(RuntimeError):
    """Invalid or missing configuration - the service refuses to start."""


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got: {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got: {value}")
    return value


def _api_key_from_env() -> str:
    key = os.environ.get("API_KEY", "")
    if not key.strip():
        raise ConfigError("API_KEY is not set - refusing to start.")
    return key


@dataclass(frozen=True)
class Settings:
    api_key: str
    num_threads: int
    max_concurrency: int
    default_hypothesis_template: str
    port: int
    model_id: str = MODEL_ID

    @classmethod
    def from_env(cls) -> "Settings":
        template = os.environ.get("DEFAULT_HYPOTHESIS_TEMPLATE") or DEFAULT_TEMPLATE
        if "{}" not in template:
            raise ConfigError("DEFAULT_HYPOTHESIS_TEMPLATE must contain '{}'.")

        return cls(
            api_key=_api_key_from_env(),
            num_threads=_int_env("NUM_THREADS", os.cpu_count() or 1),
            max_concurrency=_int_env("MAX_CONCURRENCY", 2),
            default_hypothesis_template=template,
            port=_int_env("PORT", 8000),
            model_id=(os.environ.get("MODEL_ID") or MODEL_ID).strip(),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
