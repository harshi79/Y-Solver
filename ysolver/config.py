"""Runtime configuration, all via environment variables (prefix ``YSOLVER_``)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Tuple

_TRUTHY = {"1", "true", "yes", "on", "y"}


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in _TRUTHY


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


#: Charset used when a request does not specify one. Deliberately excludes
#: letter/digit pairs that classic captchas usually avoid (o/0, l/1/i).
DEFAULT_CHARSET = "abcdefghjkmnpqrstuvwxyz23456789"


@dataclass(frozen=True)
class Settings:
    """Immutable settings snapshot, built once at startup."""

    api_keys: Tuple[str, ...] = ("demo",)
    require_key: bool = True
    backend: str = "auto"
    workers: int = 4
    db_path: str = "data/ysolver.db"
    result_ttl: int = 1800
    max_queue: int = 500
    solve_timeout: float = 30.0
    balance: float = 9999.0
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    charset: str = DEFAULT_CHARSET
    max_image_bytes: int = 6 * 1024 * 1024
    dashboard: bool = True
    store_images: bool = False
    learn: bool = True
    learn_confidence: float = 0.85
    learn_rate: float = 0.15
    field_extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Settings":
        keys = tuple(
            k.strip() for k in _env("YSOLVER_API_KEYS", "demo").split(",") if k.strip()
        )
        return cls(
            api_keys=keys or ("demo",),
            require_key=_env_bool("YSOLVER_REQUIRE_KEY", True),
            backend=_env("YSOLVER_BACKEND", "auto").strip().lower(),
            workers=max(1, _env_int("YSOLVER_WORKERS", 4)),
            db_path=_env("YSOLVER_DB", "data/ysolver.db"),
            result_ttl=max(30, _env_int("YSOLVER_RESULT_TTL", 1800)),
            max_queue=max(1, _env_int("YSOLVER_MAX_QUEUE", 500)),
            solve_timeout=max(1.0, _env_float("YSOLVER_SOLVE_TIMEOUT", 30.0)),
            balance=_env_float("YSOLVER_BALANCE", 9999.0),
            host=_env("YSOLVER_HOST", "0.0.0.0"),
            port=_env_int("YSOLVER_PORT", 8000),
            log_level=_env("YSOLVER_LOG_LEVEL", "info").lower(),
            charset=_env("YSOLVER_CHARSET", DEFAULT_CHARSET),
            max_image_bytes=_env_int("YSOLVER_MAX_IMAGE_BYTES", 6 * 1024 * 1024),
            dashboard=_env_bool("YSOLVER_DASHBOARD", True),
            store_images=_env_bool("YSOLVER_STORE_IMAGES", False),
            learn=_env_bool("YSOLVER_LEARN", True),
            learn_confidence=_env_float("YSOLVER_LEARN_CONFIDENCE", 0.85),
            learn_rate=max(0.0, min(1.0, _env_float("YSOLVER_LEARN_RATE", 0.15))),
        )

    def check_key(self, key: str | None) -> bool:
        if not self.require_key:
            return True
        return bool(key) and key.strip() in self.api_keys
