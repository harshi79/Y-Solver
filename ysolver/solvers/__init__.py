"""Solver backends. Pick one with ``YSOLVER_BACKEND`` (default ``auto``)."""

from __future__ import annotations

from .base import Engine, EngineResult, SolverError, SolverResult
from .registry import CaptchaSolver, engine_status, get_engine, resolve_backend

__all__ = [
    "CaptchaSolver",
    "Engine",
    "EngineResult",
    "SolverError",
    "SolverResult",
    "engine_status",
    "get_engine",
    "resolve_backend",
]
