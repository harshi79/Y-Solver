"""Engine contracts shared by every solver backend."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

from ..models import ERROR_MESSAGES
from ..preprocess import Prepared


@dataclass
class EngineResult:
    text: str
    confidence: Optional[float] = None


@dataclass
class SolverResult:
    text: str
    confidence: Optional[float]
    backend: str
    engine_ms: int


class SolverError(Exception):
    """A failure that maps onto a wire protocol error string."""

    def __init__(self, code: str, message: Optional[str] = None):
        self.code = code
        self.message = message or ERROR_MESSAGES.get(code, "solver error")
        super().__init__(f"{self.code}: {self.message}")


class Engine(abc.ABC):
    """A captcha-reading strategy."""

    name: str = "engine"
    description: str = ""
    install_hint: str = ""

    @abc.abstractmethod
    def available(self) -> bool:
        """Whether this engine can run in the current environment."""

    @abc.abstractmethod
    def solve(self, prepared: Prepared, charset: str) -> EngineResult:
        """Read the captcha. Raise :class:`SolverError` when unreadable."""

    def info(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "available": self.available(),
            "installHint": self.install_hint,
        }
