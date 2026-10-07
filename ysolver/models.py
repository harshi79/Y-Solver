"""Domain models for solve jobs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class JobStatus(str, Enum):
    PENDING = "pending"          # queued, not picked up yet
    PROCESSING = "processing"    # a worker is on it
    READY = "ready"              # solution available
    FAILED = "failed"            # unsolvable / error


@dataclass
class Job:
    id: str
    method: str
    charset: str
    status: JobStatus = JobStatus.PENDING
    text: Optional[str] = None
    confidence: Optional[float] = None
    backend: Optional[str] = None
    error_code: Optional[str] = None
    error_text: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    solve_ms: Optional[int] = None
    tries: int = 0
    key_hint: str = ""
    image_sha: Optional[str] = None

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.created_at)

    def to_public(self) -> dict:
        """Shape used by the dashboard / native API (never leaks the key)."""
        return {
            "id": self.id,
            "method": self.method,
            "status": self.status.value,
            "text": self.text,
            "confidence": self.confidence,
            "backend": self.backend,
            "errorCode": self.error_code,
            "errorText": self.error_text,
            "solveMs": self.solve_ms,
            "queuedMs": int((self.started_at - self.created_at) * 1000)
            if self.started_at
            else None,
            "createdAt": self.created_at,
            "keyHint": self.key_hint,
        }


ERROR_MESSAGES = {
    "ERROR_WRONG_USER_KEY": "The API key is not recognised by this server.",
    "ERROR_WRONG_CAPTCHA_ID": "Unknown or malformed captcha id.",
    "ERROR_BAD_ACTION": "Unknown action for this endpoint.",
    "ERROR_KEY_DOES_NOT_EXIST": "The API key is not recognised by this server.",
    "ERROR_ZERO_BALANCE": "Balance exhausted (self-hosted servers usually never are).",
    "ERROR_NO_SLOT_AVAILABLE": "Queue is full, retry in a moment.",
    "ERROR_WRONG_FILE_EXTENSION": "Unsupported image format.",
    "ERROR_IMAGE_TYPE_NOT_SUPPORTED": "Could not decode the supplied image.",
    "ERROR_CAPTCHA_UNSOLVABLE": "The solver could not read this captcha.",
    "ERROR_METHOD_NOT_SUPPORTED": (
        "This method is not implemented by Y-Solver. Image-to-text captchas "
        "(method=base64/post, ImageToTextTask) are supported."
    ),
    "ERROR_BAD_PARAMETERS": "Missing or malformed parameters.",
    "ERROR_INTERNAL": "Internal solver error.",
}
