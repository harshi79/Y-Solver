"""Wire-protocol parsing shared by ``in.php`` and the JSON task API.

Keeping the messy "what did this client actually send" logic in one module is
the only reason the drop-in compatibility stays maintainable.
"""

from __future__ import annotations

import base64
import binascii
import re
from typing import Any, Dict, Iterable, Mapping, Optional

from .models import ERROR_MESSAGES

TRUTHY = {"1", "true", "yes", "on", "y"}

#: ``in.php`` style image uploads.
TEXT_METHODS = {"base64", "post", "file", "image", "imagetotext", "imagetotexttask", "text"}
#: Interactive challenges. These need a human/browser; an image OCR server
#: cannot solve them, and pretending otherwise would be dishonest.
INTERACTIVE_METHODS = {
    "userrecaptcha", "recaptcha", "recaptcha2", "recaptcha3", "recaptchav2",
    "recaptchav3", "hcaptcha", "turnstile", "cloudflare", "geetest",
    "geetestv3", "geetestv4", "funcaptcha", "arkoselabs", "datadome",
    "capy", "lemin", "amazon", "mtcaptcha", "friendlycaptcha", "prosopo",
    "atb_captcha", "visionengine",
}

#: Substrings that mark an interactive challenge whatever the client called it.
_INTERACTIVE_TOKENS = (
    "recaptcha", "hcaptcha", "turnstile", "geetest", "funcaptcha", "arkose",
    "datadome", "amazon", "capy", "lemin", "mtcaptcha", "friendlycaptcha",
    "prosopo", "cloudflare", "atb_captcha", "visionengine",
)

BASE64_RE = re.compile(r"^[A-Za-z0-9+/:_\-\s=]+$")
JOB_ID_RE = re.compile(r"^\d{6,20}$")


class ProtocolError(Exception):
    def __init__(self, code: str, message: Optional[str] = None):
        self.code = code
        self.message = message or ERROR_MESSAGES.get(code, code)
        super().__init__(f"{self.code}: {self.message}")


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in TRUTHY


def first(mapping: Mapping[str, Any], names: Iterable[str]) -> Optional[str]:
    """First non-empty value among ``names`` (case-insensitive keys)."""
    lowered = {str(k).lower(): v for k, v in mapping.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return None


def mask_key(key: Optional[str]) -> str:
    if not key:
        return ""
    if len(key) <= 4:
        return "*" * len(key)
    return f"{key[:2]}***{key[-2:]}"


_IMAGE_SIGNATURES = (b"\x89PNG", b"GIF8", b"\xff\xd8", b"BM", b"RIFF", b"II*\x00", b"MM\x00*")


def looks_like_image(data: bytes) -> bool:
    return any(data.startswith(signature) for signature in _IMAGE_SIGNATURES)


def _b64_variants(raw: str):
    """Yield candidate normalisations of a base64 payload, best first.

    Clients send base64 through query strings and urlencoded forms, where ``+``
    is decoded to a space — so a payload containing spaces has to be rebuilt
    with ``+`` rather than having the whitespace stripped away. Newline-wrapped
    base64 (pasted from a script or an email) is the opposite case and is
    handled by stripping. Both are tried; an image signature decides.
    """
    stripped = re.sub(r"\s+", "", raw)
    rebuilt = re.sub(r"\s+", "+", raw.strip())
    if rebuilt != stripped and " " in raw.strip():
        yield rebuilt
        yield stripped
    else:
        yield stripped
        if rebuilt != stripped:
            yield rebuilt


def decode_base64_payload(payload: str) -> bytes:
    """Accept raw base64, data URLs, URL-safe and whitespace-mangled variants."""
    raw = (payload or "").strip()
    if raw.startswith("data:"):
        match = re.match(r"^data:[^;,]*;base64,(.*)$", raw, flags=re.DOTALL)
        raw = match.group(1) if match else raw.split(",", 1)[-1]
    raw = raw.replace("-", "+").replace("_", "/")

    fallback: Optional[bytes] = None
    for variant in _b64_variants(raw):
        if not BASE64_RE.match(variant):
            continue
        try:
            decoded = base64.b64decode(variant + "=" * ((-len(variant)) % 4), validate=False)
        except (binascii.Error, ValueError):
            continue
        if looks_like_image(decoded):
            return decoded
        if fallback is None:
            fallback = decoded
    if fallback is not None:
        return fallback
    raise ProtocolError("ERROR_WRONG_FILE_EXTENSION", "image payload is not valid base64")


def classify_method(method: Optional[str], body_kind: Optional[str]) -> str:
    """Map a client's ``method``/``task.type`` onto our three buckets."""
    if body_kind in ("file", "base64"):
        return "image"
    name = (method or "").strip().lower()
    if not name:
        return "image"  # legacy clients omitted method for the plain image API
    if name in INTERACTIVE_METHODS or any(token in name for token in _INTERACTIVE_TOKENS):
        return "interactive"
    if name in TEXT_METHODS:
        return "image"
    return "unknown"


def describe_unsupported(name: Optional[str]) -> str:
    """Explain a refusal in terms the caller can act on (see ysolver/scope.py)."""
    from .scope import explain

    label = (name or "this method").strip() or "this method"
    return f"Cannot solve {label!r} here. {explain(label)}"


def normalize_id(job_id: Optional[str]) -> str:
    if not job_id:
        raise ProtocolError("ERROR_BAD_PARAMETERS", "no captcha id supplied")
    cleaned = job_id.strip()
    if not JOB_ID_RE.match(cleaned):
        raise ProtocolError("ERROR_WRONG_CAPTCHA_ID", f"{cleaned!r} is not a valid job id")
    return cleaned


def extract_options(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Pull the knobs we honour out of a request (everything else is ignored)."""
    numeric = truthy(first(params, ("numeric", "numerals", "digits", "numeric_only")))
    charset = first(params, ("charset", "characters", "alphabet", "whitelist")) or ""
    return {
        "charset": charset,
        "numeric": numeric,
        "min_len": _as_int(first(params, ("min_len", "minlength"))),
        "max_len": _as_int(first(params, ("max_len", "maxlength"))),
        "comment": first(params, ("comment", "textinstructions")) or "",
    }


def enforce_length(text: str, min_len: Optional[int], max_len: Optional[int]) -> str:
    """Best-effort length hint: pad/trim only when the caller was explicit."""
    if min_len and len(text) < min_len:
        text = text.ljust(min_len, text[-1] if text else "0")
    if max_len and len(text) > max_len:
        text = text[:max_len]
    return text


def _as_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except ValueError:
        return None


def parse_task(task: Any) -> Dict[str, Any]:
    """Normalise ``createTask``'s ``task`` object (or a flat submission)."""
    if isinstance(task, str):
        try:
            import json

            task = json.loads(task)
        except Exception as exc:
            raise ProtocolError("ERROR_BAD_PARAMETERS", "task must be a JSON object") from exc
    if task is None:
        return {}
    if not isinstance(task, Mapping):
        raise ProtocolError("ERROR_BAD_PARAMETERS", "task must be an object")
    lowered = {str(k).lower(): v for k, v in task.items()}
    out: Dict[str, Any] = dict(lowered)
    body = first(lowered, ("body", "image", "captcha", "base64"))
    if body:
        out["image_base64"] = body
    return out


def job_id_is_valid(job_id: Optional[str]) -> bool:
    return bool(job_id) and bool(JOB_ID_RE.match(job_id.strip()))
