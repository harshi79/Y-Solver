"""What this server can and cannot solve — stated once, used everywhere.

The API refuses unsupported challenges with ``ERROR_METHOD_NOT_SUPPORTED``; this
module is where the *why* lives, so the refusal can carry a real explanation
instead of a shrug, and so ``GET /api/scope`` can answer the question
programmatically for whatever tool is wiring Y-Solver up.
"""

from __future__ import annotations

from typing import Dict, List

#: Methods answered by the image engines.
SUPPORTED: Dict[str, str] = {
    "post": "multipart upload (file/image/captcha field)",
    "base64": "base64 payload in the body parameter",
    "file": "file upload",
    "image": "raw image bytes as the request body",
    "imagetotext": "explicit text-in-image task",
    "imagetotexttask": "JSON task API equivalent",
}

#: Challenges that cannot be solved from an image at all, with the reason.
UNSUPPORTED: Dict[str, str] = {
    "recaptcha": (
        "reCAPTCHA v2/v3 returns a token produced by an interactive browser session "
        "and scored by Google's risk analysis. Nothing in the image is the answer, so "
        "an OCR server cannot help. A human or a real browser session is required."
    ),
    "hcaptcha": (
        "hCaptcha is an interactive challenge with proprietary scoring; the tiles are "
        "not the credential — the returned token is."
    ),
    "turnstile": (
        "Cloudflare Turnstile issues a token after a browser-side challenge; there is "
        "no captcha image to read."
    ),
    "geetest": (
        "GeeTest (v3/v4) is a stateful challenge (slide/puzzle/click) with server-side "
        "session validation, not an image to transcribe."
    ),
    "funcaptcha": (
        "Arkose Labs FunCaptcha presents rotating interactive puzzles validated "
        "server-side; image OCR is not the mechanism."
    ),
    "datadome": "DataDome is a bot-management layer, not a captcha image.",
    "mtcaptcha": "MTCaptcha is interactive and token-based.",
    "capy": "Capy is an interactive puzzle with server-side validation.",
}

#: The honest one-liner for API consumers.
HEADLINE = (
    "Y-Solver reads text-in-image captchas. It does not — and cannot — solve "
    "interactive challenges such as reCAPTCHA, hCaptcha, Turnstile or GeeTest: those "
    "are validated by a token from a browser session, not by the picture."
)

_RECOMMENDATIONS = {
    "recaptcha": [
        "Use Google's official test keys in development (they always pass) so your "
        "integration can be exercised without solving anything.",
        "In production, let the user's own browser solve the challenge — that is what "
        "it is for. Y-Solver is not a substitute.",
        "If you are testing your own site, register a separate key for the test "
        "environment rather than automating the production one.",
    ],
    "hcaptcha": [
        "Use the hCaptcha test keys for development and let real users solve production "
        "challenges in the browser.",
    ],
    "_default": [
        "Solve the challenge in the browser, where it is designed to be solved.",
        "If the captcha is actually a distorted-text image, submit it here instead — "
        "that is what this server is for.",
    ],
}


def scope_payload() -> dict:
    """Machine-readable capability statement."""
    return {
        "solves": "text-in-image captchas",
        "headline": HEADLINE,
        "supported": SUPPORTED,
        "unsupported": UNSUPPORTED,
        "unsupportedCount": len(UNSUPPORTED),
        "errorForUnsupported": "ERROR_METHOD_NOT_SUPPORTED",
        "notes": [
            "Interactive challenges are refused rather than guessed at, so a pipeline "
            "fails loudly instead of silently mis-sending a wrong token.",
            "No outbound requests, no telemetry: this server never contacts Google, "
            "Cloudflare or anyone else.",
        ],
    }


def recommendation(method: str) -> List[str]:
    """Practical next steps for a caller who asked for an unsupported method."""
    key = (method or "").lower()
    for name in UNSUPPORTED:
        if name in key:
            return _RECOMMENDATIONS.get(name, _RECOMMENDATIONS["_default"])
    return _RECOMMENDATIONS["_default"]


def explain(method: str) -> str:
    """A refusal message a developer can actually act on."""
    key = (method or "").lower()
    for name, reason in UNSUPPORTED.items():
        if name in key:
            return f"{name}: {reason}"
    return HEADLINE
