#!/usr/bin/env python
"""Solve a captcha through the classic form API (the 2Captcha/Anti-Captcha shape).

Run the server first:  ysolver   (or: python -m ysolver)
Then:                  python examples/solve_with_form_api.py captcha.png
"""

from __future__ import annotations

import base64
import os
import sys
import time

import requests  # pip install requests

BASE = os.environ.get("YSOLVER_URL", "http://localhost:8000")
KEY = os.environ.get("YSOLVER_KEY", "demo")


def solve(path: str, timeout: float = 60.0) -> str:
    with open(path, "rb") as handle:
        payload = base64.b64encode(handle.read()).decode("ascii")

    created = requests.post(
        f"{BASE}/in.php",
        data={"key": KEY, "method": "base64", "body": payload},
        timeout=30,
    ).text
    if not created.startswith("OK|"):
        raise RuntimeError(f"submit failed: {created}")
    job_id = created.split("|", 1)[1]

    deadline = time.time() + timeout
    while time.time() < deadline:
        result = requests.get(
            f"{BASE}/res.php", params={"key": KEY, "action": "get", "id": job_id}, timeout=30
        ).text.strip()
        if result.startswith("OK|"):
            return result.split("|", 1)[1]
        if result != "CAPCHA_NOT_READY":
            raise RuntimeError(f"solve failed: {result}")
        time.sleep(0.5)
    raise TimeoutError(f"job {job_id} did not finish in {timeout}s")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # No image given: fetch a generated sample from the server itself.
        sample = requests.get(f"{BASE}/api/sample", params={"length": 5}, timeout=30).json()
        print("generated sample — ground truth:", sample["label"])
        response = requests.get(
            f"{BASE}/api/solve",
            params={"key": KEY, "body": sample["pngBase64"]},
            timeout=60,
        ).json()
        print("server read it as:", response.get("text"), response.get("status"))
        sys.exit(0)

    print(f"{sys.argv[1]} -> {solve(sys.argv[1])}")
