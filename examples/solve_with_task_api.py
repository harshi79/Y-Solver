#!/usr/bin/env python
"""Solve a captcha through the JSON task API, using the official SDK shape.

If you already use the `2captcha-python` / `anticaptchaofficial` style of
client, only the base URL changes:

    solver = Solver(BASE_URL)   # was https://api.2captcha.com
"""

from __future__ import annotations

import os
import sys
import time

import requests

BASE = os.environ.get("YSOLVER_URL", "http://localhost:8000")
KEY = os.environ.get("YSOLVER_KEY", "demo")


def create_task(image_b64: str, **options) -> str:
    task = {"type": "ImageToTextTask", "body": image_b64, **options}
    payload = requests.post(
        f"{BASE}/createTask", json={"clientKey": KEY, "task": task}, timeout=30
    ).json()
    if payload.get("errorId"):
        raise RuntimeError(f"createTask failed: {payload}")
    return payload["taskId"]


def get_result(task_id: str) -> str:
    payload = requests.post(
        f"{BASE}/getTaskResult", json={"clientKey": KEY, "taskId": task_id}, timeout=30
    ).json()
    if payload.get("errorId"):
        raise RuntimeError(f"getTaskResult failed: {payload}")
    return payload["status"]


def solve(path: str, timeout: float = 60.0, **options) -> str:
    import base64

    with open(path, "rb") as handle:
        task_id = create_task(base64.b64encode(handle.read()).decode("ascii"), **options)

    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = requests.post(
            f"{BASE}/getTaskResult", json={"clientKey": KEY, "taskId": task_id}, timeout=30
        ).json()
        if payload.get("status") == "ready":
            return payload["solution"]["text"]
        if payload.get("errorId"):
            raise RuntimeError(f"solve failed: {payload}")
        time.sleep(0.5)
    raise TimeoutError(f"task {task_id} did not finish in {timeout}s")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit("usage: solve_with_task_api.py captcha.png")
    print(f"{sys.argv[1]} -> {solve(sys.argv[1], numeric=True)}")
