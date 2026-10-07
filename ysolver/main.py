"""``ysolver`` / ``python -m ysolver`` entry point."""

from __future__ import annotations

import logging

from . import __version__
from .api import create_app
from .config import Settings
from .solvers.registry import engine_status

BANNER = r"""
  __   __        ____        _
  \ \ / /__     / ___|  ___ | |_   _____ _ __
   \ V / _ \    \___ \ / _ \| \ \ / / _ \ '__|
    | | (_) |    ___) | (_) | |\ V /  __/ |
    |_|\___/    |____/ \___/|_| \_/ \___|_|     v{version}
"""


def _print_startup(settings: Settings, backend: str) -> None:
    print(BANNER.format(version=__version__))
    print("  Free, self-hosted CAPTCHA solving API — no per-solve fees, ever.\n")
    print(f"  backend      : {backend}")
    for engine in engine_status():
        mark = "ready" if engine["available"] else "unavailable"
        print(f"    - {engine['name']:<9} {mark:<12} {engine['description']}")
        if not engine["available"] and engine.get("installHint"):
            print(f"      install: {engine['installHint']}")
    print(f"  workers      : {settings.workers}")
    print(f"  storage      : {settings.db_path}")
    if settings.require_key:
        keys = ", ".join(_mask(k) for k in settings.api_keys)
        print(f"  api key(s)   : {keys}")
    else:
        print("  api key(s)   : disabled (YSOLVER_REQUIRE_KEY=0)")
    print(f"  result ttl   : {settings.result_ttl}s")
    print("\n  Endpoints")
    print("    form API   : GET|POST /in.php  ·  GET|POST /res.php")
    print("    task API   : POST /createTask  ·  POST /getTaskResult")
    print("    native API : GET|POST /api/solve  ·  GET /api/jobs  ·  GET /api/stats")
    print(f"    dashboard  : http://localhost:{settings.port}/")
    print()


def _mask(key: str) -> str:
    if len(key) <= 4:
        return "*" * len(key)
    return f"{key[:2]}***{key[-2:]}"


def main() -> None:
    import uvicorn

    settings = Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    app = create_app(settings)
    _print_startup(settings, app.state.solver.backend_name)

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        access_log=False,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
