"""Y-Solver — free, self-hosted CAPTCHA solving API.

The HTTP surface is intentionally compatible with the paid solving services
(``in.php`` / ``res.php`` and the newer ``createTask`` / ``getTaskResult``
JSON API) so an existing client only needs its base URL changed.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
