"""Logging setup.

Two formats are available, chosen with ``LOG_FORMAT``:

* ``text`` — readable single lines, for local development.
* ``json`` — one JSON object per line, for log aggregators.

Both include a request id when one is bound, so the lines belonging to a single
request can be grouped after the fact.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from typing import Any

from app.config import settings

# Bound by the request-id middleware; read by both formatters below.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

# Noisy third-party loggers that would otherwise dominate the output at DEBUG.
QUIET_LOGGERS = {
    "urllib3": logging.WARNING,
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "sqlalchemy.engine": logging.WARNING,
    "passlib": logging.ERROR,
}

_STANDARD_ATTRS = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"asctime", "message", "taskName"}


class RequestIdFilter(logging.Filter):
    """Attach the current request id to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """Render records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # Carry through anything passed via `extra=`.
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and key not in payload:
                try:
                    json.dumps(value)
                except (TypeError, ValueError):
                    value = repr(value)
                payload[key] = value

        return json.dumps(payload, default=str)


def configure_logging() -> None:
    """Install handlers on the root logger. Safe to call more than once."""
    if settings.LOG_FORMAT == "json":
        formatter: logging.Formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)-8s [%(request_id)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.LOG_LEVEL)

    for name, level in QUIET_LOGGERS.items():
        logging.getLogger(name).setLevel(level)

    # Route uvicorn through the same handler so the output is consistent.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
