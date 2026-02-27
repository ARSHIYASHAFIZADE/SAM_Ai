"""
Shared server utilities: structured logging, error responses, auth decorator.
"""

import logging
import json
import functools
from flask import request, session, jsonify

# ── Structured JSON logger ────────────────────────────────────────────────────
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "path": getattr(record, "path", request.path if request else ""),
        }
        if record.exc_info:
            log_obj["exc"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def configure_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)


# ── Generic safe error response (Fix 4) ──────────────────────────────────────
def error_response(message: str, status: int, exc: Exception = None):
    """Return a generic client message; log the real exception server-side."""
    if exc is not None:
        logging.getLogger("app.errors").error(
            "%s — %s", message, exc, exc_info=exc
        )
    return jsonify({"error": message}), status


# ── Auth decorator (Fix 3) ────────────────────────────────────────────────────
def require_auth(fn):
    """Decorator: returns 401 if no active server session."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"error": "Authentication required"}), 401
        return fn(*args, **kwargs)
    return wrapper
