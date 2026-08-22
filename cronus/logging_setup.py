"""Developer logging, kept strictly separate from user-facing output.

Everything at DEBUG/INFO goes to a rotating file under the data directory.
Only WARNING and above reaches the terminal, so the CLI stays readable.
A redaction filter is installed as a backstop against secrets in log records.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import re
from pathlib import Path

_SECRET_ENV_NAMES = (
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "GMAIL_APP_PASSWORD",
    "LIVEKIT_API_SECRET",
    "LIVEKIT_API_KEY",
)

_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key\"?\s*[:=]\s*)\S+"),
    re.compile(r"(?i)(password\"?\s*[:=]\s*)\S+"),
    re.compile(r"(?i)(authorization:\s*bearer\s+)\S+"),
)

_configured = False


class RedactionFilter(logging.Filter):
    """Scrub known secret values and key-shaped patterns from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - defensive
            return True
        redacted = self.redact(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True

    @staticmethod
    def redact(message: str) -> str:
        for name in _SECRET_ENV_NAMES:
            value = os.getenv(name)
            if value and len(value) > 6 and value in message:
                message = message.replace(value, "<redacted>")
        for pattern in _PATTERNS:
            message = pattern.sub(r"\1<redacted>", message)
        return message


def setup_logging(level: str = "INFO", log_path: Path | None = None) -> None:
    """Configure the ``cronus`` logger tree. Safe to call more than once."""
    global _configured

    logger = logging.getLogger("cronus")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    if _configured:
        return

    redaction = RedactionFilter()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_path is not None:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            file_handler.addFilter(redaction)
            logger.addHandler(file_handler)
        except OSError:
            # A missing log file must never stop the assistant from running.
            pass

    console = logging.StreamHandler()
    # Only genuinely unexpected failures reach the terminal; everything a user
    # needs to know is said by the assistant itself.
    console.setLevel(logging.ERROR)
    console.setFormatter(logging.Formatter("cronus: %(levelname)s %(message)s"))
    console.addFilter(redaction)
    logger.addHandler(console)

    # Third-party chatter belongs in our file at WARNING, not on the terminal.
    for noisy in ("httpx", "google_genai", "urllib3", "ddgs", "primp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child logger (``cronus.<name>``)."""
    return logging.getLogger(f"cronus.{name}")
