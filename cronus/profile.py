"""The user profile.

Small, explicit, and persistent: how the user wants to be addressed and the
settings that shape every answer. Nothing personal is hardcoded -- values come
from configuration or from what the user tells Cronus.
"""

from __future__ import annotations

import time
from typing import Any

from .config import Config
from .errors import StorageError
from .logging_setup import get_logger
from .storage.db import Database

log = get_logger("profile")

# Only these keys are persisted, so a stray tool call cannot invent fields.
KNOWN_KEYS = {
    "name": "what to call the user",
    "timezone": "the user's timezone",
    "response_style": "preferred answer length and tone",
    "default_city": "city assumed for weather and local questions",
    "email_signature": "how to sign emails sent on their behalf",
}


class UserProfile:
    """Key/value profile backed by SQLite, seeded from configuration."""

    def __init__(self, db: Database, config: Config | None = None) -> None:
        self.db = db
        if config is not None:
            if config.user_name:
                self.set("name", config.user_name, overwrite=False)
            if config.timezone:
                self.set("timezone", config.timezone, overwrite=False)

    def get(self, key: str, default: str | None = None) -> str | None:
        rows = self.db.query("SELECT value FROM profile WHERE key = ?", (key,))
        return rows[0]["value"] if rows else default

    def set(self, key: str, value: str, *, overwrite: bool = True) -> bool:
        key = key.strip().lower()
        if key not in KNOWN_KEYS:
            log.warning("ignoring unknown profile key %r", key)
            return False
        value = str(value).strip()[:300]
        if not value:
            return False
        try:
            if not overwrite and self.get(key) is not None:
                return False
            with self.db.write() as connection:
                connection.execute(
                    "INSERT INTO profile(key, value, updated_at) VALUES (?, ?, ?)"
                    " ON CONFLICT(key) DO UPDATE SET value = excluded.value,"
                    " updated_at = excluded.updated_at",
                    (key, value, time.time()),
                )
        except StorageError:
            log.warning("could not persist profile key %s", key)
            return False
        log.info("profile updated key=%s", key)
        return True

    def unset(self, key: str) -> bool:
        with self.db.write() as connection:
            cursor = connection.execute("DELETE FROM profile WHERE key = ?", (key,))
            return cursor.rowcount > 0

    def all(self) -> dict[str, Any]:
        return {row["key"]: row["value"] for row in self.db.query("SELECT * FROM profile")}

    def as_prompt_section(self) -> str:
        values = self.all()
        if not values:
            return ""
        lines = "\n".join(f"- {key}: {value}" for key, value in sorted(values.items()))
        return f"# User profile\n{lines}"
