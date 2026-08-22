"""Filesystem containment.

Model-generated paths are untrusted. Every path a file tool touches is
resolved (following symlinks) and checked against an allowlist of roots, so
``../..``, symlink escapes, and absolute paths to sensitive locations all fail
the same way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from ..errors import PathNotAllowed
from ..logging_setup import get_logger

log = get_logger("security.paths")

# Extensions Cronus will read as text. Anything else is reported, not opened.
TEXT_SUFFIXES = frozenset(
    {
        ".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json", ".yaml",
        ".yml", ".toml", ".ini", ".cfg", ".conf", ".log", ".py", ".js", ".ts",
        ".tsx", ".jsx", ".html", ".css", ".xml", ".sql", ".sh", ".bat", ".ps1",
        ".env.example", ".gitignore",
    }
)


class PathGuard:
    """Resolves and validates paths against allowlisted roots."""

    def __init__(self, roots: Sequence[Path], max_read_bytes: int = 200_000) -> None:
        self.roots: tuple[Path, ...] = tuple(
            dict.fromkeys(Path(root).expanduser().resolve(strict=False) for root in roots)
        )
        self.max_read_bytes = max_read_bytes

    @property
    def configured(self) -> bool:
        return bool(self.roots)

    def describe_roots(self) -> str:
        if not self.roots:
            return "no folders (file access is disabled)"
        return ", ".join(str(root) for root in self.roots)

    def resolve(self, raw_path: str, *, must_exist: bool = False) -> Path:
        """Return a safe absolute path, or raise :class:`PathNotAllowed`."""
        if not self.roots:
            raise PathNotAllowed(
                "no file roots configured",
                user_message=(
                    "File access is switched off. Set CRONUS_FILE_ROOTS to the "
                    "folders I may use."
                ),
            )
        if not raw_path or not str(raw_path).strip():
            raise PathNotAllowed("empty path", user_message="I need a file path.")

        candidate = Path(str(raw_path).strip()).expanduser()
        if not candidate.is_absolute():
            candidate = self.roots[0] / candidate

        try:
            resolved = candidate.resolve(strict=False)
        except OSError as exc:
            raise PathNotAllowed(f"cannot resolve {raw_path!r}: {exc}") from exc

        if not self._within_roots(resolved):
            log.warning("blocked path outside roots: %s", resolved)
            raise PathNotAllowed(
                f"{resolved} is outside the allowed roots",
                user_message=(
                    f"I can only work inside {self.describe_roots()}."
                ),
            )

        if must_exist and not resolved.exists():
            raise PathNotAllowed(
                f"{resolved} does not exist",
                user_message=f"I couldn't find {candidate.name}.",
            )
        return resolved

    def _within_roots(self, resolved: Path) -> bool:
        for root in self.roots:
            try:
                resolved.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def is_text_file(self, path: Path) -> bool:
        return path.suffix.lower() in TEXT_SUFFIXES

    def iter_roots(self) -> Iterable[Path]:
        return iter(self.roots)
