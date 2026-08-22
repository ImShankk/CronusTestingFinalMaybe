"""Long-term memory.

Memory is deliberately selective: nothing is stored unless the model
explicitly proposes it through the ``remember_this`` tool and this layer
accepts it. Conversations are never dumped in wholesale.

Retrieval uses SQLite's full-text index with a keyword-overlap fallback, so
recall works offline and needs no embedding model.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Iterable

from ..config import MemoryConfig
from ..errors import StorageError
from ..logging_setup import get_logger
from ..storage.db import Database

log = get_logger("memory")

VALID_KINDS = ("preference", "fact", "person", "place", "project", "task")

_STOPWORDS = frozenset(
    """a about all am an and any are as at be been but by can could did do does
    for from get got had has have he her him his how i if in into is it its me
    my no not of on or our out so some than that the their them then there
    these they this to too us was we were what when where which who why will
    with would you your""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_MIN_CONTENT = 3
_MAX_CONTENT = 500


@dataclass
class MemoryItem:
    id: int
    kind: str
    content: str
    tags: str = ""
    importance: int = 1
    source: str = "user"
    created_at: float = 0.0
    updated_at: float = 0.0
    use_count: int = 0

    def summary(self) -> str:
        return f"[{self.id}] ({self.kind}) {self.content}"


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) > 2 and token not in _STOPWORDS
    }


class MemoryStore:
    """CRUD plus relevance-ranked recall over the ``memories`` table."""

    def __init__(self, db: Database, config: MemoryConfig | None = None) -> None:
        self.db = db
        self.config = config or MemoryConfig()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def remember(
        self,
        content: str,
        *,
        kind: str = "fact",
        tags: Iterable[str] = (),
        importance: int = 1,
        source: str = "user",
    ) -> MemoryItem:
        """Store a memory, merging it into a near-duplicate if one exists."""
        content = (content or "").strip()
        if len(content) < _MIN_CONTENT:
            raise StorageError(
                "memory content too short",
                user_message="That's too short for me to store usefully.",
            )
        content = content[:_MAX_CONTENT]
        kind = kind if kind in VALID_KINDS else "fact"
        tag_text = " ".join(sorted({t.strip().lower() for t in tags if t.strip()}))
        importance = max(1, min(int(importance), 5))

        existing = self._find_duplicate(content, kind)
        if existing is not None:
            return self.update(existing.id, content=content, tags=tag_text or None)

        now = time.time()
        with self.db.write() as connection:
            cursor = connection.execute(
                "INSERT INTO memories(kind, content, tags, importance, source,"
                " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (kind, content, tag_text, importance, source, now, now),
            )
            memory_id = int(cursor.lastrowid)
        log.info("memory stored id=%s kind=%s", memory_id, kind)
        self._enforce_cap()
        return MemoryItem(
            id=memory_id,
            kind=kind,
            content=content,
            tags=tag_text,
            importance=importance,
            source=source,
            created_at=now,
            updated_at=now,
        )

    def update(
        self, memory_id: int, *, content: str | None = None, tags: str | None = None
    ) -> MemoryItem:
        item = self.get(memory_id)
        if item is None:
            raise StorageError(
                f"memory {memory_id} not found",
                user_message="I don't have a memory with that number.",
            )
        new_content = (content or item.content).strip()[:_MAX_CONTENT]
        new_tags = item.tags if tags is None else tags
        with self.db.write() as connection:
            connection.execute(
                "UPDATE memories SET content = ?, tags = ?, updated_at = ? WHERE id = ?",
                (new_content, new_tags, time.time(), memory_id),
            )
        log.info("memory updated id=%s", memory_id)
        item.content, item.tags = new_content, new_tags
        return item

    def forget(self, memory_id: int) -> bool:
        with self.db.write() as connection:
            cursor = connection.execute(
                "DELETE FROM memories WHERE id = ?", (memory_id,)
            )
            deleted = cursor.rowcount > 0
        if deleted:
            log.info("memory forgotten id=%s", memory_id)
        return deleted

    def forget_matching(self, text: str) -> int:
        """Delete memories that clearly match a description."""
        matches = self.recall(text, limit=5, min_score=0.4)
        for item in matches:
            self.forget(item.id)
        return len(matches)

    def clear(self) -> int:
        with self.db.write() as connection:
            cursor = connection.execute("DELETE FROM memories")
            return cursor.rowcount

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def get(self, memory_id: int) -> MemoryItem | None:
        rows = self.db.query("SELECT * FROM memories WHERE id = ?", (memory_id,))
        return _to_item(rows[0]) if rows else None

    def all(self, limit: int = 100) -> list[MemoryItem]:
        rows = self.db.query(
            "SELECT * FROM memories ORDER BY importance DESC, updated_at DESC LIMIT ?",
            (limit,),
        )
        return [_to_item(row) for row in rows]

    def count(self) -> int:
        rows = self.db.query("SELECT COUNT(*) AS n FROM memories")
        return int(rows[0]["n"]) if rows else 0

    def recall(
        self, query: str, *, limit: int | None = None, min_score: float | None = None
    ) -> list[MemoryItem]:
        """Return memories relevant to ``query``, most relevant first.

        Preferences are always considered -- they are what the user expects to
        apply even when they don't restate them.
        """
        limit = limit or self.config.max_recall
        threshold = self.config.min_relevance if min_score is None else min_score
        query_tokens = _tokens(query)

        candidates: dict[int, MemoryItem] = {}
        for item in self._fts_candidates(query_tokens):
            candidates[item.id] = item
        for item in self._preferences():
            candidates.setdefault(item.id, item)

        scored: list[tuple[float, MemoryItem]] = []
        for item in candidates.values():
            score = self._score(item, query_tokens)
            if score >= threshold:
                scored.append((score, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        chosen = [item for _, item in scored[:limit]]
        if chosen:
            self._mark_used([item.id for item in chosen])
            log.debug("recalled %d memories for query", len(chosen))
        return chosen

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _fts_candidates(self, tokens: set[str]) -> list[MemoryItem]:
        if not tokens:
            return []
        # Quote each token: FTS5 treats bare punctuation as query syntax.
        expression = " OR ".join(f'"{token}"' for token in tokens)
        try:
            rows = self.db.query(
                "SELECT m.* FROM memories_fts f JOIN memories m ON m.id = f.rowid"
                " WHERE memories_fts MATCH ? ORDER BY rank LIMIT 40",
                (expression,),
            )
            return [_to_item(row) for row in rows]
        except StorageError:
            # A corrupt or unavailable index must not break recall.
            log.warning("full-text search unavailable; scanning instead")
            return self.all(limit=200)

    def _preferences(self) -> list[MemoryItem]:
        rows = self.db.query(
            "SELECT * FROM memories WHERE kind = 'preference'"
            " ORDER BY importance DESC, updated_at DESC LIMIT 10"
        )
        return [_to_item(row) for row in rows]

    def _score(self, item: MemoryItem, query_tokens: set[str]) -> float:
        item_tokens = _tokens(f"{item.content} {item.tags}")
        if not item_tokens:
            return 0.0
        overlap = len(item_tokens & query_tokens)
        score = overlap / max(len(query_tokens), 1) if query_tokens else 0.0
        if item.kind == "preference":
            # Preferences shape every answer, so they get a standing floor.
            score = max(score, 0.5)
        score += 0.05 * (item.importance - 1)
        return min(score, 1.0)

    def _find_duplicate(self, content: str, kind: str) -> MemoryItem | None:
        tokens = _tokens(content)
        if not tokens:
            return None
        for item in self.db.query(
            "SELECT * FROM memories WHERE kind = ? ORDER BY updated_at DESC LIMIT 50",
            (kind,),
        ):
            candidate = _to_item(item)
            other = _tokens(candidate.content)
            if not other:
                continue
            overlap = len(tokens & other) / len(tokens | other)
            if overlap >= 0.7:
                return candidate
        return None

    def _mark_used(self, ids: list[int]) -> None:
        placeholders = ",".join("?" for _ in ids)
        try:
            with self.db.write() as connection:
                connection.execute(
                    f"UPDATE memories SET use_count = use_count + 1, last_used_at = ?"
                    f" WHERE id IN ({placeholders})",
                    (time.time(), *ids),
                )
        except StorageError:  # pragma: no cover - bookkeeping only
            log.debug("could not update memory usage counters")

    def _enforce_cap(self) -> None:
        """Keep the store bounded by dropping the least useful memories."""
        total = self.count()
        excess = total - self.config.max_stored
        if excess <= 0:
            return
        with self.db.write() as connection:
            connection.execute(
                "DELETE FROM memories WHERE id IN ("
                " SELECT id FROM memories WHERE kind != 'preference'"
                " ORDER BY importance ASC, use_count ASC, updated_at ASC LIMIT ?)",
                (excess,),
            )
        log.info("pruned %d memories to stay within the cap", excess)


def _to_item(row) -> MemoryItem:
    return MemoryItem(
        id=int(row["id"]),
        kind=row["kind"],
        content=row["content"],
        tags=row["tags"],
        importance=int(row["importance"]),
        source=row["source"],
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        use_count=int(row["use_count"]),
    )
