"""Memory storage, recall, and persistence."""

from __future__ import annotations

import pytest

from cronus.config import MemoryConfig
from cronus.errors import StorageError
from cronus.memory.store import MemoryStore
from cronus.storage.db import Database


def test_store_and_read_back(memory: MemoryStore):
    item = memory.remember("The user prefers concise answers.", kind="preference")
    assert item.id > 0
    assert memory.get(item.id).content == "The user prefers concise answers."
    assert memory.count() == 1


def test_content_must_be_substantial(memory: MemoryStore):
    with pytest.raises(StorageError):
        memory.remember("x")


def test_unknown_kinds_fall_back_to_fact(memory: MemoryStore):
    assert memory.remember("Something true.", kind="nonsense").kind == "fact"


def test_recall_finds_relevant_memories(memory: MemoryStore):
    memory.remember("The user's brother is called Dan and lives in Calgary.", kind="person")
    memory.remember("The user is allergic to peanuts.", kind="fact")
    memory.remember("The office wifi password is on the whiteboard.", kind="fact")

    found = memory.recall("what should I know about Dan")
    assert any("Dan" in item.content for item in found)


def test_recall_ignores_unrelated_memories(memory: MemoryStore):
    memory.remember("The user drives a blue Toyota Corolla.", kind="fact")
    assert memory.recall("quantum chromodynamics research papers") == []


def test_preferences_are_recalled_even_when_unmentioned(memory: MemoryStore):
    """Preferences shape every answer, so they surface without being named."""
    memory.remember("Keep answers short and skip the preamble.", kind="preference")
    found = memory.recall("explain how photosynthesis works")
    assert any(item.kind == "preference" for item in found)


def test_near_duplicates_are_merged_not_stacked(memory: MemoryStore):
    memory.remember("The user prefers concise answers.", kind="preference")
    memory.remember("The user prefers concise answers please.", kind="preference")
    assert memory.count() == 1


def test_update_changes_content(memory: MemoryStore):
    item = memory.remember("The user lives in Edmonton.")
    memory.update(item.id, content="The user lives in Calgary.")
    assert "Calgary" in memory.get(item.id).content


def test_forget_removes_a_memory(memory: MemoryStore):
    item = memory.remember("A fact worth dropping later.")
    assert memory.forget(item.id) is True
    assert memory.get(item.id) is None
    assert memory.forget(item.id) is False


def test_forget_by_description(memory: MemoryStore):
    memory.remember("The user's dentist appointment is on Tuesday.")
    assert memory.forget_matching("dentist appointment Tuesday") == 1
    assert memory.count() == 0


def test_recall_updates_usage_counters(memory: MemoryStore):
    item = memory.remember("The user's favourite editor is Vim.")
    memory.recall("which editor does the user like")
    assert memory.get(item.id).use_count >= 1


def test_the_store_is_capped(database: Database):
    store = MemoryStore(database, MemoryConfig(max_stored=5))
    for index in range(9):
        store.remember(f"Distinct memory number {index} about topic {index}.")
    assert store.count() <= 5


def test_memories_survive_a_reconnect(tmp_path):
    path = tmp_path / "memories.db"
    first = Database(path)
    MemoryStore(first).remember("The user's cat is called Widget.", kind="person")
    first.close()

    second = Database(path)
    try:
        assert any("Widget" in item.content for item in MemoryStore(second).all())
    finally:
        second.close()


def test_fts_special_characters_do_not_break_recall(memory: MemoryStore):
    """Model-supplied queries reach FTS5, whose syntax must not leak."""
    memory.remember("The user's project is called Cronus.")
    assert memory.recall('cronus AND "unbalanced (quote') is not None
