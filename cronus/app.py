"""Application assembly.

One place builds the object graph, so interfaces (CLI now, a GUI or API
later) only have to ask for an :class:`Assistant` and subscribe to events.
"""

from __future__ import annotations

from dataclasses import dataclass

from .automation.scheduler import Scheduler
from .config import Config, load_config
from .core.conversation import ConversationStore
from .core.events import EventEmitter
from .core.runtime import Assistant
from .llm.base import LLMProvider
from .llm.gemini import GeminiProvider
from .logging_setup import get_logger, setup_logging
from .memory.store import MemoryStore
from .profile import UserProfile
from .security.confirmation import ConfirmationManager
from .security.paths import PathGuard
from .storage.db import Database
from .tools import email_tool, files, memory_tools, system, tasks_tools, weather, web
from .tools.registry import ToolRegistry

log = get_logger("app")

# Each module contributes its tools; adding a capability means adding one entry.
TOOL_MODULES = (weather, web, email_tool, memory_tools, files, system, tasks_tools)


@dataclass
class Cronus:
    """A fully wired assistant plus the pieces an interface may need."""

    config: Config
    assistant: Assistant
    registry: ToolRegistry
    memory: MemoryStore
    profile: UserProfile
    database: Database
    emitter: EventEmitter
    scheduler: Scheduler

    def start(self) -> None:
        """Begin firing reminders.

        Deliberately separate from :func:`build`: an interface has to attach
        its delivery handler first, or a task already due would fire into
        nothing and be marked done.
        """
        self.scheduler.start()

    def shutdown(self) -> None:
        self.scheduler.stop()
        self.registry.shutdown()
        self.database.close()
        log.info("cronus stopped")


def build(
    config: Config | None = None,
    *,
    voice_mode: bool = False,
    provider: LLMProvider | None = None,
) -> Cronus:
    """Construct the assistant and everything it depends on.

    Nothing is started here. Call :meth:`Cronus.start` once the interface has
    wired up its handlers.
    """
    config = config or load_config()
    setup_logging(config.log_level, config.log_path)
    log.info("cronus starting model=%s voice=%s", config.llm.model, voice_mode)

    database = Database(config.db_path)
    memory = MemoryStore(database, config.memory)
    profile = UserProfile(database, config)
    guard = PathGuard(config.security.file_roots, config.security.max_read_bytes)
    scheduler = Scheduler(database, timezone=config.timezone)

    emitter = EventEmitter()
    registry = ToolRegistry(default_timeout=config.tool_timeout)
    for module in TOOL_MODULES:
        registry.register_all(module.build_tools())
    log.info("registered %d tools", len(registry))

    assistant = Assistant(
        config,
        provider or GeminiProvider(config.llm),
        registry,
        memory=memory,
        profile=profile,
        scheduler=scheduler,
        paths=guard,
        emitter=emitter,
        confirmations=ConfirmationManager(
            timeout=config.security.confirmation_timeout
        ),
        voice_mode=voice_mode,
        conversation_store=ConversationStore(database),
    )
    return Cronus(
        config=config,
        assistant=assistant,
        registry=registry,
        memory=memory,
        profile=profile,
        database=database,
        emitter=emitter,
        scheduler=scheduler,
    )
