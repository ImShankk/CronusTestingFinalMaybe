"""Terminal interface.

The CLI owns presentation only: it subscribes to runtime events, prints status
lines, answers confirmation requests, and optionally speaks. It holds no
assistant logic, so a GUI or an API can replace it without touching the core.
"""

from __future__ import annotations

import argparse
import sys
import threading
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..app import Cronus, build
from ..config import load_config
from ..core.events import AssistantState, Event, EventType
from ..errors import ConfigError, CronusError
from ..logging_setup import get_logger
from ..security.confirmation import ConfirmationRequest
from ..voice.base import SpeechToText, TextToSpeech

log = get_logger("cli")

BANNER = "CRONUS"
_YES = {"y", "yes", "yeah", "yep", "sure", "ok", "okay", "go ahead", "send it", "do it"}
_NO = {"n", "no", "nope", "cancel", "stop", "don't", "dont"}

HELP_TEXT = """\
Commands
  /help              show this
  /tools             list tools and their permission level
  /memory            show what Cronus remembers
  /forget <id>       delete one memory
  /profile           show the stored user profile
  /tasks             list scheduled reminders
  /voice             toggle speaking replies aloud
  /clear             start a fresh conversation
  /quit              exit

Anything else is a message to Cronus.
"""


class ConsoleUI:
    """Renders runtime events. One transient status line, never a log dump."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self._status: Any = None
        self._lock = threading.Lock()

    def handle(self, event: Event) -> None:
        if event.type is EventType.STATE:
            self._on_state(event.data.get("state"))
        elif event.type is EventType.TOOL_START:
            self._show(f"Using {event.data['tool']}...")
        elif event.type is EventType.PROGRESS and event.message:
            self._show(event.message)
        elif event.type is EventType.ERROR:
            self.stop_status()
            self.console.print(f"[red]{event.message}[/red]")

    def _on_state(self, state: AssistantState | None) -> None:
        labels = {
            AssistantState.THINKING: "Thinking...",
            AssistantState.EXECUTING: "Working...",
            AssistantState.LISTENING: "Listening...",
            AssistantState.SPEAKING: "Speaking...",
        }
        if state in labels:
            self._show(labels[state])
        else:
            self.stop_status()

    def _show(self, message: str) -> None:
        with self._lock:
            if self._status is None:
                self._status = self.console.status(f"[dim]{message}[/dim]")
                self._status.start()
            else:
                self._status.update(f"[dim]{message}[/dim]")

    def stop_status(self) -> None:
        with self._lock:
            if self._status is not None:
                self._status.stop()
                self._status = None


class CronusCLI:
    """The interactive session."""

    def __init__(self, cronus: Cronus, *, speak_replies: bool = False) -> None:
        self.cronus = cronus
        self.assistant = cronus.assistant
        self.console = Console()
        self.ui = ConsoleUI(self.console)
        self.speak_replies = speak_replies
        self.stt: SpeechToText | None = None
        self.tts: TextToSpeech | None = None
        self.wake: Any = None
        self.voice_input = False

        cronus.emitter.subscribe(self.ui.handle)
        self.assistant.confirmations.set_handler(self.confirm)
        if cronus.scheduler is not None:
            cronus.scheduler.on_due = self.announce_task

    # ------------------------------------------------------------------
    # Voice wiring
    # ------------------------------------------------------------------
    def enable_voice(self, *, listen: bool) -> None:
        from ..voice.stt import build_stt
        from ..voice.tts import build_tts
        from ..voice.wake import build_wake_word

        config = self.cronus.config.voice
        self.tts = build_tts(config)
        if self.tts is None:
            self.console.print(
                "[yellow]No speech output available. Set CRONUS_PIPER_EXE and "
                "CRONUS_PIPER_MODEL, or use the Windows voice.[/yellow]"
            )
        else:
            self.speak_replies = True

        if listen:
            self.stt = build_stt(config)
            if self.stt is None:
                self.console.print(
                    "[yellow]No microphone available, so I'll stay on typed input.[/yellow]"
                )
            else:
                self.voice_input = True
                self.assistant.voice_mode = True
                self.wake = build_wake_word(self.stt, config)

    def announce_task(self, task: Any) -> None:
        """Deliver a reminder that just came due, without wrecking the prompt."""
        self.ui.stop_status()
        self.console.print()
        self.console.print(Panel(task.title, title="Reminder", border_style="yellow"))
        if self.speak_replies and self.tts is not None:
            self.tts.speak(f"Reminder: {task.title}")

    # ------------------------------------------------------------------
    # Confirmation
    # ------------------------------------------------------------------
    def confirm(self, request: ConfirmationRequest) -> bool:
        self.ui.stop_status()
        body = Text(request.summary, style="bold")
        for label, value in request.details.items():
            body.append(f"\n{label}: ", style="dim")
            body.append(value)
        self.console.print(Panel(body, title="Confirm", border_style="yellow"))

        if self.speak_replies and self.tts is not None:
            self.tts.speak(request.summary)

        while True:
            try:
                answer = self._ask_yes_no()
            except (EOFError, KeyboardInterrupt):
                self.console.print("[dim]Cancelled.[/dim]")
                return False
            if answer is not None:
                return answer
            self.console.print("[dim]Please answer yes or no.[/dim]")

    def _ask_yes_no(self) -> bool | None:
        if self.voice_input and self.stt is not None:
            self.console.print("[dim]Say yes or no (or type it).[/dim]")
            heard = self.stt.listen(timeout=8.0).lower().strip(" .!")
            if heard:
                self.console.print(f"[dim]heard: {heard}[/dim]")
                if heard in _YES:
                    return True
                if heard in _NO:
                    return False
                return None
        raw = self.console.input("[yellow]Go ahead? [y/n][/yellow] ").strip().lower()
        if raw in _YES:
            return True
        if raw in _NO:
            return False
        return None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> int:
        self._print_header()
        while True:
            try:
                user_text = self._next_input()
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[dim]Bye.[/dim]")
                return 0

            if user_text is None:
                return 0
            if not user_text:
                continue
            if user_text.startswith("/"):
                if self._command(user_text):
                    return 0
                continue

            self._respond(user_text)

    def _next_input(self) -> str | None:
        """Get the next message, by voice when enabled, otherwise typed."""
        if not self.voice_input or self.stt is None:
            raw = self.console.input("[bold cyan]you[/bold cyan] ").strip()
            return raw

        if self.wake is not None and self.wake.name == "keyword":
            self.console.print(
                f"[dim]Say \"{self.cronus.config.voice.wake_word}\" (Ctrl+C to quit)[/dim]"
            )
            if not self.wake.wait_for_wake():
                return None
            carried = self.wake.carried_text
            if carried:
                self.console.print(f"[bold cyan]you[/bold cyan] {carried}")
                return carried
        elif self.wake is not None:
            if not self.wake.wait_for_wake():
                return None

        self.cronus.emitter.set_state(AssistantState.LISTENING)
        heard = self.stt.listen()
        self.cronus.emitter.set_state(AssistantState.IDLE)
        self.ui.stop_status()
        if not heard:
            self.console.print("[dim]I didn't catch that.[/dim]")
            return ""
        self.console.print(f"[bold cyan]you[/bold cyan] {heard}")
        return heard

    def _respond(self, user_text: str) -> None:
        try:
            result = self.assistant.send(user_text)
        except KeyboardInterrupt:
            self.assistant.cancel()
            self.ui.stop_status()
            self.console.print("[dim]Stopped.[/dim]")
            return
        finally:
            self.ui.stop_status()

        if result.cancelled or not result.text:
            return

        self.console.print(f"[bold green]cronus[/bold green] {result.text}")
        if result.tools_used:
            self.console.print(f"[dim]used: {', '.join(result.tools_used)}[/dim]")

        if self.speak_replies and self.tts is not None:
            self.cronus.emitter.set_state(AssistantState.SPEAKING)
            try:
                self.tts.speak(result.text)
            except KeyboardInterrupt:
                self.tts.stop()
            finally:
                self.cronus.emitter.set_state(AssistantState.IDLE)
                self.ui.stop_status()

    # ------------------------------------------------------------------
    # Slash commands
    # ------------------------------------------------------------------
    def _command(self, raw: str) -> bool:
        """Handle a slash command. Returns True when the session should end."""
        command, _, argument = raw[1:].strip().partition(" ")
        command = command.lower()
        argument = argument.strip()

        if command in ("quit", "exit", "q"):
            self.console.print("[dim]Bye.[/dim]")
            return True
        if command in ("help", "?"):
            self.console.print(HELP_TEXT)
        elif command == "tools":
            self._show_tools()
        elif command == "memory":
            self._show_memories()
        elif command == "forget":
            self._forget(argument)
        elif command == "profile":
            profile = self.cronus.profile.all()
            self.console.print(profile or "[dim]Nothing stored yet.[/dim]")
        elif command == "tasks":
            self._show_tasks()
        elif command == "voice":
            self._toggle_voice()
        elif command == "clear":
            self.assistant.reset_conversation()
            self.console.print("[dim]Conversation cleared.[/dim]")
        else:
            self.console.print(f"[dim]Unknown command {command!r}. Try /help.[/dim]")
        return False

    def _show_tools(self) -> None:
        table = Table(box=None, pad_edge=False)
        table.add_column("tool", style="cyan")
        table.add_column("permission")
        table.add_column("what it does", overflow="fold")
        policy = self.assistant.permissions
        for tool in self.cronus.registry:
            decision = policy.check(tool).decision.value
            colour = {"allow": "green", "confirm": "yellow", "deny": "red"}[decision]
            table.add_row(
                tool.name,
                f"[{colour}]{decision}[/{colour}]",
                tool.description.split(".")[0] + ".",
            )
        self.console.print(table)

    def _show_memories(self) -> None:
        items = self.cronus.memory.all()
        if not items:
            self.console.print("[dim]I haven't stored anything yet.[/dim]")
            return
        for item in items:
            self.console.print(f"[dim]{item.id:>3}[/dim] ({item.kind}) {item.content}")

    def _forget(self, argument: str) -> None:
        if not argument.isdigit():
            self.console.print("[dim]Usage: /forget <id>[/dim]")
            return
        if self.cronus.memory.forget(int(argument)):
            self.console.print(f"[dim]Forgot memory {argument}.[/dim]")
        else:
            self.console.print(f"[dim]No memory numbered {argument}.[/dim]")

    def _show_tasks(self) -> None:
        if self.cronus.scheduler is None:
            self.console.print("[dim]Scheduling is off.[/dim]")
            return
        tasks = self.cronus.scheduler.upcoming()
        if not tasks:
            self.console.print("[dim]Nothing scheduled.[/dim]")
            return
        for task in tasks:
            self.console.print(task.summary())

    def _toggle_voice(self) -> None:
        if self.tts is None:
            self.enable_voice(listen=False)
            return
        self.speak_replies = not self.speak_replies
        self.console.print(
            f"[dim]Speaking replies {'on' if self.speak_replies else 'off'}.[/dim]"
        )

    def _print_header(self) -> None:
        self.console.print(f"[bold]{BANNER}[/bold]", highlight=False)
        details = [
            self.cronus.config.llm.model,
            f"{len(self.cronus.registry)} tools",
        ]
        if self.voice_input:
            details.append("voice input")
        if self.speak_replies:
            details.append("speaking")
        self.console.print(f"[dim]{' · '.join(details)} · /help[/dim]\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cronus", description="Personal AI assistant")
    parser.add_argument("--voice", action="store_true", help="listen and speak")
    parser.add_argument("--speak", action="store_true", help="speak replies, type input")
    parser.add_argument("--message", "-m", help="run one message and exit")
    parser.add_argument("--no-scheduler", action="store_true", help="don't run reminders")
    args = parser.parse_args(argv)

    console = Console()
    try:
        config = load_config()
    except ConfigError as exc:
        console.print(f"[red]{exc.user_message}[/red]")
        return 2

    if sys.platform == "win32":
        # The console defaults to a legacy code page that mangles accents.
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):  # pragma: no cover - old terminals
            pass

    try:
        cronus = build(
            config,
            voice_mode=args.voice,
            start_scheduler=not args.no_scheduler and not args.message,
        )
    except CronusError as exc:
        console.print(f"[red]{exc.user_message}[/red]")
        return 2

    cli = CronusCLI(cronus)
    try:
        if args.voice or args.speak:
            cli.enable_voice(listen=args.voice)

        if args.message:
            cli._respond(args.message)
            return 0
        return cli.run()
    finally:
        if cli.tts is not None:
            cli.tts.close()
        cronus.shutdown()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
