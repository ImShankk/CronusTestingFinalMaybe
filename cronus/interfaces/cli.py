"""Terminal interface.

The CLI owns presentation only: it subscribes to runtime events, prints status
lines, answers confirmation requests, and optionally speaks. It holds no
assistant logic, so a GUI or an API can replace it without touching the core.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import threading
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..app import Cronus, build
from ..config import VoiceMode, load_config
from ..core.events import AssistantState, Event, EventType
from ..errors import ConfigError, CronusError
from ..logging_setup import get_logger
from ..security.confirmation import ConfirmationRequest
from ..voice.base import ListenOutcome, SpeechToText, TextToSpeech
from ..voice.session import VoiceSession, build_session

log = get_logger("cli")

BANNER = "CRONUS"
# How many hard microphone/service failures in a row before Cronus stops
# retrying and hands the user back a keyboard.
_MAX_LISTEN_FAILURES = 3

# What the user is told for each way listening can end without text.
_LISTEN_MESSAGES = {
    ListenOutcome.NO_SPEECH: ("dim", "No speech detected."),
    ListenOutcome.NOT_UNDERSTOOD: ("dim", "I didn't catch that."),
    ListenOutcome.MIC_ERROR: ("red", "Microphone unavailable."),
    ListenOutcome.SERVICE_ERROR: ("red", "Speech recognition failed."),
}

_LISTEN_HINTS = {
    ListenOutcome.MIC_ERROR: (
        "Check that a microphone is connected and that Windows lets this app "
        "use it, or set CRONUS_MIC_INDEX."
    ),
    ListenOutcome.SERVICE_ERROR: (
        "Speech recognition needs an internet connection."
    ),
}
# What Cronus is doing, in the user's terms. The tool that happens to be
# running is an implementation detail; a status line that names it turns the
# assistant back into a menu of functions. Anything unmapped just says
# "Working...", which is true of every tool.
_ACTIVITY = {
    "get_weather": "Checking the weather",
    "search_web": "Searching",
    "read_webpage": "Reading that page",
    "draft_email": "Writing that",
    "send_email": "Sending",
    "list_directory": "Looking through your files",
    "read_file": "Reading that",
    "search_files": "Looking through your files",
    "write_file": "Saving that",
    "move_file": "Moving that",
    "delete_file": "Deleting that",
    "remember_this": "Noting that down",
    "set_preference": "Noting that down",
    "recall_memories": "Checking what I know",
    "list_memories": "Checking what I know",
    "forget_memory": "Forgetting that",
    "create_reminder": "Setting that up",
    "list_reminders": "Checking your reminders",
    "cancel_reminder": "Cancelling that",
    "system_status": "Checking your machine",
    "open_url": "Opening that",
    "open_app": "Opening that",
}

# Answering a confirmation out loud. Negation is checked first, so "no, don't
# send it" can never be read as approval because it contains "send it". A
# leading affirmative counts; anything else is treated as unclear and asked
# again, which is the safe direction to fail in.
_NEGATIVE = {
    "n", "no", "nope", "nah", "not", "negative", "cancel", "stop", "don't",
    "dont", "never", "wait", "hold",
}
_AFFIRMATIVE = {
    "y", "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "affirmative",
    "correct", "confirmed", "go", "send", "do", "proceed", "please",
}
_ANSWER_TOKENS = re.compile(r"[a-z']+")
# How many times to ask by voice before falling back to the keyboard, so a
# user who has walked away is not left in an unanswerable loop.
_MAX_VOICE_CONFIRM_ATTEMPTS = 2

# Human labels for the fields shown above a pending action.
_DETAIL_LABELS = {
    "to": "To",
    "to_email": "To",
    "recipient": "To",
    "cc": "Cc",
    "subject": "Subject",
    "path": "File",
    "destination": "To",
    "url": "Link",
}


def interpret_answer(raw: str) -> bool | None:
    """Read a yes or a no out of ordinary speech. None means unclear."""
    tokens = _ANSWER_TOKENS.findall((raw or "").lower())
    if not tokens:
        return None
    if any(token in _NEGATIVE for token in tokens):
        return False
    return True if tokens[0] in _AFFIRMATIVE else None


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
            self._show(_ACTIVITY.get(event.data["tool"], "Working") + "...")
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
        self.session: VoiceSession | None = None
        self.voice_input = False
        self._listen_failures = 0

        cronus.emitter.subscribe(self.ui.handle)
        self.assistant.confirmations.set_handler(self.confirm)
        cronus.scheduler.on_due = self.announce_task

    # ------------------------------------------------------------------
    # Voice wiring
    # ------------------------------------------------------------------
    def enable_voice(self, *, listen: bool) -> None:
        config = self.cronus.config.voice
        self.session = build_session(
            config,
            emitter=self.cronus.emitter,
            prompt_to_talk=self._prompt_to_talk,
            listen=listen,
        )
        self.tts = self.session.tts
        self.stt = self.session.stt

        if self.tts is None:
            self.console.print(
                "[yellow]No speech output available. Set CRONUS_PIPER_EXE and "
                "CRONUS_PIPER_MODEL, or use the Windows voice.[/yellow]"
            )
        else:
            self.speak_replies = True

        if listen:
            if self.stt is None:
                self.console.print(
                    "[red]Microphone unavailable.[/red] "
                    "[dim]Staying on typed input. "
                    f"{_LISTEN_HINTS[ListenOutcome.MIC_ERROR]}[/dim]"
                )
            else:
                self.voice_input = True
                self.assistant.voice_mode = True

    def _prompt_to_talk(self) -> bool:
        """Push-to-talk only: wait for Enter. False means the user quit."""
        try:
            answer = self.console.input(
                "[dim]Press Enter to speak (or type 'quit'):[/dim] "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer not in ("quit", "exit", "q")

    def _report_listen_failure(self) -> None:
        """Say why nothing was heard, and give up on the mic if it keeps failing.

        Silence is a dim one-liner; a broken microphone or unreachable
        recogniser is called out in red with something actionable, because
        those will never fix themselves by waiting.
        """
        stt = self.session.stt if self.session is not None else self.stt
        outcome = getattr(stt, "last_outcome", ListenOutcome.NO_SPEECH)

        # A quiet room in continuous mode is not an event. Announcing it every
        # time the listen window expires fills the terminal with "No speech
        # detected." while the user is simply not talking yet.
        quiet_idle = (
            outcome is ListenOutcome.NO_SPEECH
            and self.session is not None
            and self.session.mode is VoiceMode.CONTINUOUS
        )
        if not quiet_idle:
            style, message = _LISTEN_MESSAGES.get(
                outcome, ("dim", "No speech detected.")
            )
            self.console.print(f"[{style}]{message}[/{style}]")

        if not outcome.is_failure:
            self._listen_failures = 0
            return

        hint = _LISTEN_HINTS.get(outcome)
        if hint:
            self.console.print(f"[dim]{hint}[/dim]")

        detail = getattr(stt, "last_error", None)
        if detail and log.isEnabledFor(logging.DEBUG):
            self.console.print(f"[dim]{detail}[/dim]")

        self._listen_failures += 1
        if self._listen_failures >= _MAX_LISTEN_FAILURES:
            # Repeating the same failure forever helps nobody.
            self.voice_input = False
            self.assistant.voice_mode = False
            self.console.print(
                f"[yellow]Giving up on the microphone after "
                f"{self._listen_failures} failures. Switching to typed input; "
                f"details are in {self.cronus.config.log_path}.[/yellow]"
            )

    def announce_task(self, task: Any) -> None:
        """Deliver a reminder that just came due, without wrecking the prompt."""
        self.ui.stop_status()
        self.console.print()
        self.console.print(Panel(task.title, title="Reminder", border_style="yellow"))
        if self.speak_replies and self.session is not None:
            self.session.speak(f"Reminder: {task.title}")

    # ------------------------------------------------------------------
    # Confirmation
    # ------------------------------------------------------------------
    def confirm(self, request: ConfirmationRequest) -> bool:
        """Show what is about to happen, in the shape a person reads it in.

        The wording is friendlier than it was; what is shown is not. Every
        real argument still appears, because the point of the panel is that
        you approve the thing that will actually be done.
        """
        self.ui.stop_status()
        self.console.print(
            Panel(self._render_request(request), title="Confirm", border_style="yellow")
        )

        if self.speak_replies and self.session is not None:
            self.session.speak(request.summary)

        voice_attempts = 0
        while True:
            try:
                answer, by_voice = self._ask_yes_no(voice_attempts)
            except (EOFError, KeyboardInterrupt):
                self.console.print("[dim]Cancelled.[/dim]")
                return False
            if answer is not None:
                return answer
            if by_voice:
                voice_attempts += 1
            self.console.print("[dim]Sorry -- yes or no?[/dim]")

    @staticmethod
    def _render_request(request: ConfirmationRequest) -> Text:
        """Headline, then the fields, then the body set apart as prose."""
        body = Text(request.summary, style="bold")
        details = dict(request.details)
        message = details.pop("body", None)
        for label, value in details.items():
            body.append(f"\n{_DETAIL_LABELS.get(label, label.replace('_', ' '))}: ",
                        style="dim")
            body.append(value)
        if message:
            body.append("\n\n")
            body.append(message)
        return body

    def _ask_yes_no(self, voice_attempts: int = 0) -> tuple[bool | None, bool]:
        """Read an answer. Returns (yes/no/unclear, whether voice was used)."""
        if (
            self.voice_input
            and self.stt is not None
            and voice_attempts < _MAX_VOICE_CONFIRM_ATTEMPTS
        ):
            self.console.print("[dim]Yes or no?[/dim]")
            heard = self.stt.listen(timeout=8.0)
            if heard.strip():
                self.console.print(f"[dim]heard: {heard}[/dim]")
            # Silence is unclear, not consent, and costs one of the attempts
            # before the keyboard takes over.
            return interpret_answer(heard), True
        raw = self.console.input("[yellow]Go ahead? [y/n][/yellow] ")
        return interpret_answer(raw), False

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

            self.respond(user_text)

    def _next_input(self) -> str | None:
        """Get the next message, by voice when enabled, otherwise typed."""
        if not self.voice_input or self.session is None:
            raw = self.console.input("[bold cyan]you[/bold cyan] ").strip()
            return raw

        # Nothing is announced for continuous mode: the spinner already says
        # "Listening...", and printing it as well leaves a line behind on
        # every turn. A pending utterance -- what the user said to interrupt
        # -- is already captured, so there is nothing to wait for either.
        if self.session.mode is VoiceMode.WAKE_WORD and not self.session.has_pending:
            self.console.print(
                f"[dim]Say \"{self.cronus.config.voice.wake_word}\" "
                "(Ctrl+C to quit)[/dim]"
            )

        utterance = self.session.next_utterance()
        self.ui.stop_status()

        if utterance.quit_requested:
            return None
        if not utterance.heard:
            self._report_listen_failure()
            return ""

        self._listen_failures = 0
        self.console.print(f"[bold cyan]you[/bold cyan] {utterance.text}")
        return utterance.text

    def respond(self, user_text: str) -> None:
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
        # Which tools ran is developer information, not part of the
        # conversation. It stays one CRONUS_LOG_LEVEL=DEBUG away.
        if result.tools_used and log.isEnabledFor(logging.DEBUG):
            self.console.print(f"[dim]used: {', '.join(result.tools_used)}[/dim]")

        if self.speak_replies and self.session is not None:
            try:
                # An interruption needs no announcement: the user heard the
                # speech stop, and what they said next is already on its way
                # back as the following turn.
                self.session.speak(result.text)
            except KeyboardInterrupt:
                self.session.stop_speaking()
            finally:
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
        tasks = self.cronus.scheduler.upcoming()
        if not tasks:
            self.console.print("[dim]Nothing scheduled.[/dim]")
            return
        for task in tasks:
            self.console.print(task.summary())

    def _toggle_voice(self) -> None:
        if self.session is None:
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
        if self.session is not None and (self.voice_input or self.speak_replies):
            if self.voice_input:
                details.append(f"voice: {self.session.describe_mode()}")
            details.append(self.session.describe_voice())
            if self.voice_input and self.session.barge_in_available:
                details.append("interruptible")
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

    # CRONUS_VOICE turns on both halves of voice; the flags add to it, so
    # --speak alone still means "type, but talk back". Worked out once here
    # and passed down, so there is a single source of truth.
    use_voice = args.voice or config.voice.enabled
    use_speak = args.speak or use_voice

    if sys.platform == "win32":
        # The console defaults to a legacy code page that mangles accents.
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):  # pragma: no cover - old terminals
            pass

    try:
        cronus = build(config, voice_mode=use_voice)
    except CronusError as exc:
        console.print(f"[red]{exc.user_message}[/red]")
        return 2

    cli = CronusCLI(cronus)
    try:
        if use_voice or use_speak:
            cli.enable_voice(listen=use_voice)

        if args.message:
            cli.respond(args.message)
            return 0

        # Reminders start firing only now that the handler above is attached.
        if not args.no_scheduler:
            cronus.start()
        return cli.run()
    finally:
        if cli.session is not None:
            cli.session.close()
        cronus.shutdown()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
