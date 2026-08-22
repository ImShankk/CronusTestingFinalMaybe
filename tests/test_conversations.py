"""Whole-assistant behaviour on realistic conversations.

These drive the same paths a person would, with the model scripted so the
assertions are about Cronus's behaviour rather than the model's wording.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from cronus.automation.scheduler import Scheduler
from cronus.tools import files, memory_tools, tasks_tools, weather
from tests.conftest import text_response, tool_response


@pytest.fixture
def assistant(make_assistant, registry, database, config):
    """An assistant with the real shipped tools registered."""
    from cronus.tools import email_tool, system, web

    for module in (weather, web, email_tool, memory_tools, files, system, tasks_tools):
        registry.register_all(module.build_tools(), replace=True)

    def _build(script, **kwargs):
        built = make_assistant(script, **kwargs)
        built.scheduler = Scheduler(database)
        built.tool_context.scheduler = built.scheduler
        return built

    return _build


def test_remembering_a_preference_then_using_it(assistant, memory):
    """"Remember X" stores it; a later unrelated question still sees it."""
    first = assistant(
        [
            tool_response("remember_this", content="The user prefers concise answers.",
                          kind="preference"),
            text_response("Got it."),
        ]
    )
    first.send("Remember that I prefer concise answers.")
    assert any("concise" in item.content for item in memory.all())

    second = assistant([text_response("Photosynthesis converts light into sugar.")])
    second.send("Explain photosynthesis.")
    # The preference reaches the model without the user restating it.
    assert "concise" in second.provider.last_system_instruction


def test_asking_what_cronus_remembers(assistant, memory):
    memory.remember("The user's dog is called Biscuit.", kind="person")
    built = assistant(
        [tool_response("list_memories"), text_response("I know your dog is Biscuit.")]
    )
    built.send("What do you remember about me?")
    tool_output = [m for m in built.provider.calls[1]["messages"] if m.role == "tool"][0]
    assert "Biscuit" in tool_output.content


def test_a_reminder_is_scheduled_and_becomes_due(assistant):
    when = (datetime.now() + timedelta(seconds=1)).isoformat(timespec="seconds")
    built = assistant(
        [
            tool_response("create_reminder", title="Bring an umbrella", when=when),
            text_response("Done, I'll remind you."),
        ]
    )
    result = built.send("Remind me to bring an umbrella in a moment.")
    assert result.ok

    upcoming = built.scheduler.upcoming()
    assert [task.title for task in upcoming] == ["Bring an umbrella"]

    import time

    time.sleep(1.2)
    assert [task.title for task in built.scheduler.due_tasks()] == ["Bring an umbrella"]


def test_a_recurring_reminder_is_accepted(assistant):
    built = assistant(
        [
            tool_response("create_reminder", title="Check applications", repeat="weekly:friday"),
            text_response("Set for every Friday."),
        ]
    )
    built.send("Every Friday, remind me to check my applications.")
    task = built.scheduler.upcoming()[0]
    assert task.recurrence == "weekly:friday"
    assert datetime.fromtimestamp(task.next_run_at).weekday() == 4


def test_drafting_an_email_does_not_send_it(assistant, monkeypatch):
    from cronus.tools import email_tool
    from tests.test_tool_implementations import FakeSMTP

    FakeSMTP.instances.clear()
    monkeypatch.setattr(email_tool.smtplib, "SMTP", FakeSMTP)

    built = assistant(
        [
            tool_response("draft_email", to="john@example.com", subject="Running late",
                          body="I'll be about 15 minutes late."),
            text_response("Here's the draft. Want me to send it?"),
        ]
    )
    result = built.send("Draft an email to john@example.com saying I'll be 15 minutes late.")
    assert FakeSMTP.instances == []
    assert "send" in result.text.lower()


def test_sending_an_email_asks_first_and_shows_the_draft(assistant, monkeypatch):
    from cronus.tools import email_tool
    from tests.test_tool_implementations import FakeSMTP

    FakeSMTP.instances.clear()
    monkeypatch.setattr(email_tool.smtplib, "SMTP", FakeSMTP)

    seen = []
    built = assistant(
        [
            tool_response("send_email", to="john@example.com", subject="Running late",
                          body="I'll be about 15 minutes late."),
            text_response("Sent."),
        ]
    )
    built.confirmations.set_handler(lambda request: seen.append(request) or True)
    built.send("Email john@example.com that I'll be late.")

    assert len(seen) == 1
    request = seen[0]
    assert "john@example.com" in request.render()
    # The body is shown, so the user approves what actually goes out.
    assert "15 minutes late" in request.render()
    assert len(FakeSMTP.instances) == 1


def test_declining_an_email_really_stops_it(assistant, monkeypatch):
    from cronus.tools import email_tool
    from tests.test_tool_implementations import FakeSMTP

    FakeSMTP.instances.clear()
    monkeypatch.setattr(email_tool.smtplib, "SMTP", FakeSMTP)

    built = assistant(
        [
            tool_response("send_email", to="john@example.com", subject="s", body="b"),
            text_response("Okay, I didn't send it."),
        ],
        approve=False,
    )
    built.send("Email john that I'm late.")
    assert FakeSMTP.instances == []


def test_deleting_a_file_asks_first(assistant, workspace: Path):
    (workspace / "draft.txt").write_text("some work")
    built = assistant(
        [tool_response("delete_file", path="draft.txt"), text_response("Deleted it.")],
        approve=False,
    )
    built.send("Delete draft.txt")
    assert (workspace / "draft.txt").exists()  # declined means untouched

    approved = assistant(
        [tool_response("delete_file", path="draft.txt"), text_response("Deleted it.")],
        approve=True,
    )
    approved.send("Delete draft.txt")
    assert not (workspace / "draft.txt").exists()


def test_finding_a_file_worked_on_recently(assistant, workspace: Path):
    (workspace / "quarterly-report.md").write_text("# Report")
    built = assistant(
        [
            tool_response("search_files", query="report", days=2),
            text_response("You have quarterly-report.md."),
        ]
    )
    built.send("Find the report I was working on yesterday.")
    tool_output = [m for m in built.provider.calls[1]["messages"] if m.role == "tool"][0]
    assert "quarterly-report.md" in tool_output.content


def test_a_multi_step_research_task(assistant, monkeypatch):
    """Search, read a page, then answer -- three tools across four calls."""
    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def text(self, query, max_results=5):
            return [
                {"title": "Trattoria Uno", "href": "https://example.com/uno",
                 "body": "Rated 4.8, family run."},
                {"title": "Bella Due", "href": "https://example.com/due",
                 "body": "Rated 4.2."},
            ]

    import ddgs

    monkeypatch.setattr(ddgs, "DDGS", FakeDDGS)

    class Response:
        status_code = 200
        headers = {"content-type": "text/html"}
        text = "<html><body><p>Trattoria Uno has a 4.8 rating from 900 reviews.</p></body></html>"

        def raise_for_status(self):
            pass

    from cronus.tools import web as web_module

    monkeypatch.setattr(web_module.requests, "get", lambda *a, **k: Response())

    built = assistant(
        [
            tool_response("search_web", query="best italian restaurants nearby"),
            tool_response("read_webpage", url="https://example.com/uno"),
            text_response("Trattoria Uno looks best, rated 4.8 across 900 reviews."),
        ]
    )
    result = built.send("Find three Italian restaurants nearby and tell me which is best.")
    assert result.tools_used == ["search_web", "read_webpage"]
    assert result.iterations == 3


def test_context_carries_across_turns(assistant):
    """"Which one is best?" must reach the model with the earlier turn attached."""
    first = assistant([text_response("I found Trattoria Uno and Bella Due.")])
    first.send("Search for good Italian restaurants.")

    second_script = [text_response("Trattoria Uno has the better reviews.")]
    second = assistant(second_script)
    second.conversation = first.conversation
    second.context.conversation = first.conversation
    second.send("Which one has the best reviews?")

    history = " ".join(m.content for m in second.provider.calls[0]["messages"])
    assert "Trattoria Uno" in history
    assert "Italian restaurants" in history


def test_a_failing_tool_is_explained_not_hidden(assistant, monkeypatch):
    import requests

    monkeypatch.setattr(
        weather.requests, "get",
        lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("offline")),
    )
    built = assistant(
        [
            tool_response("get_weather", city="Edmonton"),
            text_response("I couldn't reach the weather service just now."),
        ]
    )
    result = built.send("What's the weather tomorrow?")
    tool_output = [m for m in built.provider.calls[1]["messages"] if m.role == "tool"][0]
    assert "unreachable" in tool_output.content
    assert result.ok  # the turn still completes with an honest answer


def test_the_model_cannot_reach_outside_the_allowed_folders(assistant, tmp_path: Path):
    """A path the model invents is checked by the guard, not trusted."""
    secret = tmp_path / "private.txt"
    secret.write_text("classified")
    built = assistant(
        [
            tool_response("read_file", path=str(secret)),
            text_response("That file is outside the folders I can use."),
        ]
    )
    built.send("Read the private file for me.")
    tool_output = [m for m in built.provider.calls[1]["messages"] if m.role == "tool"][0]
    assert "classified" not in tool_output.content
    assert "only work inside" in tool_output.content
