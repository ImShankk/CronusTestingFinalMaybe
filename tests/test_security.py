"""Permissions, confirmation, and filesystem containment."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from cronus.config import SecurityConfig
from cronus.errors import PathNotAllowed
from cronus.security.confirmation import (
    ConfirmationManager,
    ConfirmationStatus,
    always_decline,
)
from cronus.security.paths import PathGuard
from cronus.security.permissions import Decision, PermissionPolicy
from cronus.tools.base import RiskLevel, Tool, object_schema


def make_tool(name: str, risk: RiskLevel) -> Tool:
    return Tool(
        name=name,
        description="Test tool.",
        parameters=object_schema({}),
        handler=lambda: "ok",
        risk=risk,
    )


# ----------------------------------------------------------------------
# Permission policy
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "risk,expected",
    [
        (RiskLevel.SAFE, Decision.ALLOW),
        (RiskLevel.LOW, Decision.ALLOW),
        (RiskLevel.CONFIRM, Decision.CONFIRM),
        (RiskLevel.HIGH, Decision.DENY),
        (RiskLevel.BLOCKED, Decision.DENY),
    ],
)
def test_risk_maps_to_decision(risk, expected):
    policy = PermissionPolicy(SecurityConfig())
    assert policy.check(make_tool("t", risk)).decision is expected


def test_blocked_tools_cannot_be_unblocked_by_configuration():
    policy = PermissionPolicy(SecurityConfig(permission_overrides={"t": "allow"}))
    assert policy.check(make_tool("t", RiskLevel.BLOCKED)).decision is Decision.DENY


def test_high_risk_can_be_opted_into():
    policy = PermissionPolicy(SecurityConfig(permission_overrides={"t": "confirm"}))
    assert policy.check(make_tool("t", RiskLevel.HIGH)).decision is Decision.CONFIRM


def test_safe_tools_can_be_tightened():
    policy = PermissionPolicy(SecurityConfig(permission_overrides={"t": "block"}))
    assert policy.check(make_tool("t", RiskLevel.SAFE)).decision is Decision.DENY


def test_unparseable_overrides_are_ignored_not_obeyed():
    policy = PermissionPolicy(SecurityConfig(permission_overrides={"t": "maybe"}))
    assert policy.check(make_tool("t", RiskLevel.CONFIRM)).decision is Decision.CONFIRM


# ----------------------------------------------------------------------
# Confirmation
# ----------------------------------------------------------------------
def test_approval_and_decline_are_recorded():
    manager = ConfirmationManager(handler=lambda request: True)
    request = manager.request("send_email", "Send it?", {"to": "a@b.com"})
    assert manager.resolve(request) is True
    assert request.status is ConfirmationStatus.APPROVED

    manager.handler = lambda request: False
    second = manager.request("delete_file", "Delete it?", {})
    assert manager.resolve(second) is False
    assert second.status is ConfirmationStatus.DECLINED


def test_missing_handler_declines():
    manager = ConfirmationManager()
    assert manager.handler is always_decline
    assert manager.resolve(manager.request("send_email", "Send?", {})) is False


def test_expired_requests_are_never_approved():
    clock = iter([0.0, 0.0, 10_000.0, 10_000.0])
    manager = ConfirmationManager(handler=lambda r: True, timeout=1.0,
                                  clock=lambda: next(clock))
    request = manager.request("send_email", "Send?", {})
    assert manager.resolve(request) is False
    assert request.status is ConfirmationStatus.EXPIRED


def test_cancelling_clears_the_pending_request():
    manager = ConfirmationManager(handler=lambda r: True)
    request = manager.request("send_email", "Send?", {})
    manager.cancel_pending()
    assert manager.pending is None
    assert request.status is ConfirmationStatus.CANCELLED


def test_a_broken_handler_declines_rather_than_crashing():
    def broken(request):
        raise RuntimeError("ui exploded")

    manager = ConfirmationManager(handler=broken)
    assert manager.resolve(manager.request("send_email", "Send?", {})) is False


def test_interrupting_a_prompt_declines():
    def interrupted(request):
        raise KeyboardInterrupt

    manager = ConfirmationManager(handler=interrupted)
    request = manager.request("send_email", "Send?", {})
    assert manager.resolve(request) is False
    assert request.status is ConfirmationStatus.CANCELLED


# ----------------------------------------------------------------------
# Path containment
# ----------------------------------------------------------------------
@pytest.fixture
def guard(tmp_path: Path) -> PathGuard:
    root = tmp_path / "allowed"
    root.mkdir()
    (root / "notes.txt").write_text("hello")
    (tmp_path / "secret.txt").write_text("do not read")
    return PathGuard([root])


def test_paths_inside_a_root_resolve(guard: PathGuard):
    assert guard.resolve("notes.txt", must_exist=True).name == "notes.txt"


def test_traversal_out_of_the_root_is_blocked(guard: PathGuard):
    with pytest.raises(PathNotAllowed):
        guard.resolve("../secret.txt")


def test_absolute_paths_outside_the_root_are_blocked(guard: PathGuard, tmp_path: Path):
    with pytest.raises(PathNotAllowed):
        guard.resolve(str(tmp_path / "secret.txt"))


def test_deep_traversal_is_blocked(guard: PathGuard):
    with pytest.raises(PathNotAllowed):
        guard.resolve("a/b/../../../../../../etc/passwd")


def test_empty_paths_are_rejected(guard: PathGuard):
    with pytest.raises(PathNotAllowed):
        guard.resolve("   ")


def test_missing_files_are_reported_when_required(guard: PathGuard):
    with pytest.raises(PathNotAllowed, match="does not exist"):
        guard.resolve("nope.txt", must_exist=True)


def test_no_roots_means_no_file_access():
    with pytest.raises(PathNotAllowed, match="no file roots"):
        PathGuard([]).resolve("anything.txt")


@pytest.mark.skipif(
    sys.platform == "win32" and not os.environ.get("CI"),
    reason="creating symlinks on Windows needs elevation",
)
def test_symlinks_cannot_escape_the_root(guard: PathGuard, tmp_path: Path):
    link = guard.roots[0] / "escape"
    try:
        link.symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation not permitted here")
    with pytest.raises(PathNotAllowed):
        guard.resolve("escape/secret.txt")


def test_only_known_text_types_are_readable(guard: PathGuard):
    assert guard.is_text_file(guard.roots[0] / "notes.txt")
    assert not guard.is_text_file(guard.roots[0] / "photo.png")
