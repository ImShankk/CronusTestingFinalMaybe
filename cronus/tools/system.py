"""Controlled local computer actions.

Cronus has no shell. It can open a URL, launch an application the user has
explicitly allowlisted, and report machine status -- nothing else. Anything
beyond that has to be added here as a real, bounded tool.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import webbrowser
from typing import Any
from urllib.parse import urlparse

from ..logging_setup import get_logger
from .base import RiskLevel, Tool, ToolContext, ToolResult, object_schema

log = get_logger("tools.system")

# Apps that are safe to launch by name on a desktop, resolved per platform.
_BUILTIN_APPS = {
    "win32": {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "explorer": "explorer.exe",
        "file explorer": "explorer.exe",
        "paint": "mspaint.exe",
        "task manager": "taskmgr.exe",
    },
    "darwin": {
        "notes": "Notes",
        "calculator": "Calculator",
        "finder": "Finder",
        "terminal": "Terminal",
    },
    "linux": {
        "files": "xdg-open",
        "calculator": "gnome-calculator",
    },
}


def _app_table(context: ToolContext | None) -> dict[str, str]:
    key = "win32" if platform.system() == "Windows" else (
        "darwin" if platform.system() == "Darwin" else "linux"
    )
    table = dict(_BUILTIN_APPS.get(key, {}))
    if context is not None:
        table.update(context.config.security.allowed_apps)
    return table


def open_url(url: str, context: ToolContext | None = None) -> ToolResult:
    # Parse before assuming anything: a scheme-like prefix such as
    # "javascript:" must be rejected, not quietly wrapped in https://.
    parsed = urlparse(url.strip())
    if parsed.scheme and parsed.scheme not in ("http", "https"):
        return ToolResult.failure(
            f"I can only open http and https links, not {parsed.scheme}:."
        )
    if not parsed.scheme:
        parsed = urlparse(f"https://{url.strip()}")
    if not parsed.netloc or "." not in parsed.netloc:
        return ToolResult.failure(f"{url!r} is not a usable web address.")

    full_url = parsed.geturl()
    try:
        opened = webbrowser.open(full_url)
    except Exception as exc:
        log.error("failed to open browser: %s", exc)
        return ToolResult.failure("I couldn't open the browser.")
    if not opened:
        return ToolResult.failure("No browser is available to open that link.")

    log.info("opened url host=%s", parsed.netloc)
    return ToolResult(
        content=f"Opened {full_url} in the browser.", display=f"opened {parsed.netloc}"
    )


def open_app(name: str, context: ToolContext | None = None) -> ToolResult:
    """Launch an allowlisted application. Arbitrary commands are refused."""
    table = _app_table(context)
    command = table.get(name.strip().lower())
    if command is None:
        available = ", ".join(sorted(table)) or "none"
        return ToolResult.failure(
            f"{name!r} is not on the allowed application list. I can open: {available}. "
            "The user can add more with CRONUS_ALLOWED_APPS."
        )

    try:
        if platform.system() == "Windows":
            subprocess.Popen(command, shell=False)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", "-a", command])
        else:
            subprocess.Popen([command])
    except (OSError, ValueError) as exc:
        log.error("failed to launch %s: %s", name, exc)
        return ToolResult.failure(f"I couldn't start {name}.")

    log.info("launched application name=%s", name)
    return ToolResult(content=f"Opened {name}.", display=f"opened {name}")


def system_status(context: ToolContext | None = None) -> ToolResult:
    """Report machine status: battery, disk, memory, uptime."""
    lines = [f"{platform.system()} {platform.release()} on {platform.machine()}."]

    try:
        import psutil
    except ImportError:
        lines.append("Detailed stats need the psutil package, which isn't installed.")
        return ToolResult(content="\n".join(lines), display="basic system info")

    memory = psutil.virtual_memory()
    lines.append(
        f"Memory: {memory.percent:.0f} percent used "
        f"({memory.available / 1e9:.1f} GB free of {memory.total / 1e9:.1f} GB)."
    )
    lines.append(f"CPU load: {psutil.cpu_percent(interval=0.3):.0f} percent.")

    usage = shutil.disk_usage(_home_drive())
    lines.append(
        f"Disk: {usage.free / 1e9:.0f} GB free of {usage.total / 1e9:.0f} GB."
    )

    battery = getattr(psutil, "sensors_battery", lambda: None)()
    if battery is not None:
        state = "charging" if battery.power_plugged else "on battery"
        lines.append(f"Battery: {battery.percent:.0f} percent, {state}.")

    import time as _time

    uptime_hours = (_time.time() - psutil.boot_time()) / 3600
    lines.append(f"Uptime: {uptime_hours:.1f} hours.")

    return ToolResult(content="\n".join(lines), display="system status")


def _home_drive() -> str:
    from pathlib import Path

    return str(Path.home().anchor or Path.home())


def _app_preview(arguments: dict[str, Any]) -> str:
    return f"Open {arguments.get('name')} on this computer?"


def build_tools() -> list[Tool]:
    return [
        Tool(
            name="open_url",
            description="Open a web address in the user's browser.",
            parameters=object_schema(
                {"url": {"type": "string", "description": "The address to open."}},
                required=["url"],
            ),
            handler=open_url,
            risk=RiskLevel.SAFE,
            category="system",
        ),
        Tool(
            name="open_app",
            description=(
                "Launch an application on this computer. Only applications on the "
                "user's allowed list can be started."
            ),
            parameters=object_schema(
                {"name": {"type": "string", "description": "Application name."}},
                required=["name"],
            ),
            handler=open_app,
            risk=RiskLevel.LOW,
            category="system",
            preview=_app_preview,
        ),
        Tool(
            name="system_status",
            description="Report this computer's battery, memory, disk, CPU load, and uptime.",
            parameters=object_schema({}),
            handler=system_status,
            risk=RiskLevel.SAFE,
            category="system",
        ),
    ]
