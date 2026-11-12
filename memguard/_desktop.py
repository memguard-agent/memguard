"""Talking to the person at the Mac: notifications and dialogs via AppleScript.

One place builds AppleScript, because AppleScript strings take DOUBLE quotes only:
Python's repr() gives 'single' quotes, which made every memguard notification a silent
AppleScript syntax error until 2026-11-12.
"""
from __future__ import annotations

import subprocess

#: Notifications started and not yet reaped (they are never waited for).
_RUNNING: list[subprocess.Popen] = []


def applescript_str(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def notification(title: str, message: str) -> None:
    """Show a notification without waiting: under memory pressure launching osascript
    can take seconds, and the watchdog must keep sampling meanwhile."""
    _RUNNING[:] = [n for n in _RUNNING if n.poll() is None]
    try:
        _RUNNING.append(subprocess.Popen(
            ["osascript", "-e", f"display notification {applescript_str(message)} "
                                f"with title {applescript_str(title)}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    except OSError:
        pass


def dialog(text: str, buttons: tuple[str, ...], timeout_s: int = 600) -> str:
    """Ask with buttons (the last is the default); the button pressed, or "" if nobody
    answered within timeout_s."""
    script = (f"display dialog {applescript_str(text)} buttons "
              f"{{{', '.join(applescript_str(b) for b in buttons)}}} default button "
              f"{applescript_str(buttons[-1])} with title \"memguard\" with icon caution "
              f"giving up after {timeout_s}")
    out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True).stdout
    for part in out.strip().split(","):
        key, _, value = part.partition(":")
        if key.strip() == "button returned":
            return value.strip()
    return ""
