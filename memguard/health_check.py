"""Run by launchd every minute (com.memguard.healthcheck): if the watchdog is down, ask
the person what to do: restart it, see why it failed, or be reminded later.

    uv run --no-sync python -m memguard.health_check
"""
from __future__ import annotations

import subprocess
import sys
import time

from memguard import _desktop, _health

SNOOZE = _health.STATE_DIR / "health_snooze_until"
RESTART, SHOW_LOG, LATER = "Restart memguard", "Show error log", "Remind me in 1 hour"


def _snoozed() -> bool:
    try:
        return time.time() < float(SNOOZE.read_text())
    except (FileNotFoundError, ValueError):
        return False


def _snooze(seconds: float) -> None:
    SNOOZE.parent.mkdir(parents=True, exist_ok=True)
    SNOOZE.write_text(str(time.time() + seconds))


def _show_log() -> None:
    subprocess.run(["open", "-a", "Console", str(_health.ERR_LOG)], capture_output=True)


def main() -> int:
    if not _health.is_down() or _snoozed():
        return 0
    time.sleep(5)                 # just woke from sleep? the watchdog beats within 1 s
    if not _health.is_down():
        return 0
    choice = _desktop.dialog(
        f"memguard is not running (since {_health.down_since()}), so your Mac is not "
        f"protected against running out of memory.\n\nWhy: {_health.diagnosis()}",
        (LATER, SHOW_LOG, RESTART))
    if choice == RESTART:
        if _health.restart():
            _desktop.notification("memguard", "memguard is running again.")
            return 0
        choice = _desktop.dialog(
            f"Restarting did not help.\n\nWhy: {_health.diagnosis()}\n\nReinstalling "
            f"memguard from its disk image usually fixes a broken installation.",
            (LATER, SHOW_LOG))
    if choice == SHOW_LOG:
        _show_log()
        _snooze(600)              # time to read it before being asked again
    elif choice == LATER:
        _snooze(3600)
    else:                         # unanswered: nobody at the Mac
        _snooze(600)
    return 0


if __name__ == "__main__":
    sys.exit(main())
