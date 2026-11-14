"""Is the watchdog alive? One source of truth for the heartbeat, the diagnosis and the
restart, used by the daemon (writes the heartbeat), the health check (asks the person
what to do) and the agent hook (tells the person inside Claude Code).

launchd restarts a watchdog that exits, but nothing told the person when it could not
start at all, was unloaded, or hung: the Mac was silently unprotected.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Optional

LABEL = "com.memguard.memwatch"
HEALTHCHECK_LABEL = "com.memguard.healthcheck"
PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
ERR_LOG = Path(__file__).resolve().parents[1] / "logs" / "memwatch.err.log"

STATE_DIR = Path(os.environ.get("MEMGUARD_STATE_DIR",
                                os.path.expanduser("~/.cache/memguard")))
HEARTBEAT = STATE_DIR / "heartbeat"

#: The armed watchdog beats every 15 s. Older than this means it is not running or not
#: sampling; generous enough that a slow sample under pressure is not "down".
STALE_S = 90.0


def service(label: str = LABEL) -> str:
    return f"gui/{os.getuid()}/{label}"


def _loaded(label: str) -> bool:
    return subprocess.run(["launchctl", "print", service(label)],
                          capture_output=True, timeout=10).returncode == 0


def restart_argvs() -> list[list[str]]:
    """The commands that start the watchdog again from its current launchd state: a
    loaded job is kickstarted; an unloaded one (`launchctl bootout`, a partial
    uninstall) must be bootstrapped from its plist, and so must the health check if it
    was unloaded with it. kickstart alone fails on an unloaded job."""
    if _loaded(LABEL):
        argvs = [["launchctl", "kickstart", "-k", service()]]
    else:
        argvs = [["launchctl", "bootstrap", f"gui/{os.getuid()}", str(PLIST)]]
    if not _loaded(HEALTHCHECK_LABEL):
        argvs.append(["launchctl", "bootstrap", f"gui/{os.getuid()}",
                      str(PLIST.with_name(f"{HEALTHCHECK_LABEL}.plist"))])
    return argvs


def restart_command() -> str:
    """restart_argvs() as one line for a person to run."""
    return " && ".join(shlex.join(argv) for argv in restart_argvs())


def beat() -> None:
    """Called by the armed watchdog every memwatch.HEARTBEAT_EVERY_S (15 s)."""
    try:
        os.utime(HEARTBEAT)
    except FileNotFoundError:
        HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
        HEARTBEAT.touch()


def heartbeat_age_s() -> Optional[float]:
    """Seconds since the last beat, or None if the watchdog never beat."""
    try:
        return time.time() - HEARTBEAT.stat().st_mtime
    except FileNotFoundError:
        return None


def is_down() -> bool:
    age = heartbeat_age_s()
    return age is None or age > STALE_S


def down_since() -> str:
    """When the last beat was, for messages ("15:10"), or "unknown"."""
    age = heartbeat_age_s()
    return "unknown" if age is None else time.strftime(
        "%H:%M", time.localtime(time.time() - age))


def diagnosis() -> str:
    """Why it is down, in one line, from launchd and the watchdog's error log."""
    result = subprocess.run(["launchctl", "print", service()],
                            capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        return "it is not loaded in launchd (stopped or partly uninstalled)"
    fields = {}
    for line in result.stdout.splitlines():
        key, _, value = line.strip().partition(" = ")
        fields.setdefault(key, value)
    reason = f"launchd state: {fields.get('state', '?')}"
    if fields.get("last exit code"):
        reason += f", last exit code {fields['last exit code']}"
    try:
        lines = [l for l in ERR_LOG.read_text(errors="replace").splitlines() if l.strip()]
        if lines:
            reason += f"; last error: {lines[-1].strip()[:160]}"
    except FileNotFoundError:
        pass
    return reason


def restart(wait_s: float = 20.0) -> bool:
    """Start the watchdog again; True once it is beating."""
    for argv in restart_argvs():
        subprocess.run(argv, capture_output=True, timeout=30)
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if not is_down() and heartbeat_age_s() < 5:
            return True
        time.sleep(1)
    return False
