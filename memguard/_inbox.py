"""Per-agent-session inbox: how memwatch's warnings reach the AI agent itself.

A warning written to a job's terminal reaches the screen, not the agent: the LLM only
sees what its tools return. And agents usually run jobs in the background, reparented
to launchd, with no terminal at all. So memwatch identifies the agent SESSION that
started a job, and an agent hook (`memguard.agent_hook`) delivers that session's
messages into the agent's context on its next tool call or prompt.

Identification: Claude Code exports `CLAUDE_CODE_SESSION_ID` into every child
process, and every descendant inherits it, even after being reparented to launchd.
memwatch reads ONLY that one variable from the job's environment
(`sysctl(KERN_PROCARGS2)`; works for the user's own processes, unprivileged).

One source of truth for the inbox location and format, used by both the writer
(memwatch) and the reader (the hook).
"""
from __future__ import annotations

import ctypes
import os
import re
import struct
import time
from pathlib import Path
from typing import Optional

SESSION_ENV_VAR = "CLAUDE_CODE_SESSION_ID"
#: The pid of the agent that started a job, likewise inherited by every descendant.
AGENT_PID_ENV_VAR = "CLAUDE_PID"

INBOX_DIR = Path(os.environ.get(
    "MEMGUARD_INBOX_DIR", os.path.expanduser("~/.cache/memguard/inbox")
))

#: Session ids become file names, so anything but a plain id is rejected.
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,128}")

_CTL_KERN = 1
_KERN_PROCARGS2 = 49
_libc = ctypes.CDLL(None, use_errno=True)


def env_var_of(pid: int, name: str) -> Optional[str]:
    """Value of one environment variable of process `pid`, or None if unreadable.

    Layout of KERN_PROCARGS2: int argc, exec path, NUL padding, argv[argc], env...
    None for other users' processes (EPERM) and processes that have exited.
    """
    mib = (ctypes.c_int * 3)(_CTL_KERN, _KERN_PROCARGS2, int(pid))
    size = ctypes.c_size_t(0)
    if _libc.sysctl(mib, 3, None, ctypes.byref(size), None, 0) != 0 or size.value < 4:
        return None
    buf = ctypes.create_string_buffer(size.value)
    if _libc.sysctl(mib, 3, buf, ctypes.byref(size), None, 0) != 0:
        return None
    raw = buf.raw[:size.value]
    argc = struct.unpack_from("i", raw)[0]
    parts = raw[4:].split(b"\0")
    i = 1                                   # parts[0] is the exec path
    while i < len(parts) and parts[i] == b"":
        i += 1
    key = name.encode() + b"="
    for entry in parts[i + argc:]:
        if entry.startswith(key):
            return entry[len(key):].decode(errors="replace")
    return None


def session_of(pid: int) -> Optional[str]:
    """The agent session that started `pid`, or None (not launched by an agent)."""
    session = env_var_of(pid, SESSION_ENV_VAR)
    if session is None or not _SESSION_ID_RE.fullmatch(session):
        return None
    return session


def inbox_path(session: str) -> Path:
    if not _SESSION_ID_RE.fullmatch(session):
        raise ValueError(f"not a valid agent session id: {session!r}")
    return INBOX_DIR / f"{session}.txt"


def post(session: str, text: str) -> Path:
    """Append one message to a session's inbox. Returns the inbox path."""
    path = inbox_path(session)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%H:%M:%S")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"[memguard {stamp}] {text}\n")
    return path


def take(session: str) -> str:
    """Remove and return everything in a session's inbox ("" when empty).

    The inbox is renamed before it is read, so a message memwatch appends at the same
    moment lands in a fresh inbox and is delivered next time instead of being lost.
    """
    path = inbox_path(session)
    claimed = path.with_name(f"{path.name}.{os.getpid()}.delivering")
    try:
        os.rename(path, claimed)
    except FileNotFoundError:
        return ""
    try:
        return claimed.read_text(encoding="utf-8")
    finally:
        claimed.unlink()
