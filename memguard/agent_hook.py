"""Claude Code hook: deliver memguard's messages for THIS session into the agent's context.

Registered for PostToolUse and UserPromptSubmit by `tools/install_agent_hook.sh`.
Reads the hook input from stdin, takes the session's inbox (see `memguard._inbox`),
and, if there is anything, prints it as `additionalContext` so the agent reads it on
its next step, and as `systemMessage` so the person watching that session sees it in
Claude Code's own transcript. Prints nothing when the inbox is empty: the common case
costs one failed `rename`.
"""
from __future__ import annotations

import json
import sys
import time

from memguard import _health, _inbox
from memguard._inbox import take

#: How often one session is told that the watchdog is down.
DOWN_NOTICE_EVERY_S = 600

PREAMBLE = (
    "memguard (the machine's memory watchdog) reports on a job YOU started. The machine "
    "is shared with other agents and can crash when memory runs out. Act on this now: "
    "reduce the job's memory (smaller batch, fewer parallel workers/folds, stream data "
    "instead of loading it whole, release GPU tensors / torch.mps.empty_cache()), or "
    "stop it and rerun it bounded. A SUSPENDED job is frozen, not dead: it resumes on "
    "its own when memory frees up.\n\n"
)


def _down_notice(session: str) -> tuple[str, str]:
    """(for the person, for the agent) when the watchdog is down and this session was
    not told in the last DOWN_NOTICE_EVERY_S; ("", "") otherwise. One stat when up."""
    if not _health.is_down():
        return "", ""
    marker = _inbox.inbox_path(session).with_suffix(".down-noticed")
    try:
        if time.time() - marker.stat().st_mtime < DOWN_NOTICE_EVERY_S:
            return "", ""
    except FileNotFoundError:
        pass
    _inbox.INBOX_DIR.mkdir(parents=True, exist_ok=True)
    marker.touch()
    why = _health.diagnosis()
    person = (f"memguard is NOT running (since {_health.down_since()}): your Mac is not "
              f"protected against running out of memory. Why: {why}\n"
              f"Options: restart it: ! {_health.restart_command()}  |  see the error "
              f"log: ! tail -20 \"{_health.ERR_LOG}\"  |  or ask the agent to restart it.")
    agent = (f"memguard, the memory watchdog of this Mac, is NOT running (since "
             f"{_health.down_since()}; {why}). Tell the user, and offer to restart it with "
             f"`{_health.restart_command()}`; run it only if the user agrees.")
    return person, agent


def main() -> int:
    event = json.load(sys.stdin)
    messages = take(event["session_id"])
    person, agent = _down_notice(event["session_id"])
    if messages:
        person = (person + "\n" if person else "") + messages.rstrip("\n")
        agent = (agent + "\n\n" if agent else "") + PREAMBLE + messages
    if not person:
        return 0
    json.dump({
        "systemMessage": person,
        "hookSpecificOutput": {
            "hookEventName": event["hook_event_name"],
            "additionalContext": agent,
        }
    }, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
