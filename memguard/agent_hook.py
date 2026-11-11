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

from memguard._inbox import take

PREAMBLE = (
    "memguard (the machine's memory watchdog) reports on a job YOU started. The machine "
    "is shared with other agents and can crash when memory runs out. Act on this now: "
    "reduce the job's memory (smaller batch, fewer parallel workers/folds, stream data "
    "instead of loading it whole, release GPU tensors / torch.mps.empty_cache()), or "
    "stop it and rerun it bounded. A SUSPENDED job is frozen, not dead: it resumes on "
    "its own when memory frees up.\n\n"
)


def main() -> int:
    event = json.load(sys.stdin)
    messages = take(event["session_id"])
    if not messages:
        return 0
    json.dump({
        "systemMessage": messages.rstrip("\n"),
        "hookSpecificOutput": {
            "hookEventName": event["hook_event_name"],
            "additionalContext": PREAMBLE + messages,
        }
    }, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
