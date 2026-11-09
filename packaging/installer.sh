#!/bin/bash
# "Install memguard.app" — copies memguard into ~/Library/Application Support/memguard,
# starts the watchdog (launchd agent) and optionally installs the Claude Code hook.
#   --yes   no dialogs: accept every default (scripted installs)
set -euo pipefail

YES=0; [ "${1:-}" = "--yes" ] && YES=1
PAYLOAD="$(cd "$(dirname "$0")/../Resources/payload" && pwd)"
DEST="$HOME/Library/Application Support/memguard"
LOG="$HOME/Library/Logs/memguard-install.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "=== memguard install $(date)"

say()  { [ $YES = 1 ] && { echo "$1"; return; }; osascript -e "display dialog \"$1\" buttons {\"OK\"} default button 1 with title \"memguard\" with icon note" >/dev/null; }
ask()  { [ $YES = 1 ] && return 0; osascript -e "display dialog \"$1\" buttons {\"No\", \"Yes\"} default button 2 with title \"memguard\"" 2>/dev/null | grep -q "Yes"; }
fail() { echo "ERROR: $1"; [ $YES = 1 ] || osascript -e "display dialog \"Installation failed: $1\n\nDetails: $LOG\" buttons {\"OK\"} with title \"memguard\" with icon stop" >/dev/null; exit 1; }

ask "Install memguard?\n\nIt runs a small background watchdog that stops AI agents' jobs from exhausting this Mac's memory (pauses the largest job before the machine crashes)." || exit 0

# uv provides Python and the one dependency (psutil). Apps start with a minimal PATH.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
if ! command -v uv >/dev/null; then
    ask "memguard needs uv (Python package manager by Astral). Download and install it now from astral.sh?" \
        || fail "uv is required. Install it from https://docs.astral.sh/uv/ and run the installer again."
    curl -LsSf https://astral.sh/uv/install.sh | sh || fail "could not install uv"
    command -v uv >/dev/null || fail "uv was installed but is not on PATH"
fi

mkdir -p "$DEST"
rsync -a --delete --exclude logs --exclude .venv "$PAYLOAD/" "$DEST/" || fail "could not copy files to $DEST"

mkdir -p "$HOME/.config/memguard"
[ -f "$HOME/.config/memguard/exempt" ] || printf '%s\n' \
    "# memguard: processes that must never be paused or killed." \
    "# One per line: part of a program's command line, e.g." \
    "# redis-server" \
    "# or a folder: everything running in it, or started by an agent working in it, e.g." \
    "# dir: ~/code/important_project" > "$HOME/.config/memguard/exempt"

"$DEST/tools/install_memwatch.sh" || fail "the watchdog did not start"

HOOK_MSG="not installed"
if [ -d "$HOME/.claude" ] && ask "Let Claude Code agents receive memguard's warnings?\n\nAdds a hook to ~/.claude/settings.json (a backup is kept), so an agent is told when its job uses too much memory and can shrink it before it is paused."; then
    "$DEST/tools/install_agent_hook.sh" || fail "could not install the Claude Code hook"
    HOOK_MSG="installed"
fi

say "memguard is running.\n\nClaude Code hook: $HOOK_MSG\nWatchdog log: $DEST/logs/memwatch.log\nProtected processes: ~/.config/memguard/exempt\n\nSee who uses memory (Terminal):\ncd ~/Library/Application\\\\ Support/memguard && uv run memguard/memwatch.py"
echo "=== done"
