"""memwatch -> agent feedback: a warning must reach the AI agent's CONTEXT, not just a screen.

Chain under test: job carries CLAUDE_CODE_SESSION_ID (inherited from the agent) ->
memwatch.notify(pid=...) reads it and posts to that session's inbox ->
memguard.agent_hook (a Claude Code hook) emits it as additionalContext, once.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time

import pytest

from memguard import _inbox, agent_hook, memwatch

SESSION = "c9af15b2-e31c-43f5-9fff-0cc71efe017b"


@pytest.fixture(autouse=True)
def _tmp_inbox(tmp_path, monkeypatch):
    monkeypatch.setattr(_inbox, "INBOX_DIR", tmp_path / "inbox")


@pytest.fixture
def agent_job():
    """A real child process started 'by an agent session' (env var inherited)."""
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        env={**os.environ, _inbox.SESSION_ENV_VAR: SESSION},
    )
    deadline = time.monotonic() + 10
    while _inbox.session_of(child.pid) is None and time.monotonic() < deadline:
        time.sleep(0.05)
    yield child
    child.kill()
    child.wait(timeout=10)


class TestSessionLookup:
    def test_reads_the_session_of_an_agent_started_job(self, agent_job):
        assert _inbox.session_of(agent_job.pid) == SESSION

    def test_job_without_an_agent_has_no_session(self):
        env = {k: v for k, v in os.environ.items() if k != _inbox.SESSION_ENV_VAR}
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                 env=env)
        try:
            time.sleep(0.3)
            assert _inbox.session_of(child.pid) is None
        finally:
            child.kill()
            child.wait(timeout=10)

    def test_unreadable_process_has_no_session(self):
        assert _inbox.session_of(1) is None          # launchd, owned by root

    @pytest.mark.parametrize("bad", ["../../etc/passwd", "a/b", "", "x" * 200])
    def test_session_ids_that_are_not_plain_ids_are_rejected(self, bad):
        with pytest.raises(ValueError):
            _inbox.inbox_path(bad)


class TestInbox:
    def test_post_then_take_delivers_exactly_once(self):
        _inbox.post(SESSION, "first")
        _inbox.post(SESSION, "second")
        text = _inbox.take(SESSION)
        assert "first" in text and "second" in text
        assert _inbox.take(SESSION) == ""

    def test_take_of_an_empty_inbox_is_empty(self):
        assert _inbox.take(SESSION) == ""


class TestNotifyReachesTheAgent:
    def test_warning_about_an_agent_job_lands_in_that_agents_inbox(self, agent_job,
                                                                  monkeypatch):
        monkeypatch.setattr(memwatch.subprocess, "run", lambda *a, **k: None)  # osascript
        memwatch.notify("Memory low — your job may be suspended soon",
                        f"python pid {agent_job.pid} using 20.7 GB", pid=agent_job.pid)
        text = _inbox.take(SESSION)
        assert "may be suspended soon" in text and str(agent_job.pid) in text

    def test_terminal_line_and_log_are_signed_memguard(self, monkeypatch, capsys,
                                                       tmp_path):
        """The product is memguard; `memwatch` is an internal module name and must not
        appear in what a person or agent reads."""
        monkeypatch.setattr(memwatch.subprocess, "run", lambda *a, **k: None)
        written = []
        real_open = open
        def fake_open(path, mode="r", *a, **k):
            if str(path).startswith("/dev/"):
                written.append(tmp_path / "tty")
                return real_open(written[-1], mode, *a, **k)
            return real_open(path, mode, *a, **k)
        monkeypatch.setattr("builtins.open", fake_open)
        memwatch.notify("SUSPENDED your job (memory)", "x", tty="ttys999")
        terminal = written[0].read_text()
        log = capsys.readouterr().out
        assert "[memguard]" in terminal and "[memguard]" in log
        assert "memwatch" not in terminal + log

    def test_dry_run_tells_no_agent_anything(self, agent_job):
        memwatch.notify("SUSPENDED your job (memory)", "x", pid=agent_job.pid,
                        dry_run=True)
        assert _inbox.take(SESSION) == ""


class TestHook:
    def _run_hook(self, monkeypatch, event: dict) -> str:
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(event)))
        out = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        assert agent_hook.main() == 0
        return out.getvalue()

    def test_hook_injects_the_message_as_additional_context(self, monkeypatch):
        _inbox.post(SESSION, "SUSPENDED your job (memory): boltz pid 25963 using 20.7 GB")
        out = self._run_hook(monkeypatch, {"session_id": SESSION,
                                           "hook_event_name": "PostToolUse"})
        payload = json.loads(out)["hookSpecificOutput"]
        assert payload["hookEventName"] == "PostToolUse"
        assert "boltz pid 25963" in payload["additionalContext"]
        assert "resumes on its own" in payload["additionalContext"]

    def test_hook_is_silent_when_there_is_nothing_to_say(self, monkeypatch):
        assert self._run_hook(monkeypatch, {"session_id": SESSION,
                                            "hook_event_name": "UserPromptSubmit"}) == ""

    def test_hook_only_delivers_this_sessions_messages(self, monkeypatch):
        _inbox.post("other-session", "not for you")
        assert self._run_hook(monkeypatch, {"session_id": SESSION,
                                            "hook_event_name": "PostToolUse"}) == ""
        assert "not for you" in _inbox.take("other-session")

    def test_hook_as_installed_command(self, tmp_path):
        """The exact command the installer registers, run as Claude Code runs it."""
        env = {**os.environ, "MEMGUARD_INBOX_DIR": str(tmp_path / "inbox2")}
        subprocess.run([sys.executable, "-c",
                        f"from memguard import _inbox; _inbox.post({SESSION!r}, 'hello agent')"],
                       env=env, check=True)
        result = subprocess.run(
            [sys.executable, "-m", "memguard.agent_hook"],
            input=json.dumps({"session_id": SESSION, "hook_event_name": "PostToolUse"}),
            capture_output=True, text=True, env=env, cwd=str(tmp_path),
        )
        assert result.returncode == 0, result.stderr
        assert "hello agent" in json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
