"""The watchdog being down must never be silent: the person is asked what to do."""
from __future__ import annotations

import io
import json
import os
import sys
import time

import pytest

from memguard import _desktop, _health, _inbox, agent_hook, health_check

SESSION = "health-test-session"


def _stop_beating(age_s=_health.STALE_S + 60):
    old = time.time() - age_s
    os.utime(_health.HEARTBEAT, (old, old))


@pytest.fixture(autouse=True)
def _tmp_inbox(tmp_path, monkeypatch):
    monkeypatch.setattr(_inbox, "INBOX_DIR", tmp_path / "inbox")
    monkeypatch.setattr(_health, "diagnosis", lambda: "launchd state: not running")


class TestHeartbeat:
    def test_a_beating_watchdog_is_up(self):
        assert not _health.is_down()

    def test_a_stale_heartbeat_is_down(self):
        _stop_beating()
        assert _health.is_down()

    def test_a_watchdog_that_never_beat_is_down(self):
        _health.HEARTBEAT.unlink()
        assert _health.is_down()


class TestDesktop:
    @pytest.mark.parametrize("text", ["Memory low", 'say "hi"', "it's \\ odd"])
    def test_applescript_strings_are_valid(self, text):
        """Python repr() gave 'single quotes': every notification was a syntax error."""
        import subprocess
        script = f"return {_desktop.applescript_str(text)}"
        out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert out.stdout.rstrip("\n") == text


class TestHealthCheckDialog:
    @pytest.fixture
    def down(self, monkeypatch):
        _stop_beating()
        monkeypatch.setattr(health_check.time, "sleep", lambda _s: None)
        asked, done = [], []
        monkeypatch.setattr(health_check, "_show_log", lambda: done.append("log"))
        monkeypatch.setattr(_desktop, "notification", lambda t, m: done.append("notified"))
        def answer(*choices):
            replies = iter(choices)
            monkeypatch.setattr(_desktop, "dialog",
                                lambda text, buttons, timeout_s=600:
                                    (asked.append((text, buttons)), next(replies))[1])
        return asked, done, answer

    def test_nothing_happens_while_the_watchdog_beats(self, monkeypatch):
        monkeypatch.setattr(_desktop, "dialog", lambda *a, **k: pytest.fail("asked"))
        assert health_check.main() == 0

    def test_the_person_is_offered_restart_log_and_later(self, down, monkeypatch):
        asked, done, answer = down
        answer(health_check.RESTART)
        monkeypatch.setattr(_health, "restart", lambda: True)
        health_check.main()
        text, buttons = asked[0]
        assert "not running" in text and "launchd state" in text
        assert buttons == (health_check.LATER, health_check.SHOW_LOG, health_check.RESTART)
        assert done == ["notified"]

    def test_a_failed_restart_asks_again_with_what_is_left(self, down, monkeypatch):
        asked, done, answer = down
        answer(health_check.RESTART, health_check.SHOW_LOG)
        monkeypatch.setattr(_health, "restart", lambda: False)
        health_check.main()
        assert "did not help" in asked[1][0]
        assert done == ["log"]

    def test_remind_later_is_quiet_for_an_hour(self, down, monkeypatch):
        asked, _, answer = down
        answer(health_check.LATER)
        health_check.main()
        monkeypatch.setattr(_desktop, "dialog", lambda *a, **k: pytest.fail("asked again"))
        health_check.main()
        assert float(health_check.SNOOZE.read_text()) - time.time() > 3500


class TestHookDownNotice:
    def _run(self, monkeypatch):
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(
            {"session_id": SESSION, "hook_event_name": "PostToolUse"})))
        out = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        agent_hook.main()
        return out.getvalue()

    def test_silent_while_up(self, monkeypatch):
        assert self._run(monkeypatch) == ""

    def test_down_tells_person_and_agent_with_options_once(self, monkeypatch):
        _stop_beating()
        payload = json.loads(self._run(monkeypatch))
        assert "NOT running" in payload["systemMessage"]
        assert _health.restart_command() in payload["systemMessage"]
        assert "offer to restart" in payload["hookSpecificOutput"]["additionalContext"]
        assert self._run(monkeypatch) == "", "not again within DOWN_NOTICE_EVERY_S"
