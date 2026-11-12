"""memwatch: attribution, exemptions, and the watchdog state machine.

The watchdog decides whether to SIGSTOP or SIGKILL a real process, so its decision
function is pure and tested here with injected readings — no signal is ever sent by
this suite.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from memguard.memwatch import (  # noqa: E402
    RESUME_SAMPLES, SAMPLE_SECONDS, WARN_GRACE_SAMPLES, ProcInfo, WatchState, WindowUsage,
    _clean_title, build_usage, decide_action, pick_victim, resolve_ttys,
)


def proc(pid, ppid=1, tty="ttys001", rss_gb=1.0, command="python job.py"):
    return ProcInfo(pid=pid, ppid=ppid, tty=tty, rss_gb=rss_gb, command=command)


class TestExemptions:
    @pytest.mark.parametrize("command", [
        "python protected_service.py --port 8000",
        "redis-server *:6379",
        "/Users/x/code/myproject/.venv/bin/python3 protected_service.py",
        "/Applications/Database.app/Contents/MacOS/db",
    ])
    def test_processes_in_the_exempt_file_are_never_targeted(self, command):
        """Services named in the machine's exemption file must never stall."""
        assert proc(1, command=command).is_exempt
        assert not proc(1, command=command).is_targetable

    def test_exempt_file_parsing(self, tmp_path):
        from memguard.memwatch import load_exempt
        f = tmp_path / "exempt"
        f.write_text("# comment\nservice_a\n\n  service b  # trailing comment\n"
                     f"dir: {tmp_path}/proj  # a folder\n")
        assert load_exempt(str(f)) == (("service_a", "service b"),
                                       (os.path.realpath(f"{tmp_path}/proj"),))

    def test_missing_exempt_file_means_no_exemptions(self, tmp_path):
        from memguard.memwatch import load_exempt
        assert load_exempt(str(tmp_path / "absent")) == ((), ())

    def test_the_claude_cli_is_never_the_victim(self):
        """Killing the agent leaves its runaway job running — strictly worse."""
        assert not proc(1, command="claude").is_targetable

    def test_ordinary_compute_is_targetable(self):
        assert proc(1, command="python research/measure_variants.py").is_targetable


class TestAttribution:
    def test_detached_child_is_charged_to_the_launching_window(self):
        """A backgrounded job shows tty '??' but belongs to the session that spawned it."""
        procs = [
            proc(100, ppid=1, tty="ttys025", command="login"),
            proc(200, ppid=100, tty="", rss_gb=5.0, command="python heavy.py"),
        ]
        assert resolve_ttys(procs)[200] == "ttys025"

    def test_genuinely_detached_project_process_is_bucketed_not_dropped(self):
        """A project background job reparented to launchd (no tty anywhere in its
        ancestry — e.g. a nohup'd sweep whose launching shell has since exited)
        must still be visible and still count toward totals, not vanish from
        every window.
        """
        procs = [
            proc(1, tty="ttys001", rss_gb=1.0),
            # ppid=999 is not any pid in this list, mimicking a chain that walks all
            # the way to launchd (pid 1, absent here) without finding a tty.
            proc(500, ppid=999, tty="", rss_gb=7.0,
                 command="/Users/x/code/myproject/.venv/bin/python heavy_sweep.py"),
        ]
        windows = build_usage(procs, {"ttys001": "small"})
        assert sum(w.total_gb for w in windows) == pytest.approx(8.0), (
            "detached project process's memory must still be counted somewhere"
        )
        detached = [w for w in windows if w.tty != "ttys001"]
        assert len(detached) == 1
        assert detached[0].total_gb == pytest.approx(7.0)

    def test_genuinely_detached_non_project_process_is_excluded(self):
        """A tty-less OS/GUI daemon (WindowServer, Finder, Dropbox, ...) must NOT
        land in the operator-facing 'agent-owned' DETACHED_TTY bucket, or
        `--kill detached` would SIGKILL WindowServer and Terminal.app.
        """
        procs = [
            proc(1, tty="ttys001", rss_gb=1.0),
            proc(500, ppid=999, tty="", rss_gb=3.0,
                 command="/System/Library/PrivateFrameworks/SkyLight.framework"
                         "/Resources/WindowServer"),
        ]
        windows = build_usage(procs, {"ttys001": "small"})
        assert [w.tty for w in windows] == ["ttys001"], (
            "a non-project, tty-less process must not create/join a detached bucket"
        )
        assert sum(w.total_gb for w in windows) == pytest.approx(1.0)

    def test_usage_groups_and_sorts_by_total(self):
        procs = [
            proc(1, tty="ttys001", rss_gb=1.0),
            proc(2, tty="ttys002", rss_gb=4.0),
            proc(3, tty="ttys001", rss_gb=0.5),
        ]
        windows = build_usage(procs, {"ttys001": "small", "ttys002": "big"})
        assert [w.tty for w in windows] == ["ttys002", "ttys001"]
        assert windows[1].total_gb == pytest.approx(1.5)

    def test_idle_window_reports_no_top_child(self):
        """An idle session must not report its `login` shell as a finding."""
        window = WindowUsage(tty="ttys001", title="idle",
                             procs=[proc(1, rss_gb=0.05, command="login")])
        assert window.top_child is None

    def test_footprint_supersedes_rss_when_measured(self):
        """Judge by committed footprint: RSS over-states mmap-heavy jobs and does not
        see Metal/MPS (GPU) allocations at all."""
        p = proc(1, rss_gb=6.0)
        assert p.best_gb == 6.0
        p.footprint_gb = 2.0
        assert p.best_gb == 2.0


class TestTitleCleaning:
    def test_extracts_the_task_name_claude_code_set(self):
        raw = "myproject — ⠐ Fix memory issues with parallel agent jobs — caffeinate ◂ claude — 87×26"
        assert _clean_title(raw) == "Fix memory issues with parallel agent jobs"

    def test_strips_the_status_glyph(self):
        raw = "myproject — ✳ Add filtering options to the plots — claude — 80×24"
        assert _clean_title(raw) == "Add filtering options to the plots"

    def test_plain_shell_window_survives(self):
        assert "zsh" in _clean_title("research — -zsh — 187×37")


class TestWatchdogDecisions:
    def test_healthy_box_does_nothing(self):
        assert decide_action(free_gb=12.0, state=WatchState()) is None

    def test_warns_before_acting(self):
        assert decide_action(5.0, WatchState()) == "warn"

    def test_warns_the_victim_before_suspending_at_the_stop_threshold(self):
        """The FIRST low-memory sample now warns instead of jumping straight to
        'stop' — the victim gets a chance to react (e.g. switch to the streaming
        loader) before actually being suspended. The stop/kill mechanism itself is
        unchanged; this is a new step inserted before it.
        """
        assert decide_action(3.0, WatchState()) == "warn_victim"

    def test_waits_out_the_grace_period_after_warning(self):
        """Still critical, already warned, grace not yet elapsed -> do nothing (no
        repeated warnings, no premature stop)."""
        state = WatchState(warned_pid=99, warned_samples=WARN_GRACE_SAMPLES - 2)
        assert decide_action(3.0, state) is None

    def test_suspends_once_the_grace_period_elapses(self):
        """Still critical after WARN_GRACE_SAMPLES samples since the warning ->
        proceed to 'stop' exactly as the unmodified code always did."""
        state = WatchState(warned_pid=99, warned_samples=WARN_GRACE_SAMPLES - 1)
        assert decide_action(3.0, state) == "stop"

    def test_severe_pressure_bypasses_the_grace_period_entirely(self):
        """The grace period is a courtesy for SHALLOW pressure only. Free memory
        already past KILL_FREE_GB — a genuine OOM cliff — must stop IMMEDIATELY,
        with no warning and no wait: waiting WARN_GRACE_SAMPLES samples to even
        attempt a first suspension at that depth would leave the box with NO
        mitigation during the exact window the watchdog exists for. It is the
        DISTINCT action 'stop_severe', not 'stop' — the caller must pick a fresh
        victim for it, never a stale warned_pid (see TestWarnBeforeStop).

        Regression: an earlier version had no
        severity bypass, so a cliff to 0.3 GB free / 9 GB swap left the box doing
        nothing for ~27s (9 samples) before its first SIGSTOP attempt. A second
        pre-merge finding: an earlier fix reused the 'stop' string for this case,
        which let a stale warned_pid from an earlier, shallower episode hijack the
        target at exactly the moment size matters most.
        """
        assert decide_action(0.3, WatchState()) == "stop_severe"

    def test_severity_bypass_is_gated_on_free_memory_only_never_swap(self):
        """swap_used_gb is NOT a reliable pressure signal on macOS — it can sit far
        above any fixed swap threshold from ordinary background usage while free memory is
        genuinely healthy or only shallowly low. The severity bypass must never
        fire on swap alone, at ANY free-memory level.

        Regression for TWO earlier versions, both reproduced against a
        box's REAL readings (~25-30 GB swap used at rest):
        - First version: the bypass checked free memory OR swap
          unconditionally and would have returned 'stop_severe' even with 12 GB
          genuinely free, purely from swap — then 'kill' on the very next
          sample, looping stop-then-kill on a healthy box.
        - Second version: the bypass added a `free_gb < STOP_FREE_GB` gate but
          KEPT the swap disjunct, which fires on EVERY entry into the stop
          regime on such a box (the swap condition was permanently true
          there too) — collapsing the whole feature back to the pre-existing
          immediate-'stop' behavior with the grace period never reachable in
          practice. This is the case that matters most: 3.0 GB free (inside the
          stop regime) with high swap must warn, not bypass.
        """
        assert decide_action(12.0, WatchState()) is None
        assert decide_action(8.0, WatchState()) is None
        assert decide_action(5.0, WatchState()) == "warn"
        assert decide_action(3.0, WatchState()) == "warn_victim", (
            "high swap alone, inside the stop regime, must still go through the "
            "grace path — not bypass it"
        )

    def test_severity_bypass_fires_even_with_a_pending_warning(self):
        """A shallow-pressure warning can be PENDING (warned_pid set, grace not yet
        elapsed) when severity suddenly spikes — the bypass must still fire
        'stop_severe' (never fall through to the grace-period 'stop'/None logic),
        so the caller picks a FRESH victim rather than the possibly-smaller one
        that happened to be warned earlier.
        """
        state = WatchState(warned_pid=99, warned_samples=1)
        assert decide_action(0.3, state) == "stop_severe"

    def test_does_not_suspend_a_second_process(self):
        """One victim at a time; escalate on that one rather than freezing the box."""
        assert decide_action(3.0, WatchState(stopped_pid=42)) is None

    def test_escalates_to_kill_when_suspension_did_not_help(self):
        """SIGSTOP does not free memory, so the kill escalation is genuinely required."""
        assert decide_action(1.5, WatchState(stopped_pid=42)) == "kill"

    def test_swap_alone_never_escalates_to_kill(self):
        """Regression: kill used to fire on `swap used > 2 GB` alone. macOS keeps
        tens of GB of swap in use at rest, so every suspended victim was killed on
        the very next sample (observed live: killed at 3.4 and 4.1 GB free). Only
        free memory under KILL_FREE_GB may escalate."""
        assert decide_action(4.0, WatchState(stopped_pid=42)) is None

    def test_resumes_once_the_box_has_drained(self):
        state = WatchState(stopped_pid=42, recovered_samples=RESUME_SAMPLES - 1)
        assert decide_action(9.0, state) == "resume"

    def test_does_not_resume_on_a_single_good_sample(self):
        """Resuming on one lucky reading would oscillate."""
        state = WatchState(stopped_pid=42, recovered_samples=0)
        assert decide_action(9.0, state) is None


class TestWarnBeforeStop:
    """The daemon's stop/kill escalation is unchanged; a targeted warning to the
    victim, with a grace period to react, is now inserted before the SIGSTOP.
    Driven end-to-end through run_daemon (not just decide_action) so the actual
    notify()/_signal_pid() call sequence is what's under test, not just the
    decision function.
    """

    @staticmethod
    def _drive(monkeypatch, free_gb_sequence, victim_pid=77, clock_step=200.0,
               events=None):
        """Drive run_daemon. `clock_step` controls the fake time.monotonic() advance
        PER CALL — large by default (200s) so the per-victim notification's 120s
        throttle never interferes with tests that are not specifically about it (a
        real test run's actual wall-clock elapses in milliseconds, which would
        otherwise throttle every notification after the first regardless of how many
        distinct warning episodes the test drives). Pass a small clock_step (e.g.
        SAMPLE_SECONDS) to instead exercise the throttle itself.
        """
        from memguard import memwatch

        notifications = []
        signaled = []
        samples = iter(free_gb_sequence)
        clock = [0.0]

        def fake_memory():
            try:
                return next(samples)
            except StopIteration:
                raise KeyboardInterrupt

        def fake_monotonic():
            clock[0] += clock_step
            return clock[0]

        monkeypatch.setattr(memwatch, "system_memory", fake_memory)
        monkeypatch.setattr(memwatch, "scan_processes", lambda: [
            ProcInfo(pid=victim_pid, ppid=1, tty="ttys999", rss_gb=9.0,
                     command="python heavy.py"),
        ])
        monkeypatch.setattr(memwatch, "measure_footprints", lambda procs: None)
        monkeypatch.setattr(memwatch, "resolve_ttys", lambda procs: {victim_pid: "ttys999"})
        events = [] if events is None else events     # notify + signal, in call order
        monkeypatch.setattr(memwatch, "notify",
                            lambda title, message, tty="", dry_run=False, pid=None:
                                (notifications.append((title, message)),
                                 events.append(("notify", title))))
        monkeypatch.setattr(memwatch, "_signal_pid",
                            lambda pid, sig, verb, dry_run=False:
                                (signaled.append((pid, sig)), events.append(("signal", sig))))
        monkeypatch.setattr(memwatch, "_other_daemon_pids", lambda: [])
        monkeypatch.setattr(memwatch.time, "sleep", lambda _s: None)
        monkeypatch.setattr(memwatch.time, "monotonic", fake_monotonic)

        memwatch.run_daemon()
        return notifications, signaled

    def test_first_low_memory_sample_warns_but_does_not_suspend(self, monkeypatch):
        notifications, signaled = self._drive(monkeypatch, [(3.0, 24.0, 0.0)])

        assert len(notifications) == 1
        title, message = notifications[0]
        assert "suspended soon" in title
        assert "77" in message
        from memguard.memwatch import STREAMING_LOADER_GUIDANCE
        assert STREAMING_LOADER_GUIDANCE in message, (
            "must tell the agent how to bound its job"
        )
        assert signaled == [], "must not SIGSTOP on the first low-memory sample — only warn"

    def test_stop_fires_only_after_the_grace_period(self, monkeypatch):
        import signal as sig_mod

        # +1 sample past the grace period so run_daemon's KeyboardInterrupt fires
        # AFTER the stop — its shutdown safety net then resumes the victim it just
        # stopped (see TestDaemonShutdown), which is a second, expected signal.
        _notifications, signaled = self._drive(
            monkeypatch, [(3.0, 24.0, 0.0)] * (WARN_GRACE_SAMPLES + 1))

        assert signaled == [(77, sig_mod.SIGSTOP), (77, sig_mod.SIGCONT)], (
            "must SIGSTOP exactly once, only after the grace period elapses, then "
            "the shutdown safety net resumes it"
        )

    def test_severity_bypass_suspends_immediately_with_no_warning(self, monkeypatch):
        """A genuine OOM cliff (free < KILL_FREE_GB) must SIGSTOP on the very FIRST
        sample, with zero warn_victim notifications — the grace period must never
        intervene here.

        Regression for a test gap: `victim = pick_victim(procs)` on
        the stop_severe path could be deleted entirely — i.e. NOTHING is ever
        suspended at a cold OOM cliff — and the full suite still passed, because no
        test drove run_daemon through a cold-start cliff end-to-end (the shutdown
        test was the only thing near this path, and it asserts nothing about
        whether a suspension actually happened).
        """
        import signal as sig_mod

        notifications, signaled = self._drive(monkeypatch, [(1.0, 24.0, 0.0)])

        assert signaled == [(77, sig_mod.SIGSTOP), (77, sig_mod.SIGCONT)], (
            "must SIGSTOP on the very first sample at a genuine OOM cliff, then the "
            "shutdown safety net resumes it"
        )
        assert not any(t == "Memory low — your job may be suspended soon"
                       for t, _m in notifications), (
            "the severity bypass must never route through the warning path"
        )

    def test_severity_bypass_targets_the_largest_process_even_with_a_pending_warning(
        self, monkeypatch
    ):
        """A shallow-pressure warning can be PENDING (warned_pid set, grace not yet
        elapsed) when severity suddenly spikes. The severity bypass must target the
        LARGEST process right now, never the smaller one that merely happened to be
        warned earlier — 'stop_severe' must pick fresh, not reuse warned_pid.

        Regression: conflating 'stop' (grace
        elapsed, target the warned pid) and 'stop_severe' (target the largest
        process) under one 'stop' string would have suspended the smaller,
        already-warned process at exactly the moment size matters most, while the
        actual largest consumer kept growing.
        """
        from memguard import memwatch
        import signal as sig_mod

        victim_a = ProcInfo(pid=77, ppid=1, tty="ttys999", rss_gb=5.0, command="python a.py")
        victim_b = ProcInfo(pid=88, ppid=1, tty="ttys998", rss_gb=12.0, command="python b.py")
        call_count = [0]

        def fake_scan():
            call_count[0] += 1
            # A alone during the shallow-pressure warning; B (much larger) appears
            # only once the box crashes to a severe cliff.
            return [victim_a] if call_count[0] == 1 else [victim_a, victim_b]

        signaled = []
        samples = iter([
            (3.0, 24.0, 0.0),   # shallow pressure -> warns A
            (0.5, 24.0, 9.0),   # sudden cliff -> stop_severe, B now present too
        ])

        def fake_memory():
            try:
                return next(samples)
            except StopIteration:
                raise KeyboardInterrupt

        monkeypatch.setattr(memwatch, "system_memory", fake_memory)
        monkeypatch.setattr(memwatch, "scan_processes", fake_scan)
        monkeypatch.setattr(memwatch, "measure_footprints", lambda procs: None)
        monkeypatch.setattr(memwatch, "resolve_ttys",
                            lambda procs: {77: "ttys999", 88: "ttys998"})
        monkeypatch.setattr(memwatch, "notify", lambda *a, **k: None)
        monkeypatch.setattr(memwatch, "_signal_pid",
                            lambda pid, sig, verb, dry_run=False: signaled.append((pid, sig)))
        monkeypatch.setattr(memwatch, "_other_daemon_pids", lambda: [])
        monkeypatch.setattr(memwatch.time, "sleep", lambda _s: None)

        memwatch.run_daemon()

        assert (88, sig_mod.SIGSTOP) in signaled, (
            "the severity bypass must target the LARGEST current process (B), "
            "never the smaller one (A) that merely had a pending warning"
        )
        assert (77, sig_mod.SIGSTOP) not in signaled, (
            "A must not be suspended just because it happened to be warned earlier"
        )

    def test_non_consecutive_recovery_touches_do_not_prevent_eventual_stop(self, monkeypatch):
        """Reviewer's exact repro for a wider F2 regression: alternating a
        just-below-STOP_FREE_GB sample with an ISOLATED (non-consecutive)
        at-or-above-RESUME_FREE_GB sample must not let genuinely sustained
        sub-threshold pressure escape 'stop' forever. warn_recovered_samples resets
        on every low sample (only a truly SUSTAINED recovery clears the warning),
        so warned_samples keeps accumulating on the low samples regardless of the
        isolated highs mixed in, until the grace period elapses.
        """
        import signal as sig_mod

        sequence = [(3.4, 24.0, 0.0), (7.0, 24.0, 0.0)] * WARN_GRACE_SAMPLES
        _notifications, signaled = self._drive(monkeypatch, sequence)

        assert (77, sig_mod.SIGSTOP) in signaled, (
            "isolated (non-consecutive) touches at RESUME_FREE_GB must not prevent "
            "'stop' from firing under sustained underlying low-memory pressure"
        )

    def test_grace_period_resets_only_after_sustained_recovery(self, monkeypatch):
        """A pending warning clears only after RESUME_SAMPLES CONSECUTIVE samples at
        RESUME_FREE_GB — the same hysteresis as resuming an already-stopped victim,
        not a single tick back above STOP_FREE_GB. After that sustained recovery, a
        later low-memory episode warns again rather than skipping straight to stop
        off a stale counter.
        """
        notifications, signaled = self._drive(monkeypatch,
            [(3.0, 24.0, 0.0)]                    # low -> warns
            + [(10.0, 24.0, 0.0)] * RESUME_SAMPLES  # sustained recovery -> clears it
            + [(3.0, 24.0, 0.0)]                    # low again -> must warn again
        )

        warn_titles = [t for t, _m in notifications
                      if t == "Memory low — your job may be suspended soon"]
        assert len(warn_titles) == 2, (
            "sustained recovery must clear the warning, so a later low-memory "
            "episode warns again"
        )
        assert signaled == [], "must never actually stop across this sequence"

    def test_repeated_warning_notifications_are_throttled(self, monkeypatch):
        """Rapid bounce (warns, sustained-recovers, warns again, ...) for the SAME
        process must not spam a fresh desktop notification / tty write on every
        single cycle — that costs an osascript subprocess per firing (the
        scan_processes/refine_top scan itself still runs every warn_victim sample
        regardless, so the throttle saves the notification, not the scan). The
        underlying grace-period STATE still re-arms every cycle (signaled stays
        empty, exactly as an un-throttled run would show); only the notification is
        suppressed within the 120s window. Uses a REALISTIC clock_step (matching
        SAMPLE_SECONDS), unlike the other end-to-end tests here (which use a large
        clock_step specifically to neutralize the throttle) — this is the one test
        for the throttle itself.
        """
        from memguard import memwatch

        proc = ProcInfo(pid=77, ppid=1, tty="ttys999", rss_gb=9.0, command="python heavy.py")
        notifications = []
        signaled = []
        # A shallow bounce (6.0 <-> 3.4 GB): a 7 GB plunge in one sample would now be
        # suspended at once as a fast fall (see TestFastFall), not throttled.
        cycle = [(3.4, 24.0, 0.0)] + [(6.0, 24.0, 0.0)] * RESUME_SAMPLES
        samples = iter(cycle * 5)
        # Large base so the FIRST notification isn't itself throttled (mirrors a
        # real daemon's actual wall-clock start time, unlike a fake clock starting
        # at 0 racing last_warn_victim's own 0.0 initial value).
        clock = [1000.0]

        def fake_memory():
            try:
                return next(samples)
            except StopIteration:
                raise KeyboardInterrupt

        def fake_monotonic():
            clock[0] += SAMPLE_SECONDS
            return clock[0]

        monkeypatch.setattr(memwatch, "system_memory", fake_memory)
        monkeypatch.setattr(memwatch, "scan_processes", lambda: [proc])
        monkeypatch.setattr(memwatch, "measure_footprints", lambda procs: None)
        monkeypatch.setattr(memwatch, "resolve_ttys", lambda procs: {77: "ttys999"})
        monkeypatch.setattr(memwatch, "notify",
                            lambda title, message, tty="", dry_run=False, pid=None:
                                notifications.append((title, message)))
        monkeypatch.setattr(memwatch, "_signal_pid",
                            lambda pid, sig, verb, dry_run=False: signaled.append((pid, sig)))
        monkeypatch.setattr(memwatch, "_other_daemon_pids", lambda: [])
        monkeypatch.setattr(memwatch.time, "sleep", lambda _s: None)
        monkeypatch.setattr(memwatch.time, "monotonic", fake_monotonic)

        memwatch.run_daemon()

        warn_titles = [t for t, _m in notifications
                      if t == "Memory low — your job may be suspended soon"]
        assert len(warn_titles) == 1, (
            "repeated warn-recover cycles within the 120s throttle window must "
            "notify only once, not spam a fresh notification every cycle"
        )
        assert signaled == [], "must never actually stop across this bouncing sequence"

    def test_a_new_victim_is_warned_even_inside_another_victims_cooldown(self, monkeypatch):
        """The 120s throttle is keyed on the PREVIOUSLY-WARNED PID, not on wall time
        alone — a BRAND NEW victim (e.g. a bigger process that only appears after
        the first one recovered) must always get its own first warning, even
        inside the cooldown window a DIFFERENT process's earlier warning started.

        Regression, reproduced end-to-end
        with a wall-time-only cooldown: process B was SIGSTOPped having never
        received any warning, because process A's cooldown was still active.
        """
        from memguard import memwatch

        victim_a = ProcInfo(pid=77, ppid=1, tty="ttys999", rss_gb=5.0, command="python a.py")
        victim_b = ProcInfo(pid=88, ppid=1, tty="ttys998", rss_gb=12.0, command="python b.py")
        call_count = [0]

        def fake_scan():
            call_count[0] += 1
            # A alone through its warn + recovery cycle; B appears only afterward.
            return [victim_a] if call_count[0] <= 1 else [victim_b]

        notifications = []
        clock = [1000.0]
        samples = iter(
            [(3.0, 24.0, 0.0)]                      # warns A
            + [(10.0, 24.0, 0.0)] * RESUME_SAMPLES    # sustained recovery -> clears A
            + [(3.0, 24.0, 0.0)]                      # B (heavier) is now present
        )

        def fake_memory():
            try:
                return next(samples)
            except StopIteration:
                raise KeyboardInterrupt

        def fake_monotonic():
            # Small step: stays well INSIDE A's 120s cooldown window when B
            # appears, so this can only pass if the throttle is keyed on the pid.
            clock[0] += SAMPLE_SECONDS
            return clock[0]

        monkeypatch.setattr(memwatch, "system_memory", fake_memory)
        monkeypatch.setattr(memwatch, "scan_processes", fake_scan)
        monkeypatch.setattr(memwatch, "measure_footprints", lambda procs: None)
        monkeypatch.setattr(memwatch, "resolve_ttys",
                            lambda procs: {77: "ttys999", 88: "ttys998"})
        monkeypatch.setattr(memwatch, "notify",
                            lambda title, message, tty="", dry_run=False, pid=None:
                                notifications.append((title, message)))
        monkeypatch.setattr(memwatch, "_signal_pid", lambda *a, **k: None)
        monkeypatch.setattr(memwatch, "_other_daemon_pids", lambda: [])
        monkeypatch.setattr(memwatch.time, "sleep", lambda _s: None)
        monkeypatch.setattr(memwatch.time, "monotonic", fake_monotonic)

        memwatch.run_daemon()

        assert any("88" in message for _t, message in notifications), (
            "B must get its own warning even inside A's still-active cooldown window"
        )

    def test_a_single_tick_above_stop_threshold_does_not_clear_the_warning(self, monkeypatch):
        """Regression: the ORIGINAL grace
        logic cleared warned_pid on ANY sample back above STOP_FREE_GB (zero
        hysteresis), so memory oscillating around that threshold re-warned forever
        and the SIGSTOP never fired — a livelock. Sustained sub-STOP_FREE_GB
        pressure, even with a few ticks into the STOP_FREE_GB..RESUME_FREE_GB band
        mixed in (never reaching genuine RESUME_FREE_GB recovery), must still
        eventually reach 'stop'.
        """
        import signal as sig_mod

        # Oscillate just below/above STOP_FREE_GB (3.5) — well short of
        # RESUME_FREE_GB (6.0) — for long enough that WARN_GRACE_SAMPLES
        # sub-threshold samples accumulate regardless of the noise.
        sequence = [(3.4, 24.0, 0.0), (3.6, 24.0, 0.0)] * WARN_GRACE_SAMPLES
        _notifications, signaled = self._drive(monkeypatch, sequence)

        assert (77, sig_mod.SIGSTOP) in signaled, (
            "oscillation around STOP_FREE_GB must not prevent 'stop' from ever "
            "firing — that is the livelock this fix closes"
        )

    def test_stops_the_same_process_that_was_warned_not_a_later_heavier_one(self, monkeypatch):
        """The grace period promises the WARNED process a chance to react — not
        whichever process happens to be heaviest once the grace period elapses.

        Regression: the 'stop' branch used to
        re-run pick_victim() from scratch, so a NEW heavier process appearing during
        the grace window would be suspended instead of the one actually warned.
        """
        from memguard import memwatch
        import signal as sig_mod

        victim_a = ProcInfo(pid=77, ppid=1, tty="ttys999", rss_gb=5.0, command="python a.py")
        victim_b = ProcInfo(pid=88, ppid=1, tty="ttys998", rss_gb=9.0, command="python b.py")
        call_count = [0]

        def fake_scan():
            call_count[0] += 1
            # A alone at warning time; B (heavier) appears on every later scan.
            return [victim_a] if call_count[0] == 1 else [victim_a, victim_b]

        signaled = []
        samples = iter([(3.0, 24.0, 0.0)] * WARN_GRACE_SAMPLES)

        def fake_memory():
            try:
                return next(samples)
            except StopIteration:
                raise KeyboardInterrupt

        monkeypatch.setattr(memwatch, "system_memory", fake_memory)
        monkeypatch.setattr(memwatch, "scan_processes", fake_scan)
        monkeypatch.setattr(memwatch, "measure_footprints", lambda procs: None)
        monkeypatch.setattr(memwatch, "resolve_ttys",
                            lambda procs: {77: "ttys999", 88: "ttys998"})
        monkeypatch.setattr(memwatch, "notify", lambda *a, **k: None)
        monkeypatch.setattr(memwatch, "_signal_pid",
                            lambda pid, sig, verb, dry_run=False: signaled.append((pid, sig)))
        monkeypatch.setattr(memwatch, "_other_daemon_pids", lambda: [])
        monkeypatch.setattr(memwatch.time, "sleep", lambda _s: None)

        memwatch.run_daemon()

        assert (77, sig_mod.SIGSTOP) in signaled, "the warned process (A) must be stopped"
        assert (88, sig_mod.SIGSTOP) not in signaled, (
            "B was never warned and must not be substituted in just for being heavier"
        )

    def test_stops_nobody_if_the_warned_process_already_exited(self, monkeypatch):
        """If the warned process exits on its own before the grace period elapses,
        a DIFFERENT, never-warned process must not be punished in its place — the
        next low-memory sample re-warns whoever is heaviest then, fresh.
        """
        from memguard import memwatch

        victim_a = ProcInfo(pid=77, ppid=1, tty="ttys999", rss_gb=5.0, command="python a.py")
        victim_b = ProcInfo(pid=88, ppid=1, tty="ttys998", rss_gb=9.0, command="python b.py")
        call_count = [0]

        def fake_scan():
            call_count[0] += 1
            # A is warned, then exits; only B is present from then on.
            return [victim_a] if call_count[0] == 1 else [victim_b]

        notifications = []
        signaled = []
        samples = iter([(3.0, 24.0, 0.0)] * (WARN_GRACE_SAMPLES + 1))
        clock = [0.0]

        def fake_memory():
            try:
                return next(samples)
            except StopIteration:
                raise KeyboardInterrupt

        def fake_monotonic():
            # Large step so the per-victim notification's 120s throttle never
            # interferes here — this test is about WHO gets warned, not about the
            # throttle (that has its own dedicated test).
            clock[0] += 200.0
            return clock[0]

        monkeypatch.setattr(memwatch, "system_memory", fake_memory)
        monkeypatch.setattr(memwatch, "scan_processes", fake_scan)
        monkeypatch.setattr(memwatch, "measure_footprints", lambda procs: None)
        monkeypatch.setattr(memwatch, "resolve_ttys",
                            lambda procs: {77: "ttys999", 88: "ttys998"})
        monkeypatch.setattr(memwatch, "notify",
                            lambda title, message, tty="", dry_run=False, pid=None:
                                notifications.append((title, message)))
        monkeypatch.setattr(memwatch, "_signal_pid",
                            lambda pid, sig, verb, dry_run=False: signaled.append((pid, sig)))
        monkeypatch.setattr(memwatch, "_other_daemon_pids", lambda: [])
        monkeypatch.setattr(memwatch.time, "sleep", lambda _s: None)
        monkeypatch.setattr(memwatch.time, "monotonic", fake_monotonic)

        memwatch.run_daemon()

        assert signaled == [], (
            "nobody was warned-and-still-present at grace-elapse time, so nobody "
            "should be stopped this cycle"
        )
        assert any("88" in message for _title, message in notifications), (
            "B must get its OWN fresh warning once A is gone, not be skipped"
        )

    def test_full_stop_to_kill_cycle_still_warns_fresh_on_the_next_episode(self, monkeypatch):
        """After a complete warn -> grace -> stop -> kill cycle, a LATER, unrelated
        low-memory episode must warn again from scratch — not skip straight to stop
        off warned-state left over from the earlier cycle.

        Regression: mutation-deleting the warned-state
        reset in the 'stop' branch survived the original test suite — this pins the
        observable consequence (a second warning fires) rather than the reset line
        itself, so it can't be silently deleted again without a test failing.
        """
        from memguard import memwatch

        proc = ProcInfo(pid=77, ppid=1, tty="ttys999", rss_gb=9.0, command="python heavy.py")
        notifications = []
        signaled = []
        samples = iter(
            [(3.0, 24.0, 0.0)] * WARN_GRACE_SAMPLES     # warn -> grace elapses -> stop
            + [(0.5, 24.0, 9.0)]                          # still critical -> kill
            + [(3.0, 24.0, 0.0)]                          # a LATER, fresh episode
        )

        def fake_memory():
            try:
                return next(samples)
            except StopIteration:
                raise KeyboardInterrupt

        clock = [0.0]

        def fake_monotonic():
            # Large step so the per-victim notification's 120s throttle never
            # interferes — this test is about the reset surviving a full stop->kill
            # cycle, not about the throttle (that has its own dedicated test).
            clock[0] += 200.0
            return clock[0]

        monkeypatch.setattr(memwatch, "system_memory", fake_memory)
        monkeypatch.setattr(memwatch, "scan_processes", lambda: [proc])
        monkeypatch.setattr(memwatch, "measure_footprints", lambda procs: None)
        monkeypatch.setattr(memwatch, "resolve_ttys", lambda procs: {77: "ttys999"})
        monkeypatch.setattr(memwatch, "notify",
                            lambda title, message, tty="", dry_run=False, pid=None:
                                notifications.append((title, message)))
        monkeypatch.setattr(memwatch, "_signal_pid",
                            lambda pid, sig, verb, dry_run=False: signaled.append(sig))
        monkeypatch.setattr(memwatch, "_other_daemon_pids", lambda: [])
        monkeypatch.setattr(memwatch.time, "sleep", lambda _s: None)
        monkeypatch.setattr(memwatch.time, "monotonic", fake_monotonic)

        memwatch.run_daemon()

        warn_count = sum(1 for t, _m in notifications
                         if t == "Memory low — your job may be suspended soon")
        assert warn_count == 2, (
            "a fresh low-memory episode after a full stop->kill cycle must warn "
            "again, not skip straight to stop off leftover state"
        )


class TestSignalPath:
    """SIGSTOP/SIGCONT must actually suspend and resume on macOS.

    The watchdog's whole non-destructive claim rests on this: if SIGCONT did not
    resume, "suspend first, kill only if that did not help" would silently be
    "freeze the job forever". Exercised on a sacrificial child this test owns — the
    daemon loop is NOT run here, because `pick_victim` targets the largest process on
    the whole box and would suspend a real agent's job.
    """

    def test_stop_then_continue_round_trip(self):
        import os
        import signal as sig
        import subprocess
        import time

        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            def state() -> str:
                out = subprocess.run(["ps", "-o", "state=", "-p", str(child.pid)],
                                     capture_output=True, text=True).stdout.strip()
                return out[:1]

            # Started = alive and not stopped. 'U' (uninterruptible, e.g. paging in
            # under memory pressure) counts: requiring S/R made this test fail on a
            # loaded box, which is exactly where the watchdog runs.
            def started() -> bool:
                return state() not in ("", "T", "Z")

            deadline = time.monotonic() + 10
            while not started() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert started(), f"child never started (state={state()!r})"

            os.kill(child.pid, sig.SIGSTOP)
            deadline = time.monotonic() + 10
            while state() != "T" and time.monotonic() < deadline:
                time.sleep(0.05)
            assert state() == "T", "SIGSTOP did not suspend the process"

            os.kill(child.pid, sig.SIGCONT)
            deadline = time.monotonic() + 10
            while state() == "T" and time.monotonic() < deadline:
                time.sleep(0.05)
            assert state() != "T", "SIGCONT did not resume the process"
        finally:
            child.kill()
            child.wait(timeout=10)


class TestDryRun:
    """`--daemon --dry-run` must decide exactly as the armed daemon and change NOTHING.

    A daemon that SIGKILLs processes has to earn trust before it is armed, so the dry
    run has to be genuinely inert — including not writing "SUSPENDED your job" into
    another agent's terminal, which would be a lie about something that did not happen.
    """

    def test_signal_is_not_sent(self):
        import subprocess
        import time
        from memguard.memwatch import _signal_pid
        import signal as sig

        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            _signal_pid(child.pid, sig.SIGKILL, "killed", dry_run=True)
            time.sleep(0.5)
            assert child.poll() is None, "dry run actually sent SIGKILL"
        finally:
            child.kill()
            child.wait(timeout=10)

    def test_nothing_is_written_to_the_victims_tty(self, tmp_path, monkeypatch, capsys):
        """The tty write and the desktop notification are both suppressed."""
        from memguard import memwatch

        fake_tty = tmp_path / "ttys999"
        fake_tty.write_text("")
        called = []
        monkeypatch.setattr(memwatch.subprocess, "run",
                            lambda *a, **k: called.append(a) or None)

        memwatch.notify("SUSPENDED", "your job", tty="ttys999", dry_run=True)

        assert called == [], "dry run fired a desktop notification"
        assert fake_tty.read_text() == "", "dry run wrote into the victim's tty"
        assert "DRY RUN" in capsys.readouterr().out


class TestOperatorControl:
    """`--stop` / `--resume` / `--kill` on one window's compute children.

    Uses a REAL child process and sends REAL signals; only the tty attribution is
    stubbed, because a test harness has no controlling terminal to attach a child to.
    """

    @staticmethod
    def _state(pid: int) -> str:
        out = subprocess.run(["ps", "-o", "state=", "-p", str(pid)],
                             capture_output=True, text=True).stdout.strip()
        return out[:1]

    @pytest.fixture
    def child_on_tty(self, monkeypatch):
        """A live child, presented to memwatch as if it were owned by ttys999."""
        from memguard import memwatch

        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 10
        while self._state(proc.pid) not in ("S", "R") and time.monotonic() < deadline:
            time.sleep(0.05)

        def fake_scan():
            state = subprocess.run(["ps", "-o", "state=", "-p", str(proc.pid)],
                                   capture_output=True, text=True).stdout.strip()
            return [ProcInfo(pid=proc.pid, ppid=1, tty="ttys999", rss_gb=1.0,
                             command=f"{sys.executable} heavy_job.py", state=state)]

        monkeypatch.setattr(memwatch, "scan_processes", fake_scan)
        monkeypatch.setattr(memwatch, "resolve_ttys", lambda procs: {proc.pid: "ttys999"})
        try:
            yield proc
        finally:
            proc.kill()
            proc.wait(timeout=10)

    def test_stop_then_resume_round_trip(self, child_on_tty):
        """The reversible pair — the whole reason to prefer stop over kill."""
        from memguard.memwatch import run_action

        assert run_action("ttys999", "stop") == 0
        deadline = time.monotonic() + 10
        while self._state(child_on_tty.pid) != "T" and time.monotonic() < deadline:
            time.sleep(0.05)
        assert self._state(child_on_tty.pid) == "T", "--stop did not suspend"

        assert run_action("ttys999", "resume") == 0
        deadline = time.monotonic() + 10
        while self._state(child_on_tty.pid) == "T" and time.monotonic() < deadline:
            time.sleep(0.05)
        assert self._state(child_on_tty.pid) != "T", "--resume did not resume"

    def test_accepts_dev_prefixed_tty(self, child_on_tty):
        """`/dev/ttys999` and `ttys999` must behave identically — ps prints both forms."""
        from memguard.memwatch import run_action
        assert run_action("/dev/ttys999", "stop") == 0
        assert run_action("/dev/ttys999", "resume") == 0

    def test_resume_reports_when_nothing_is_suspended(self, child_on_tty):
        from memguard.memwatch import run_action
        assert run_action("ttys999", "resume") == 1

    def test_unknown_tty_is_reported_not_silently_ignored(self):
        from memguard.memwatch import run_action
        assert run_action("ttys000nonexistent", "stop") == 1

    def test_the_claude_session_is_spared(self, monkeypatch):
        """Stopping a window must never freeze the agent that owns it."""
        from memguard import memwatch
        monkeypatch.setattr(memwatch, "scan_processes", lambda: [
            ProcInfo(pid=1, ppid=1, tty="ttys999", rss_gb=2.0, command="claude"),
        ])
        monkeypatch.setattr(memwatch, "resolve_ttys", lambda procs: {1: "ttys999"})
        assert memwatch.run_action("ttys999", "stop") == 1

    def test_kill_detached_only_targets_project_processes_not_os_daemons(self, monkeypatch):
        """`--kill detached` must select the SAME set the report shows under that
        name — project background jobs with no owning terminal — never a tty-less
        OS/GUI daemon such as WindowServer.
        """
        from memguard import memwatch

        repo_job = ProcInfo(pid=10, ppid=999, tty="", rss_gb=1.0,
                            command="/Users/x/code/myproject/.venv/bin/python sweep.py")
        os_daemon = ProcInfo(pid=20, ppid=999, tty="", rss_gb=3.0,
                             command="/System/Library/.../WindowServer")
        monkeypatch.setattr(memwatch, "scan_processes", lambda: [repo_job, os_daemon])
        monkeypatch.setattr(memwatch, "resolve_ttys", lambda procs: {10: "", 20: ""})
        killed = []
        monkeypatch.setattr(memwatch, "_signal_pid",
                            lambda pid, sig, verb, dry_run=False: killed.append(pid))

        assert memwatch.run_action("detached", "kill") == 0
        assert killed == [10], "must kill only the project-owned detached process"

    def test_kill_detached_spares_an_exempt_service(self, monkeypatch):
        """A tty-less, project-owned service is VISIBLE in the detached bucket, but
        visible must not mean targetable when the exemption file names it.
        """
        from memguard import memwatch

        service = ProcInfo(
            pid=30, ppid=999, tty="", rss_gb=0.5,
            command="/Users/x/code/myproject/.venv/bin/python3 protected_service.py")
        monkeypatch.setattr(memwatch, "scan_processes", lambda: [service])
        monkeypatch.setattr(memwatch, "resolve_ttys", lambda procs: {30: ""})
        killed = []
        monkeypatch.setattr(memwatch, "_signal_pid",
                            lambda pid, sig, verb, dry_run=False: killed.append(pid))

        assert memwatch.run_action("detached", "kill") == 1, (
            "an exempt service in the detached bucket must stay untargetable"
        )
        assert killed == []


class TestSuspendedVisibility:
    """A suspended job still holds its memory, so it must never look idle."""

    def test_window_reports_suspended_gb(self):
        win = WindowUsage(tty="ttys001", title="x", procs=[
            proc(1, rss_gb=3.0, command="python heavy.py"),
            proc(2, rss_gb=2.0, command="python frozen.py"),
        ])
        win.procs[1].state = "T"
        assert win.suspended_gb == pytest.approx(2.0)

    def test_running_window_reports_none_suspended(self):
        win = WindowUsage(tty="ttys001", title="x",
                          procs=[proc(1, rss_gb=3.0, command="python heavy.py")])
        assert win.suspended_gb == 0.0

    def test_report_names_stopped_windows_and_how_to_resume(self, monkeypatch):
        """Freezing something and forgetting it is the failure mode this prevents."""
        from memguard import memwatch
        frozen = ProcInfo(pid=1, ppid=1, tty="ttys777", rss_gb=3.0,
                          command="python heavy.py", state="T")
        monkeypatch.setattr(memwatch, "scan_processes", lambda: [frozen])
        monkeypatch.setattr(memwatch, "measure_footprints", lambda procs: None)
        monkeypatch.setattr(memwatch, "terminal_windows", lambda: {"ttys777": "some agent"})
        monkeypatch.setattr(memwatch, "system_memory", lambda: (8.0, 24.0, 0.0))

        report = memwatch.render_report()
        assert "STOPPED" in report
        assert "ttys777" in report
        assert "--resume" in report


class TestDaemonShutdown:
    """Stopping the watchdog must never leave the box worse than never running it."""

    @staticmethod
    def _state(pid: int) -> str:
        return subprocess.run(["ps", "-o", "state=", "-p", str(pid)],
                              capture_output=True, text=True).stdout.strip()[:1]

    def test_shutdown_resumes_a_suspended_victim(self):
        """The defect this guards: a SIGSTOPped process cannot un-suspend itself.

        If the watchdog exits while holding one — Ctrl-C, SIGTERM from
        `launchctl unload`, a crash — that job stays frozen FOREVER, still holding all
        its memory, with nobody left to resume it. Exercised with real signals on a real
        child, driving the daemon's actual `finally` block via run_daemon.
        """
        from memguard import memwatch

        victim = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + 10
            while self._state(victim.pid) not in ("S", "R") and time.monotonic() < deadline:
                time.sleep(0.05)

            # Drive run_daemon through the full warn -> grace period -> stop sequence
            # via the GRACE path specifically (3.0 GB free is within the grace band,
            # above KILL_FREE_GB — NOT the severity bypass, which would stop on
            # sample 1 with no warning and never exercise this guard at all: an
            # earlier version of this test used
            # 1.0 GB free and the severity bypass silently made the test pass without
            # ever driving a suspend-then-resume cycle). WARN_GRACE_SAMPLES identical
            # low samples land 'stop' on the last one; the next sample then raises
            # KeyboardInterrupt the way Ctrl-C/SIGTERM would, while the victim is
            # still suspended.
            samples = iter([(3.0, 24.0, 0.0)] * WARN_GRACE_SAMPLES)

            def fake_memory():
                try:
                    return next(samples)
                except StopIteration:
                    raise KeyboardInterrupt

            monkey = pytest.MonkeyPatch()
            monkey.setattr(memwatch, "system_memory", fake_memory)
            monkey.setattr(memwatch, "scan_processes", lambda: [
                ProcInfo(pid=victim.pid, ppid=1, tty="ttys999", rss_gb=9.0,
                         command=f"{sys.executable} heavy_job.py"),
            ])
            monkey.setattr(memwatch, "measure_footprints", lambda procs: None)
            monkey.setattr(memwatch, "resolve_ttys", lambda procs: {victim.pid: "ttys999"})
            monkey.setattr(memwatch, "notify", lambda *a, **k: None)
            monkey.setattr(memwatch, "_other_daemon_pids", lambda: [])
            monkey.setattr(memwatch.time, "sleep", lambda _s: None)
            try:
                memwatch.run_daemon()
            finally:
                monkey.undo()

            deadline = time.monotonic() + 10
            while self._state(victim.pid) == "T" and time.monotonic() < deadline:
                time.sleep(0.05)
            assert self._state(victim.pid) != "T", (
                "the watchdog exited while holding this process suspended — it is "
                "stranded frozen, holding all its memory, with nothing left to resume it"
            )
        finally:
            victim.kill()
            victim.wait(timeout=10)

    def test_refuses_to_start_alongside_another_watchdog(self, monkeypatch):
        """Two watchdogs both pick victims and both escalate to SIGKILL."""
        from memguard import memwatch
        monkeypatch.setattr(memwatch, "_other_daemon_pids", lambda: [4242])
        assert memwatch.run_daemon() == 2

    @pytest.mark.parametrize("command, is_daemon", [
        ("/Users/x/.cargo/bin/uv run --no-sync memguard/memwatch.py --daemon", True),
        ("/Users/x/memguard/.venv/bin/python memguard/memwatch.py --daemon", True),
        ("python3.12 /Users/x/Library/Application Support/memguard/memguard/memwatch.py"
         " --daemon --dry-run", True),
        ("/bin/zsh -c pgrep -f \"memwatch.py --daemon\"", False),
        ("grep memwatch.py --daemon log.txt", False),
        ("vim memguard/memwatch.py --daemon", False),
        ("uv run memguard/memwatch.py --watch", False),
    ])
    def test_only_a_running_watchdog_counts_as_one(self, command, is_daemon):
        """2026-11-12: shell commands that merely mentioned the watchdog kept it from
        starting for a minute."""
        from memguard.memwatch import _is_daemon_command
        assert _is_daemon_command(command) is is_daemon

    def test_dry_run_may_run_alongside_an_armed_watchdog(self, monkeypatch):
        """Dry run sends no signals, so it cannot conflict — observing is always safe."""
        from memguard import memwatch
        monkeypatch.setattr(memwatch, "_other_daemon_pids", lambda: [4242])
        monkeypatch.setattr(memwatch, "system_memory",
                            lambda: (_ for _ in ()).throw(KeyboardInterrupt))
        assert memwatch.run_daemon(dry_run=True) == 0


class TestVictimSelection:
    def test_picks_the_largest_targetable_process(self):
        procs = [
            proc(1, rss_gb=9.0, command="python protected_service.py"),   # exempt
            proc(2, rss_gb=5.0, command="python measure_variants.py"),
            proc(3, rss_gb=1.0, command="python small.py"),
        ]
        assert pick_victim(procs).pid == 2

    def test_returns_none_when_only_exempt_processes_are_heavy(self):
        """Never sacrifice an exempt service to save the box."""
        procs = [proc(1, rss_gb=9.0, command="python protected_service.py")]
        assert pick_victim(procs) is None


class TestSingleDaemonCheck:
    """Two watchdogs both pick victims and both escalate — but the check must not
    mistake THIS daemon's own launcher (`uv run ... memwatch.py --daemon`, the parent
    of the python process under launchd) for a second watchdog. It did: the launchd
    job refused to start, every 30 s, and never protected anything."""

    PS = (
        "  100     1 /usr/local/bin/uv run --no-sync memguard/memwatch.py --daemon\n"
        "  101   100 /x/.venv/bin/python3 memguard/memwatch.py --daemon\n"
        "  200     1 /usr/local/bin/uv run memguard/memwatch.py --daemon\n"
        "  201   200 /x/.venv/bin/python3 memguard/memwatch.py --daemon\n"
        "  300     1 python other.py\n"
    )

    def _patch(self, monkeypatch, mine):
        from memguard import memwatch

        class Result:
            stdout = self.PS
        monkeypatch.setattr(memwatch.subprocess, "run", lambda *a, **k: Result())
        monkeypatch.setattr(memwatch.os, "getpid", lambda: mine)
        return memwatch

    def test_own_launcher_is_not_another_daemon(self, monkeypatch):
        memwatch = self._patch(monkeypatch, mine=101)
        assert 100 not in memwatch._other_daemon_pids()

    def test_a_genuinely_separate_daemon_is_still_detected(self, monkeypatch):
        memwatch = self._patch(monkeypatch, mine=101)
        assert sorted(memwatch._other_daemon_pids()) == [200, 201]


class TestFolderExemptions:
    """`dir:` lines: real processes, because the folder is read from the live process."""

    @pytest.fixture
    def folders(self, tmp_path, monkeypatch):
        from memguard import memwatch
        ignored, elsewhere = tmp_path / "ignored", tmp_path / "elsewhere"
        (ignored / "sub").mkdir(parents=True)
        elsewhere.mkdir()
        monkeypatch.setattr(memwatch, "EXEMPT_DIRS", (os.path.realpath(ignored),))
        children = []

        def start(cwd, env_extra=None):
            # A Python child, not /bin/sleep: macOS hides the environment of its own
            # system binaries, while an agent's jobs (python, node, ...) are readable.
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                     cwd=cwd,
                                     env={**os.environ, **(env_extra or {})})
            children.append(child)
            time.sleep(0.3)          # let exec finish so the environment is readable
            return child
        yield ignored, elsewhere, start
        for child in children:
            child.kill()
            child.wait()

    def _job(self, pid):
        return ProcInfo(pid=pid, ppid=1, tty="", rss_gb=5.0, command="python job.py")

    def test_a_job_running_inside_the_folder_is_never_targeted(self, folders):
        ignored, _, start = folders
        job = self._job(start(ignored / "sub", {"CLAUDE_PID": ""}).pid)
        assert job.is_exempt and not job.is_targetable

    def test_a_job_whose_agent_works_in_the_folder_is_never_targeted(self, folders):
        """The job changed directory, but its agent was started in the folder."""
        ignored, elsewhere, start = folders
        agent = start(ignored)
        job = self._job(start(elsewhere, {"CLAUDE_PID": str(agent.pid)}).pid)
        assert job.is_exempt and not job.is_targetable

    def test_an_unrelated_job_stays_targetable(self, folders):
        _, elsewhere, start = folders
        job = self._job(start(elsewhere, {"CLAUDE_PID": ""}).pid)
        assert not job.is_exempt and job.is_targetable

    def test_an_agent_younger_than_the_job_is_a_recycled_pid(self, folders):
        ignored, elsewhere, start = folders
        job_proc = start(elsewhere, {"CLAUDE_PID": "PLACEHOLDER"})
        agent = start(ignored)
        from memguard import _inbox
        real = _inbox.env_var_of
        try:
            _inbox.env_var_of = lambda pid, name: (str(agent.pid) if pid == job_proc.pid
                                                   else real(pid, name))
            assert not self._job(job_proc.pid).is_exempt
        finally:
            _inbox.env_var_of = real


class TestFastFall:
    """2026-11-12 panic: a job was warned at 8.0 GB with 2.8 GB free and given a 27 s
    grace; it reached 13.9 GB and the Mac froze 20 s after the warning. A grace period
    is granted only if the machine can survive it."""

    @staticmethod
    def _samples(*free):
        return [(f, 24.0, 0.0) for f in free]

    def test_memory_falling_fast_is_suspended_without_a_grace_period(self, monkeypatch):
        import signal as sig_mod
        notifications, signaled = TestWarnBeforeStop._drive(
            monkeypatch, self._samples(5.5, 4.6, 3.7, 2.8))
        assert signaled and signaled[0] == (77, sig_mod.SIGSTOP)
        assert not any("suspended soon" in t for t, _ in notifications), (
            "falling ~0.3 GB/s toward the cliff: a grace period would outlast the Mac")

    def test_a_normal_bounce_still_gets_the_grace_period(self, monkeypatch):
        """Single-sample swings of ~2 GB are routine on a busy box (see the log); the
        fall rate is averaged over FALL_WINDOW_SAMPLES readings, so one swing after a
        steady stretch is not a fast fall."""
        from memguard.memwatch import FALL_WINDOW_SAMPLES
        steady = [5.9, 5.8] * FALL_WINDOW_SAMPLES
        notifications, signaled = TestWarnBeforeStop._drive(
            monkeypatch, self._samples(*steady[:FALL_WINDOW_SAMPLES], 3.4))
        assert signaled == []
        assert any("suspended soon" in t for t, _ in notifications)

    def test_the_job_is_suspended_before_anyone_is_told(self, monkeypatch):
        """Telling (osascript, terminal lookup) costs time the machine may not have."""
        import signal as sig_mod
        events = []
        TestWarnBeforeStop._drive(monkeypatch, self._samples(1.5), events=events)
        assert events[0] == ("signal", sig_mod.SIGSTOP)
        assert events[1][0] == "notify"


class TestDiskAlert:
    def test_low_disk_is_reported_once_per_hour(self, monkeypatch):
        from memguard import memwatch
        monkeypatch.setattr(memwatch, "DISK_WARN_GB", 1e9)
        notifications, _ = TestWarnBeforeStop._drive(
            monkeypatch, [(10.0, 24.0, 0.0)] * 3, clock_step=SAMPLE_SECONDS)
        assert [t for t, _ in notifications].count(
            "Disk almost full: memory protection weakened") == 1


class TestSwapRoom:
    """Both 2026 panics reported "LOW swap space": with no room to swap, free memory
    overstates what is really available."""

    def test_plenty_of_swap_room_changes_nothing(self, monkeypatch):
        from memguard import memwatch
        monkeypatch.setattr(memwatch, "SWAP_ROOM_GB", 8.0)
        assert memwatch.usable_free_gb(4.0, swap_room=50.0) == 4.0

    def test_a_swap_room_shortfall_lowers_usable_memory(self, monkeypatch):
        from memguard import memwatch
        monkeypatch.setattr(memwatch, "SWAP_ROOM_GB", 8.0)
        assert memwatch.usable_free_gb(4.0, swap_room=5.0) == 1.0

    def test_the_daemon_acts_on_usable_memory(self, monkeypatch):
        """4.0 GB of RAM free is no emergency, but with 5 GB of swap room left it
        counts as 1.0 GB: the largest job is paused at once."""
        import signal as sig_mod
        from memguard import memwatch
        monkeypatch.setattr(memwatch, "SWAP_ROOM_GB", 8.0)
        monkeypatch.setattr(memwatch, "swap_room_gb", lambda: 5.0)
        _, signaled = TestWarnBeforeStop._drive(monkeypatch, [(4.0, 24.0, 0.0)])
        assert signaled and signaled[0] == (77, sig_mod.SIGSTOP)
