#!/usr/bin/env python3
"""Per-terminal memory monitor and OOM watchdog for a shared multi-agent machine.

Two modes, one accounting core:

    uv run memguard/memwatch.py                 # monitor: which window is eating the box
    uv run memguard/memwatch.py --watch         # ... refreshed continuously
    uv run memguard/memwatch.py --daemon        # watchdog: SIGSTOP / SIGCONT / SIGKILL
                                                #   stop it: Ctrl-C, or SIGTERM.
                                                #   under launchd: `launchctl unload` —
                                                #   `stop` alone is undone by KeepAlive.
    uv run memguard/memwatch.py --stop ttys025   # SUSPEND one window's compute children
    uv run memguard/memwatch.py --resume ttys025 # ... and let them continue
    uv run memguard/memwatch.py --kill ttys025   # kill them (irreversible)

WHY a watchdog exists at all. Many coding agents share one machine, each launching
compute jobs (model inference, structure prediction, data processing). No single job
needs to be pathological for the box to die: several jobs each legitimately holding a
few GB, or one job accumulating across many bounded steps, exhaust physical memory and
swap together. macOS then stalls in the compressor until the kernel watchdog panics
the machine. This daemon is the backstop for that aggregate pressure.

WHY suspend before kill. SIGSTOP does not release resident memory, so suspension alone
cannot resolve pressure and the SIGKILL escalation is genuinely required. What
suspension buys is that it halts GROWTH immediately and reversibly: the common case is
N agents climbing toward a peak together, and suspending the largest lets the rest
finish and release, after which it resumes. Concurrent big jobs get time-sliced instead
of collectively crashing the machine, and the destructive action is reserved for the
case where suspension demonstrably did not help.

WHY warn before suspend. The FIRST time a specific victim is about to be suspended for
SHALLOW, stable pressure, it gets a targeted message in its own terminal (naming it and
its size) and up to WARN_GRACE_SAMPLES samples (~(WARN_GRACE_SAMPLES-1)*SAMPLE_SECONDS
s — the warning's own sample counts toward the countdown) to react before the SIGSTOP
lands. The agent driving that job can often bound it in the time a SIGSTOP would
otherwise just freeze it for.

This grace period is a courtesy, not a delay applied everywhere. It is granted only if
the machine can survive it: when free memory is FALLING so fast that it would pass
KILL_FREE_GB within FALL_HORIZON_SAMPLES samples, the largest job is suspended at once
(2026-11-12: a job grew from 8.0 to 13.9 GB within the 27 s grace it was given at
2.8 GB free, and the Mac froze 20 s after the warning). And once free memory is
past KILL_FREE_GB — a genuine OOM cliff, not shallow pressure — the daemon suspends the
largest process IMMEDIATELY (action 'stop_severe', distinct from the grace-elapsed
'stop') with no warning and no wait. 'stop' (grace elapsed) always targets the SPECIFIC
process that was warned; 'stop_severe' always picks fresh, even if a warning happens to
be pending from an earlier, shallower episode, so a cliff never suspends a smaller
already-warned process instead of the actual largest consumer.

MEASUREMENT. Memory is judged by COMMITTED footprint (`phys_footprint`), not RSS. RSS is
wrong in both directions on macOS: it over-states mmap-heavy jobs, and it does not see
Metal/MPS (GPU) allocations at all — a torch/MPS job can hold 20 GB while `ps` reports
a few MB of RSS. Every process is measured via `proc_pid_rusage` (microseconds each,
unprivileged for the user's own processes; see `phys_footprint_gb`). Every number
reported carries its unit (GB).

Processes listed in the exemption file (`EXEMPT_FILE`) are never suspended or killed.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

from memguard import _inbox

GB = 1024.0 ** 3

# ---------------------------------------------------------------------------
# Thresholds (GB). One block, env-overridable.
# ---------------------------------------------------------------------------
WARN_FREE_GB = float(os.environ.get("MEMWATCH_WARN_FREE_GB", 6.0))
STOP_FREE_GB = float(os.environ.get("MEMWATCH_STOP_FREE_GB", 3.5))
KILL_FREE_GB = float(os.environ.get("MEMWATCH_KILL_FREE_GB", 2.0))
RESUME_FREE_GB = float(os.environ.get("MEMWATCH_RESUME_FREE_GB", 6.0))
#: A window at or above this is marked "!" in the report as unusually heavy.
#:
#: DISPLAY ONLY. The daemon never suspends on a per-job size — it suspends the LARGEST
#: job when BOX free memory falls under STOP_FREE_GB. Raising this does not make a big
#: job safe from suspension; it changes which jobs the report calls out as anomalous.
HEAVY_WINDOW_GB = float(os.environ.get("MEMWATCH_HEAVY_WINDOW_GB", 6.0))
RESUME_SAMPLES = int(os.environ.get("MEMWATCH_RESUME_SAMPLES", 9))
#: One reading (free memory, swap room, the decision) costs ~0.01 ms; the full process
#: scan runs only when acting. Every second, because a job growing 0.3 GB/s (seen on
#: 2026-11-12) moves ~1 GB between readings 3 s apart. The sample counts below are set
#: for this interval: grace 27 s, resume after 9 s, fall rate over 15 s, look-ahead 6 s.
SAMPLE_SECONDS = float(os.environ.get("MEMWATCH_SAMPLE_SECONDS", 1.0))

#: Consecutive low-memory samples a warned victim gets to react before it is actually
#: SIGSTOPped — see "WHY warn before suspend" above. Same sample-count convention as
#: RESUME_SAMPLES (a duration in samples, not wall time, so it stays correct if
#: SAMPLE_SECONDS is overridden).
WARN_GRACE_SAMPLES = int(os.environ.get("MEMWATCH_WARN_GRACE_SAMPLES", 28))

#: The fall rate of free memory is measured over this many recent samples (smoothing the
#: +-2 GB single-sample bounces that are normal on a busy box) and projected this many
#: samples ahead: margin for the daemon itself being slowed down by memory pressure.
FALL_WINDOW_SAMPLES = int(os.environ.get("MEMWATCH_FALL_WINDOW_SAMPLES", 15))
FALL_HORIZON_SAMPLES = int(os.environ.get("MEMWATCH_FALL_HORIZON_SAMPLES", 6))

#: macOS swaps to the startup disk. With too little free disk, swap cannot grow, memory
#: that looks reclaimable is not, and the machine can freeze before free memory reads
#: low (both 2026 panics reported "LOW swap space"; the disk had ~9 GB free). Below this
#: the person is told, once an hour.
DISK_WARN_GB = float(os.environ.get("MEMWATCH_DISK_WARN_GB", 20.0))
SWAP_DIR = "/private/var/vm"

#: Swap room the thresholds assume. Free memory includes inactive pages that macOS
#: frees by compressing them and swapping the result out; with no swap room they stay
#: in RAM, so free memory overstates what is really available. When swap room (space
#: left in the swap files plus free disk) is below this, every decision counts free
#: memory as lower by the shortfall: memguard warns, pauses and kills earlier, and
#: resumes later. A judgment, not a measured constant: room for one more burst like
#: the 6 GB in 20 s seen on 2026-11-12, plus margin.
SWAP_ROOM_GB = float(os.environ.get("MEMWATCH_SWAP_ROOM_GB", 8.0))

#: Guidance quoted verbatim in the pre-stop warning. memwatch only observes a pid's
#: aggregate footprint, never what it is doing, so the advice is necessarily generic.
STREAMING_LOADER_GUIDANCE = (
    "stream/chunk data instead of loading it whole, lower the batch size or the "
    "number of parallel workers, release tensors (and torch.mps.empty_cache()) "
    "between items, or wait for the box to drain."
)

#: What is NEVER suspended or killed, one entry per line (`#` comments allowed): a
#: substring of a command line, or `dir: <folder>` for every job that runs inside that
#: folder or was started by an agent running inside it. Machine-local, deliberately not part of this repository: it names
#: whatever services on THIS box must not stall. A missing file means "no exemptions"
#: (announced at daemon start); an unreadable one raises, because silently dropping
#: exemptions would make protected services targetable.
EXEMPT_FILE = os.environ.get(
    "MEMWATCH_EXEMPT_FILE", os.path.expanduser("~/.config/memguard/exempt")
)


def load_exempt(path: str = EXEMPT_FILE) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(command-line substrings, folders) from the exemption file."""
    if not os.path.exists(path):
        return (), ()
    with open(path, encoding="utf-8") as handle:
        lines = [line.split("#", 1)[0].strip() for line in handle]
    entries = [line for line in lines if line]
    dirs = tuple(os.path.realpath(os.path.expanduser(e[4:].strip()))
                 for e in entries if e.startswith("dir:"))
    return tuple(e for e in entries if not e.startswith("dir:")), dirs


EXEMPT_PATTERNS, EXEMPT_DIRS = load_exempt()


def _is_within(path: str, folder: str) -> bool:
    return path == folder or path.startswith(folder.rstrip("/") + "/")


def work_dirs(pid: int) -> tuple[str, ...]:
    """The folder `pid` runs in and, if an AI agent started it, the agent's folder.

    Claude Code exports `CLAUDE_PID` to every job it starts. The agent keeps the folder
    it was started in even when its job changes directory, so this attributes a job to
    its agent's project. The agent counts only if it is still alive and older than the
    job, so a recycled pid is never mistaken for it.
    """
    import psutil
    dirs = []
    try:
        job = psutil.Process(pid)
        dirs.append(job.cwd())
        agent_pid = _inbox.env_var_of(pid, _inbox.AGENT_PID_ENV_VAR)
        if agent_pid and agent_pid.isdigit():
            agent = psutil.Process(int(agent_pid))
            if agent.create_time() <= job.create_time():
                dirs.append(agent.cwd())
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        pass
    return tuple(dirs)

#: The `claude` CLI itself is never the target: killing an agent instead of its runaway
#: job destroys the session's context while LEAVING the memory consumer running, which
#: is strictly worse than doing nothing. Its heavy children remain fair game.
NEVER_TARGET_PATTERNS = ("claude", "memwatch.py", "caffeinate")


# ---------------------------------------------------------------------------
# Process scan
# ---------------------------------------------------------------------------
@dataclass
class ProcInfo:
    pid: int
    ppid: int
    tty: str            # "ttys025", or "" when detached
    rss_gb: float
    command: str
    state: str = ""     # ps state: "T" = suspended, "R"/"S" = running/sleeping
    footprint_gb: Optional[float] = None
    _in_exempt_dir: Optional[bool] = field(default=None, repr=False, compare=False)

    @property
    def is_suspended(self) -> bool:
        """True when this process is SIGSTOPped.

        Surfaced everywhere, because a suspended job still holds all its memory and
        looks idle: without this you stop something, forget, and leave it frozen.
        """
        return self.state.startswith("T")

    @property
    def best_gb(self) -> float:
        """Committed footprint when measured, else RSS (which over-states)."""
        return self.footprint_gb if self.footprint_gb is not None else self.rss_gb

    @property
    def is_exempt(self) -> bool:
        if any(p in self.command for p in EXEMPT_PATTERNS):
            return True
        if not EXEMPT_DIRS:
            return False
        if self._in_exempt_dir is None:     # looked up once per scan, not per call
            self._in_exempt_dir = any(_is_within(d, folder) for d in work_dirs(self.pid)
                                      for folder in EXEMPT_DIRS)
        return self._in_exempt_dir

    @property
    def is_targetable(self) -> bool:
        """May the watchdog suspend or kill this process?

        Matched on the EXECUTABLE (first token) rather than the whole command line, so
        an agent's `uv run python heavy.py` stays targetable while the `claude` process
        that launched it does not.
        """
        if self.is_exempt:
            return False
        executable = self.command.split()[0] if self.command else ""
        return not any(p in executable for p in NEVER_TARGET_PATTERNS)

    @property
    def short_command(self) -> str:
        """The script name if this is a python process, else the bare executable."""
        parts = self.command.split()
        for part in parts:
            if part.endswith(".py"):
                return os.path.basename(part)
        return os.path.basename(parts[0]) if parts else "?"


def scan_processes() -> list[ProcInfo]:
    """All processes with pid, ppid, controlling tty, RSS (GB) and command."""
    out = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,tty=,rss=,state=,command="],
        capture_output=True, text=True, timeout=30,
    ).stdout
    procs: list[ProcInfo] = []
    for line in out.splitlines():
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        pid, ppid, tty, rss_kb, state, command = parts
        try:
            procs.append(ProcInfo(
                pid=int(pid), ppid=int(ppid),
                tty="" if tty in ("??", "-") else tty,
                rss_gb=int(rss_kb) * 1024 / GB,
                state=state,
                command=command,
            ))
        except ValueError:
            continue
    return procs


def resolve_ttys(procs: Iterable[ProcInfo]) -> dict[int, str]:
    """pid -> owning tty, walking the parent chain for detached processes.

    A `nohup`/background job shows tty "??" but its memory still belongs to the session
    that launched it, so it is charged to the nearest ancestor that has a tty.
    """
    by_pid = {p.pid: p for p in procs}
    resolved: dict[int, str] = {}

    for proc in procs:
        seen: set[int] = set()
        cur: Optional[ProcInfo] = proc
        while cur is not None and cur.pid not in seen:
            if cur.tty:
                resolved[proc.pid] = cur.tty
                break
            seen.add(cur.pid)
            cur = by_pid.get(cur.ppid)
        else:
            # Walked to init (or hit a cycle) without finding a tty — genuinely detached.
            resolved[proc.pid] = ""
    return resolved


class _RusageInfoV0(ctypes.Structure):
    """`struct rusage_info_v0` from <sys/resource.h> (RUSAGE_INFO_V0 = 0)."""
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64) for name in (
            "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups",
            "ri_pageins", "ri_wired_size", "ri_resident_size", "ri_phys_footprint",
            "ri_proc_start_abstime", "ri_proc_exit_abstime",
        )
    ]


_LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib")
_LIBC = ctypes.CDLL(None, use_errno=True)


def phys_footprint_gb(pid: int) -> Optional[float]:
    """macOS committed footprint in GB via `proc_pid_rusage`, or None if unreadable.

    The same number `footprint -p` reports, but ~5 us instead of ~86 ms, so EVERY
    process is measured rather than only the top-N by RSS. That matters: Metal/MPS
    memory (torch, boltz) is counted in phys_footprint but NOT in RSS — measured
    2026-11-06: a 2 GB MPS tensor showed 0.10 GB RSS / 2.2 GB footprint, and a boltz
    run held 20.7 GB footprint at 7.8 MB RSS. Ranking by RSS made such jobs invisible.
    None for processes of other users (EPERM) — those fall back to RSS.
    """
    info = _RusageInfoV0()
    if _LIBPROC.proc_pid_rusage(int(pid), 0, ctypes.byref(info)) != 0:
        return None
    return info.ri_phys_footprint / GB


# ---------------------------------------------------------------------------
# Terminal window titles
# ---------------------------------------------------------------------------
_WINDOW_SCRIPT = r'''
set out to ""
tell application "Terminal"
  repeat with i from 1 to (count windows)
    set w to window i
    try
      set t to (tty of tab 1 of w)
      set out to out & t & "\t" & (name of w) & linefeed
    end try
  end repeat
end tell
return out
'''


def terminal_windows() -> dict[str, str]:
    """tty -> window title, e.g. "ttys025" -> "Fix memory issues with parallel agents".

    Terminal.app exposes no per-tab title (`name of tab` raises -1728), so a multi-tab
    window reports its SELECTED tab's title. On this machine every window holds one tab,
    making the map exact; the tty and command columns are exact regardless.
    """
    try:
        out = subprocess.run(
            ["osascript", "-e", _WINDOW_SCRIPT],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    mapping: dict[str, str] = {}
    for line in out.splitlines():
        if "\t" not in line:
            continue
        tty, title = line.split("\t", 1)
        mapping[tty.strip().replace("/dev/", "")] = _clean_title(title.strip())
    return mapping


def _clean_title(title: str) -> str:
    """Strip Terminal.app's decoration down to the task name Claude Code set.

    "myproject — ⠐ Fix memory issues — caffeinate ◂ claude — 87×26"  ->
    "Fix memory issues"
    """
    parts = [p.strip() for p in title.split("—")]
    parts = [p for p in parts if p and not re.fullmatch(r"\d+×\d+", p)]
    if len(parts) > 1:
        parts = parts[1:]                                    # drop the folder name
    parts = [p for p in parts if not p.startswith("-") and "◂" not in p] or parts
    best = max(parts, key=len) if parts else title
    # Strip Claude Code's leading status glyphs (braille spinner, ✳/✶/·) so the task
    # name starts at a stable column and sorts predictably.
    return re.sub(r"^[⠀-⣿✳✶✻✽·∗\s]+", "", best).strip()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
@dataclass
class WindowUsage:
    tty: str
    title: str
    procs: list[ProcInfo] = field(default_factory=list)

    @property
    def total_gb(self) -> float:
        return sum(p.best_gb for p in self.procs)

    @property
    def suspended_gb(self) -> float:
        """GB held by SIGSTOPped children of this window.

        Suspension does not release memory, so a stopped job is still occupying the box
        — it must stay visible or it will sit frozen and forgotten.
        """
        return sum(p.best_gb for p in self.procs if p.is_suspended and p.is_targetable)

    @property
    def top_child(self) -> Optional[ProcInfo]:
        """The window's heaviest compute job, or None when it is merely idle.

        Ignores anything under 0.2 GB so an idle session reports "-" rather than its
        `login` shell, which would read as a finding where there is none.
        """
        targetable = [p for p in self.procs if p.is_targetable and p.best_gb >= 0.2]
        return max(targetable, key=lambda p: p.best_gb) if targetable else None


#: Synthetic tty key for processes whose ppid chain never reaches a controlling
#: terminal — background jobs started via `nohup`/`&` whose launching shell has
#: since exited get reparented to launchd (ppid 1). `resolve_ttys` correctly
#: reports these as "" (genuinely detached), but that must not mean invisible:
#: they still hold real memory, so they are bucketed here instead of dropped.
DETACHED_TTY = "detached"

#: Command-line substrings identifying a detached process as agent/project work
#: (colon-separated, default: anything under ~/code/). Scopes DETACHED_TTY: without
#: it, every OS/GUI daemon with no controlling terminal (WindowServer, Finder, ...)
#: would land in the "agent-owned" total and become a target of `--stop/--kill
#: detached`. KNOWN LIMITATION: this matches the COMMAND LINE, so a detached job
#: invoked with a bare relative path (`nohup python3 heavy.py &`) carries no marker
#: and is excluded from the report. The daemon's victim choice (`pick_victim`) is
#: NOT affected — it considers every targetable process on the box.
REPO_PATH_MARKERS = tuple(
    m for m in os.environ.get("MEMWATCH_PROJECT_MARKERS", os.path.expanduser("~/code/")).split(":") if m
)


def _is_repo_process(command: str) -> bool:
    return any(marker in command for marker in REPO_PATH_MARKERS)


def detached_bucket_key(proc: ProcInfo, resolved_tty: str) -> Optional[str]:
    """The tty this process groups/targets under, or None to exclude it.

    Single source of truth for both build_usage (the report) and run_action
    (--stop/--resume/--kill), so they cannot diverge on which detached
    processes are in scope — see REPO_PATH_MARKERS for why a tty-less process
    is excluded rather than bucketed when it is not project work.
    """
    if resolved_tty:
        return resolved_tty
    return DETACHED_TTY if _is_repo_process(proc.command) else None


def build_usage(procs: list[ProcInfo], titles: dict[str, str]) -> list[WindowUsage]:
    """Group processes onto their owning terminal window, heaviest first.

    Project processes with no owning terminal are grouped under DETACHED_TTY rather
    than dropped; other tty-less processes (OS/GUI daemons) are excluded — see
    detached_bucket_key.
    """
    tty_of = resolve_ttys(procs)
    windows: dict[str, WindowUsage] = {}
    for proc in procs:
        tty = detached_bucket_key(proc, tty_of.get(proc.pid, ""))
        if tty is None:
            continue
        title = titles.get(tty, "") if tty != DETACHED_TTY else \
            "DETACHED background jobs (no owning terminal - nohup/reparented)"
        win = windows.setdefault(tty, WindowUsage(tty=tty, title=title))
        win.procs.append(proc)
    return sorted(windows.values(), key=lambda w: -w.total_gb)


def measure_footprints(procs: list[ProcInfo]) -> None:
    """Set committed footprint on every process, in place (see phys_footprint_gb)."""
    for proc in procs:
        proc.footprint_gb = phys_footprint_gb(proc.pid)


def system_memory() -> tuple[float, float, float]:
    """(free_gb, total_gb, swap_used_gb)."""
    import psutil
    vm = psutil.virtual_memory()
    return vm.available / GB, vm.total / GB, psutil.swap_memory().used / GB


def memguard_disabled_pids(procs: list[ProcInfo]) -> list[int]:
    """Processes running with the guard switched off — the hole, made visible.

    psutil cannot read another process's environment on macOS without root, so this
    falls back to the command line (`MEMGUARD_DISABLED=1 uv run ...`). It reports what
    it can see and never claims the absence of a result means the absence of the switch.
    """
    hits = []
    for proc in procs:
        if "MEMGUARD_DISABLED" in proc.command:
            hits.append(proc.pid)
    return hits


# ---------------------------------------------------------------------------
# Watchdog decision — pure, so it is testable without sending signals
# ---------------------------------------------------------------------------
@dataclass
class WatchState:
    stopped_pid: Optional[int] = None
    recovered_samples: int = 0
    #: The victim already warned about an impending suspend, and how many consecutive
    #: low-memory samples have elapsed since — see decide_action's "warn_victim" case.
    warned_pid: Optional[int] = None
    warned_samples: int = 0
    #: Consecutive samples at/above RESUME_FREE_GB since the warning — mirrors
    #: recovered_samples/RESUME_SAMPLES exactly, so clearing a pending warning
    #: requires the SAME sustained recovery as resuming an already-stopped victim,
    #: not a single tick back above STOP_FREE_GB (a zero-hysteresis reset livelocks:
    #: memory oscillating around STOP_FREE_GB re-warns forever and never stops).
    warn_recovered_samples: int = 0
    #: The last FALL_WINDOW_SAMPLES free-memory readings, oldest first.
    recent_free: list[float] = field(default_factory=list)


def fall_rate_gb_s(free_gb: float, state: WatchState) -> float:
    """How fast free memory is falling (GB/s, positive = falling), over the window."""
    if not state.recent_free:
        return 0.0
    return (state.recent_free[0] - free_gb) / (len(state.recent_free) * SAMPLE_SECONDS)


def decide_action(free_gb: float, state: WatchState) -> Optional[str]:
    """One of 'warn', 'warn_victim', 'stop', 'stop_severe', 'kill', 'resume', or None.
    No side effects.

    'warn_victim': the first sample crossing STOP_FREE_GB warns the (as-yet-unpicked)
    victim once and waits WARN_GRACE_SAMPLES samples before 'stop'. 'stop' means
    "suspend the SPECIFIC process that was warned" — the caller must honor that, not
    re-pick a fresh victim.

    'stop_severe' is a DISTINCT action: the caller must pick a FRESH victim for it
    (pick_victim(procs)), never a stale state.warned_pid. Once free memory is past
    KILL_FREE_GB, or below STOP_FREE_GB and falling fast enough to pass KILL_FREE_GB
    within FALL_HORIZON_SAMPLES, waiting out a grace period would leave the box
    unprotected during a genuine OOM cliff, so it bypasses warn_victim entirely.

    DELIBERATELY free memory ONLY — swap used does not gate any decision. It is not a
    pressure signal on macOS: tens of GB of swap sit in use from ordinary background
    activity with plenty of memory free. A swap term in the kill escalation killed
    every suspended victim on the next sample, at 3-4 GB free.
    """
    if state.stopped_pid is not None:
        if free_gb >= RESUME_FREE_GB and state.recovered_samples + 1 >= RESUME_SAMPLES:
            return "resume"
        if free_gb < KILL_FREE_GB:
            return "kill"
        return None
    if free_gb < KILL_FREE_GB:
        return "stop_severe"
    if free_gb < STOP_FREE_GB:
        horizon_s = FALL_HORIZON_SAMPLES * SAMPLE_SECONDS
        if free_gb - fall_rate_gb_s(free_gb, state) * horizon_s < KILL_FREE_GB:
            return "stop_severe"            # no time for a grace period
        if state.warned_pid is None:
            return "warn_victim"
        if state.warned_samples + 1 >= WARN_GRACE_SAMPLES:
            return "stop"
        return None
    if free_gb < WARN_FREE_GB:
        return "warn"
    return None


def pick_victim(procs: list[ProcInfo]) -> Optional[ProcInfo]:
    """The largest process we are permitted to suspend, or None."""
    targetable = [p for p in procs if p.is_targetable]
    return max(targetable, key=lambda p: p.best_gb) if targetable else None


# ---------------------------------------------------------------------------
# Notification
# ---------------------------------------------------------------------------
_NOTIFIERS: list[subprocess.Popen] = []


def notify(title: str, message: str, tty: str = "", dry_run: bool = False,
           pid: Optional[int] = None) -> None:
    """Tell everyone who must know: the human and, when `pid` was started by an AI
    agent, the AGENT itself.

    - Desktop notification and the daemon log, for the human.
    - The agent's inbox (`memguard._inbox`), when `pid` carries an agent session id.
      `memguard.agent_hook` delivers it into the agent's context on its next step, so
      the agent can shrink its job before memwatch suspends it, and shows it to the
      person in Claude Code's transcript.
    - Otherwise a line in the job's own terminal, when it has one. Never for an agent's
      job: that terminal is drawn by Claude Code, and raw text written into it lands
      on top of Claude Code's screen.

    Under dry_run this stays on THIS process's stdout and touches nothing else: telling
    another agent its job was suspended when it was not would be a lie, and a dry run
    that writes into other people's terminals is not a dry run.
    """
    if dry_run:
        print(f"{_stamp()} [memguard] DRY RUN: would have notified {tty or 'desktop'} — "
              f"{title}: {message}", flush=True)
        return
    # Started, never waited for: under memory pressure launching osascript can take
    # seconds, and the watchdog must keep sampling meanwhile. Finished ones are reaped
    # on the next call.
    _NOTIFIERS[:] = [n for n in _NOTIFIERS if n.poll() is None]
    try:
        _NOTIFIERS.append(subprocess.Popen(
            ["osascript", "-e",
             f'display notification {message!r} with title {title!r}'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ))
    except OSError:
        pass
    session = _inbox.session_of(pid) if pid is not None else None
    if tty and session is None:
        try:
            with open(f"/dev/{tty}", "w") as handle:
                handle.write(f"\n\033[1;31m[memguard] {title}: {message}\033[0m\n")
        except OSError:
            pass
    if session is not None:
        _inbox.post(session, f"{title}: {message}")
    print(f"{_stamp()} [memguard] {title}: {message}"
          + (f"  -> agent session {session}" if session else ""), flush=True)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def render_report() -> str:
    procs = scan_processes()
    measure_footprints(procs)
    titles = terminal_windows()
    windows = build_usage(procs, titles)
    free_gb, total_gb, swap_gb = system_memory()

    lines = [
        f"FREE {free_gb:.1f} GB / {total_gb:.1f} GB   swap {swap_gb:.1f} GB"
        f"          [SIGSTOP threshold {STOP_FREE_GB:.1f} GB]",
        "",
        f"{'TTY':<9} {'GB':>6}  {'WINDOW TITLE':<40} {'STATE':<9} TOP CHILD",
    ]
    agent_total = 0.0
    for win in windows:
        if win.total_gb < 0.1:
            continue
        child = win.top_child
        child_txt = f"{child.short_command}  {child.best_gb:.1f} GB" if child else "-"
        title = (win.title or "?")[:40]
        mark = "!" if win.total_gb >= HEAVY_WINDOW_GB else " "
        state = f"STOPPED" if win.suspended_gb > 0 else ""
        lines.append(f"{win.tty:<9} {win.total_gb:>6.1f}{mark} {title:<40} "
                     f"{state:<9} {child_txt}")
        agent_total += sum(p.best_gb for p in win.procs if p.is_targetable)

    exempt_gb = sum(p.best_gb for p in procs if p.is_exempt)
    lines += [
        "",
        f"TOTAL agent-owned {agent_total:.1f} GB    "
        f"exempt {exempt_gb:.1f} GB",
        "GB = committed phys_footprint (RSS only where footprint is unreadable).",
    ]

    frozen = [w for w in windows if w.suspended_gb > 0]
    if frozen:
        held = sum(w.suspended_gb for w in frozen)
        lines.append(
            f"\nSUSPENDED: {held:.1f} GB held by stopped jobs on "
            f"{', '.join(w.tty for w in frozen)} — still occupying the box.\n"
            f"  resume:  uv run memguard/memwatch.py --resume <tty>")

    disabled = memguard_disabled_pids(procs)
    if disabled:
        lines.append(
            f"\n!! MEMGUARD_DISABLED seen on pids {disabled} — the load guard is OFF there."
        )
    return "\n".join(lines)


def run_monitor(watch: bool) -> int:
    if not watch:
        print(render_report())
        return 0
    try:
        while True:
            report = render_report()
            print("\033[2J\033[H" + report, flush=True)
            time.sleep(SAMPLE_SECONDS)
    except KeyboardInterrupt:
        return 0


def _is_daemon_command(command: str) -> bool:
    """Is this command line a watchdog (`[uv run] python ... memwatch.py --daemon`)?

    Matched on the arguments, not on substrings: a shell command that merely MENTIONS
    the watchdog (`pgrep -f "memwatch.py --daemon"`, an editor, a grep) once blocked it
    from starting for a minute on 2026-11-12.
    """
    tokens = command.split()
    if not tokens or "--daemon" not in tokens:
        return False
    executable = os.path.basename(tokens[0])
    return ((executable == "uv" or executable.startswith("python"))
            and any(t.endswith("memwatch.py") for t in tokens))


def _other_daemon_pids() -> list[int]:
    """PIDs of OTHER running `memwatch --daemon` processes.

    Two watchdogs are actively dangerous, not merely redundant: each tracks only the
    victim IT suspended, so both can pick victims and both can escalate to SIGKILL,
    and one may resume a process the other is still counting as stopped. Easy to do by
    accident by running it by hand after installing the launchd job.

    This process's own ANCESTORS are excluded: under launchd the daemon runs as
    `uv run ... memwatch.py --daemon`, whose uv parent carries the same command line.
    Counting it made the daemon refuse to start, forever.
    """
    out = subprocess.run(["ps", "-axo", "pid=,ppid=,command="],
                         capture_output=True, text=True, timeout=30).stdout
    parent_of: dict[int, int] = {}
    daemons: list[int] = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid, ppid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        parent_of[pid] = ppid
        if _is_daemon_command(parts[2]):
            daemons.append(pid)

    mine = os.getpid()
    lineage = {mine}
    cur = mine
    while cur in parent_of and parent_of[cur] not in lineage:
        cur = parent_of[cur]
        lineage.add(cur)
    return [pid for pid in daemons if pid not in lineage]


def _stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def disk_free_gb() -> float:
    """Free space on the volume macOS swaps to."""
    st = os.statvfs(SWAP_DIR)
    return st.f_bavail * st.f_frsize / GB


def swap_room_gb() -> float:
    """How much more macOS can swap: space left in the swap files plus free disk."""
    import psutil
    return psutil.swap_memory().free / GB + disk_free_gb()


def usable_free_gb(free_gb: float, swap_room: float) -> float:
    """Free memory as the decisions count it: lowered by any swap-room shortfall."""
    return free_gb - max(0.0, SWAP_ROOM_GB - swap_room)


def kernel_pressure_level() -> int:
    """macOS's own verdict: 1 normal, 2 warning, 4 critical."""
    value = ctypes.c_int(0)
    size = ctypes.c_size_t(ctypes.sizeof(value))
    if _LIBC.sysctlbyname(b"kern.memorystatus_vm_pressure_level", ctypes.byref(value),
                          ctypes.byref(size), None, 0) != 0:
        raise OSError(ctypes.get_errno(), "sysctl kern.memorystatus_vm_pressure_level")
    return value.value


def _exempt_file_mtime() -> Optional[float]:
    try:
        return os.stat(EXEMPT_FILE).st_mtime
    except FileNotFoundError:
        return None


def _announce_exempt() -> None:
    print(f"{_stamp()} [memguard] exempt (never suspended/killed), from {EXEMPT_FILE}: "
          f"{', '.join(EXEMPT_PATTERNS) if EXEMPT_PATTERNS else 'NONE'}"
          f"; folders: {', '.join(EXEMPT_DIRS) if EXEMPT_DIRS else 'NONE'}", flush=True)


def run_daemon(dry_run: bool = False) -> int:
    """Watch memory and act. With dry_run, decide and LOG but send no signal.

    A daemon that SIGKILLs processes has to earn trust before it is armed, so
    `--daemon --dry-run` lets an operator watch a full pressure cycle — which process
    would have been suspended, when it would have resumed — with nothing at risk.
    """
    others = _other_daemon_pids()
    if others and not dry_run:
        print(f"{_stamp()} [memguard] REFUSING to start — another watchdog is already running "
              f"(pid {', '.join(str(p) for p in others)}).\n"
              f"  Two watchdogs both pick victims and both escalate to SIGKILL.\n"
              f"  Stop the other first:  kill {others[0]}\n"
              f"  ...or, if it is the launchd job:  "
              f"launchctl unload ~/Library/LaunchAgents/com.memguard.memwatch.plist",
              file=sys.stderr, flush=True)
        return 2

    state = WatchState()
    mode = "DRY RUN — no signals will be sent" if dry_run else "ARMED"
    print(f"{_stamp()} [memguard] watchdog up [{mode}] (pid {os.getpid()}) — "
          f"warn<{WARN_FREE_GB} stop<{STOP_FREE_GB} kill<{KILL_FREE_GB} GB free, "
          f"{WARN_GRACE_SAMPLES}-sample grace before stop, "
          f"sampling {SAMPLE_SECONDS}s.  Stop with Ctrl-C or SIGTERM.", flush=True)
    exempt_mtime = _exempt_file_mtime()
    _announce_exempt()

    # launchd's `unload` sends SIGTERM; without this it would kill the watchdog
    # outright and any process it had suspended would stay frozen forever.
    def _request_shutdown(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, _request_shutdown)

    last_warn = 0.0
    last_disk_warn = -3600.0
    #: Separate 120s cooldown for the per-victim notification, mirroring the general
    #: "warn" rate limit — a warning that keeps clearing (sustained recovery) and
    #: re-arming under bouncy memory would otherwise re-notify (osascript + tty write)
    #: every single low sample with no cap. Throttles only the NOTIFICATION; the grace
    #: mechanics (state.warned_pid/warned_samples) always update regardless, so a
    #: suppressed re-notification never weakens the actual protection. Keyed on the
    #: victim's PID (not wall time alone) — the module docstring promises "the FIRST
    #: time a specific victim is about to be suspended" gets a message; a wall-time-only
    #: cooldown would silently skip that promise for a BRAND NEW victim who happens to
    #: appear inside another victim's cooldown window.
    last_warn_victim_pid: Optional[int] = None
    last_warn_victim_at = 0.0
    try:
        while True:
            try:
                if _exempt_file_mtime() != exempt_mtime:   # edited: applies at once
                    global EXEMPT_PATTERNS, EXEMPT_DIRS
                    EXEMPT_PATTERNS, EXEMPT_DIRS = load_exempt()
                    exempt_mtime = _exempt_file_mtime()    # after a SUCCESSFUL load
                    _announce_exempt()
                ram_free_gb, _total, swap_gb = system_memory()
                swap_room = swap_room_gb()
                # Every decision and message below uses the usable figure.
                free_gb = usable_free_gb(ram_free_gb, swap_room)
                action = decide_action(free_gb, state)
                if free_gb < STOP_FREE_GB or action not in (None, "warn"):
                    # the record a post-mortem needs (alarm zone only: 1 Hz)
                    print(f"{_stamp()} [memguard] sample: usable {free_gb:.1f} GB "
                          f"(RAM free {ram_free_gb:.1f}, swap room {swap_room:.1f}), "
                          f"falling {fall_rate_gb_s(free_gb, state):+.2f} GB/s, kernel "
                          f"pressure {kernel_pressure_level()}, swap used {swap_gb:.1f} "
                          f"GB -> {action or 'none'}", flush=True)
                if time.monotonic() - last_disk_warn > 3600:
                    disk_gb = disk_free_gb()
                    if disk_gb < DISK_WARN_GB:
                        last_disk_warn = time.monotonic()
                        notify("Disk almost full: memory protection weakened",
                               f"only {disk_gb:.1f} GB free on the startup disk. macOS "
                               f"needs free disk space when memory runs low; without it "
                               f"the Mac can freeze. Free up disk space.",
                               dry_run=dry_run)

                if action == "resume":
                    notify("RESUMED your job",
                           f"pid {state.stopped_pid} continues; box has {free_gb:.1f} GB "
                           f"free again.",
                           dry_run=dry_run, pid=state.stopped_pid)
                    _signal_pid(state.stopped_pid, signal.SIGCONT, "resumed", dry_run)
                    state.stopped_pid, state.recovered_samples = None, 0

                elif action == "kill":
                    # Act first, then tell: telling costs time the machine may not have.
                    _signal_pid(state.stopped_pid, signal.SIGKILL, "killed", dry_run)
                    notify("KILLED suspended job",
                           f"pid {state.stopped_pid}: free memory {free_gb:.1f} GB is under "
                           f"the {KILL_FREE_GB:.1f} GB kill threshold even with it "
                           f"suspended. Rerun it with less memory.",
                           dry_run=dry_run, pid=state.stopped_pid)
                    state.stopped_pid, state.recovered_samples = None, 0

                elif action == "warn_victim":
                    procs = scan_processes()
                    measure_footprints(procs)
                    victim = pick_victim(procs)
                    if victim is not None:
                        if (victim.pid != last_warn_victim_pid
                                or time.monotonic() - last_warn_victim_at > 120):
                            last_warn_victim_pid = victim.pid
                            last_warn_victim_at = time.monotonic()
                            tty = resolve_ttys(procs).get(victim.pid, "")
                            # The warning's own sample already counts toward the
                            # grace countdown (bookkeeping below increments
                            # warned_samples on this same iteration), so the elapsed
                            # time to "stop" is WARN_GRACE_SAMPLES - 1 further
                            # intervals, not WARN_GRACE_SAMPLES — stating the larger
                            # figure would overstate the agent's real reaction window.
                            grace_s = max(0, WARN_GRACE_SAMPLES - 1) * SAMPLE_SECONDS
                            notify(
                                "Memory low — your job may be suspended soon",
                                f"{victim.short_command} pid {victim.pid} using "
                                f"{victim.best_gb:.1f} GB; box has {free_gb:.1f} GB "
                                f"free. Reduce its memory now, or it will be "
                                f"SUSPENDED in ~{grace_s:.0f}s if memory is still low. "
                                f"Options: {STREAMING_LOADER_GUIDANCE}",
                                tty=tty,
                                dry_run=dry_run,
                                pid=victim.pid,
                            )
                        state.warned_pid = victim.pid
                        state.warned_samples = 0

                elif action in ("stop", "stop_severe"):
                    procs = scan_processes()
                    measure_footprints(procs)
                    if action == "stop":
                        # Grace period elapsed: honor the warning by suspending the
                        # SAME process that was warned, never a different,
                        # never-warned one. If it already exited (or is no longer
                        # is_targetable — e.g. now matches an exemption) on its own,
                        # the warning did its job; don't punish a different process
                        # in its place this cycle — the next low-memory sample
                        # re-evaluates from scratch and warns whoever is heaviest
                        # THEN.
                        victim = next(
                            (p for p in procs
                             if p.pid == state.warned_pid and p.is_targetable),
                            None,
                        )
                    else:
                        # stop_severe: a genuine OOM cliff, no time was given for a
                        # warning. Pick FRESH — even if a warning happens to be
                        # PENDING from an earlier, shallower episode, it must not pin
                        # the target to that (possibly smaller) process while the
                        # actual largest consumer keeps growing.
                        victim = pick_victim(procs)
                    if victim is not None:
                        # Act first, then tell: telling costs time the machine may
                        # not have.
                        _signal_pid(victim.pid, signal.SIGSTOP, "suspended", dry_run)
                        tty = resolve_ttys(procs).get(victim.pid, "")
                        notify(
                            "SUSPENDED your job (memory)",
                            f"{victim.short_command} pid {victim.pid} using "
                            f"{victim.best_gb:.1f} GB; box had {free_gb:.1f} GB free. "
                            f"It resumes automatically when the box drains.",
                            tty=tty,
                            dry_run=dry_run,
                            pid=victim.pid,
                        )
                        state.stopped_pid = victim.pid
                        state.recovered_samples = 0
                    # The warning episode (if any) is over either way — a fresh
                    # cycle later should warn again (a new victim may be heaviest by
                    # then), not skip straight to stop using a stale grace-period
                    # counter. Also clears a stale pending warning that stop_severe
                    # just superseded.
                    state.warned_pid, state.warned_samples, state.warn_recovered_samples = (
                        None, 0, 0)

                elif action == "warn" and time.monotonic() - last_warn > 120:
                    last_warn = time.monotonic()
                    notify("Memory low", f"only {free_gb:.1f} GB free — "
                                         f"run `uv run memguard/memwatch.py` to see who",
                           dry_run=dry_run)

                if state.stopped_pid is not None and free_gb >= RESUME_FREE_GB:
                    state.recovered_samples += 1
                elif state.stopped_pid is not None:
                    state.recovered_samples = 0

                # Grace-period countdown for an already-warned victim, and its
                # clearing on recovery — SAME hysteresis as the stopped_pid ->
                # resume transition above (RESUME_FREE_GB sustained for
                # RESUME_SAMPLES), not a single tick back above STOP_FREE_GB. A
                # zero-hysteresis reset livelocks: memory oscillating around
                # STOP_FREE_GB would re-warn forever and never reach "stop". Free memory sitting in the band
                # between STOP_FREE_GB and RESUME_FREE_GB neither counts toward the
                # grace period nor clears it — genuinely low samples still
                # accumulate toward "stop" even with noise in that band, and only a
                # SUSTAINED recovery clears the warning.
                if state.warned_pid is not None:
                    if free_gb >= RESUME_FREE_GB:
                        state.warn_recovered_samples += 1
                        if state.warn_recovered_samples >= RESUME_SAMPLES:
                            state.warned_pid, state.warned_samples, \
                                state.warn_recovered_samples = None, 0, 0
                    else:
                        state.warn_recovered_samples = 0
                        if free_gb < STOP_FREE_GB:
                            state.warned_samples += 1

                state.recent_free = (state.recent_free + [free_gb])[-FALL_WINDOW_SAMPLES:]
                time.sleep(SAMPLE_SECONDS)
            except KeyboardInterrupt:
                return 0
            except Exception as exc:                      # a watchdog must not die
                print(f"{_stamp()} [memguard] sample failed: {exc!r}", file=sys.stderr, flush=True)
                time.sleep(SAMPLE_SECONDS)
    finally:
        # THE point of this block: a suspended process holds all its memory and has no
        # way to un-suspend itself. If the watchdog exits while holding one — Ctrl-C,
        # SIGTERM from `launchctl unload`, an unhandled crash — that job would stay
        # frozen forever with nobody left to resume it. Shutting the watchdog down must
        # never leave the box worse than not running it at all.
        if state.stopped_pid is not None:
            print(f"{_stamp()} [memguard] shutting down — resuming suspended pid "
                  f"{state.stopped_pid} first", flush=True)
            _signal_pid(state.stopped_pid, signal.SIGCONT, "resumed on shutdown", dry_run)
            state.stopped_pid = None
        print(f"{_stamp()} [memguard] watchdog stopped", flush=True)


def _signal_pid(pid: Optional[int], sig: int, verb: str, dry_run: bool = False) -> None:
    if pid is None:
        return
    if dry_run:
        print(f"{_stamp()} [memguard] DRY RUN: would have {verb} pid {pid}", flush=True)
        return
    try:
        os.kill(pid, sig)
        print(f"{_stamp()} [memguard] {verb} pid {pid}", flush=True)
    except ProcessLookupError:
        print(f"{_stamp()} [memguard] pid {pid} already gone", flush=True)
    except PermissionError:
        print(f"{_stamp()} [memguard] not permitted to signal pid {pid}", file=sys.stderr)


#: Operator actions on one window's compute children. Kill is last and separate
#: because it is the only irreversible one.
_ACTIONS = {
    "stop":   (signal.SIGSTOP, "suspended"),
    "resume": (signal.SIGCONT, "resumed"),
    "kill":   (signal.SIGKILL, "killed"),
}


def run_action(tty: str, action: str) -> int:
    """Apply `action` to one window's compute children, sparing its `claude` session.

    Manual counterpart to the watchdog: `stop` freezes a runaway job WITHOUT losing it
    (its memory is still held, but it stops growing), `resume` lets it continue, `kill`
    discards it. Suspend/resume is the pair to reach for first — a stopped job can always
    be resumed, and a killed one cannot.
    """
    sig, verb = _ACTIONS[action]
    tty = tty.replace("/dev/", "")
    procs = scan_processes()
    tty_of = resolve_ttys(procs)

    # Same bucketing as the report (detached_bucket_key), so "detached" here can
    # never select a different, wider set of processes than what memwatch's own
    # report shows under that name.
    owned = [p for p in procs if detached_bucket_key(p, tty_of.get(p.pid, "")) == tty]
    if not owned:
        print(f"no processes on {tty} — check the TTY column of `memwatch`")
        return 1

    # `resume` must target the SUSPENDED children specifically; the others are the
    # compute jobs a stop/kill applies to.
    if action == "resume":
        targets = [p for p in owned if p.is_suspended]
        if not targets:
            print(f"nothing suspended on {tty}")
            return 1
    else:
        targets = [p for p in owned if p.is_targetable and p.best_gb >= 0.05]
        if not targets:
            print(f"no targetable compute processes on {tty} "
                  f"(its `claude` session is deliberately spared)")
            return 1

    for proc in sorted(targets, key=lambda p: -p.rss_gb):
        print(f"{action}: pid {proc.pid} ({proc.short_command}, "
              f"{proc.rss_gb:.1f} GB RSS)")
        _signal_pid(proc.pid, sig, verb)

    if action == "stop":
        print(f"\nStill holding its memory — suspension stops GROWTH, not usage.\n"
              f"Resume with:  uv run memguard/memwatch.py --resume {tty}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--watch", action="store_true", help="refresh continuously")
    parser.add_argument("--daemon", action="store_true", help="run the OOM watchdog")
    parser.add_argument("--dry-run", action="store_true",
                        help="with --daemon: decide and log, but send no signals")
    parser.add_argument("--stop", metavar="TTY",
                        help="SUSPEND this window's compute children (reversible)")
    parser.add_argument("--resume", metavar="TTY",
                        help="resume this window's suspended children")
    parser.add_argument("--kill", metavar="TTY",
                        help="KILL this window's compute children (irreversible)")
    args = parser.parse_args(argv)

    if args.daemon:
        return run_daemon(dry_run=args.dry_run)
    for action in ("stop", "resume", "kill"):
        target = getattr(args, action)
        if target:
            return run_action(target, action)
    return run_monitor(args.watch)


if __name__ == "__main__":
    sys.exit(main())
