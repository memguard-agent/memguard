<p align="center"><img src="assets/memguard_logo.png" alt="memguard — Resource management for autonomous AI agents" width="720"></p>

**The resource manager for autonomous AI agents on macOS.**

Running several AI agents on one Mac can bring it down. Each agent starts its own jobs: model inference, structure prediction, data processing. Together they can use up all the memory, and the Mac freezes or restarts. memguard stops that from happening.

It watches every job on your Mac, **including GPU memory that Activity Monitor doesn't show**. When memory runs low, it **tells the agent responsible** so the agent can shrink its job. If that doesn't help, memguard pauses the largest job and resumes it once memory frees up. A job is killed only as a last resort.

**Nothing to configure. No changes to your agents.** Install it once and it runs in the background.

## Install

1. Download the latest **memguard .dmg** from [Releases](https://github.com/memguard-agent/memguard/releases).
2. Open it and double-click **Install memguard**.
3. Answer the questions. That's it.

Requires macOS. Tested on macOS 26.5.

## What you'll notice

- **Normally, nothing.** memguard uses almost no resources.
- **When memory gets low,** you get a notification naming the job that uses the most memory.
- **Your Claude Code agents are told too.** An agent learns that its job is using too much memory and how to reduce it, before anything gets paused. It is also told if its job is paused, resumed or stopped.
- **If memory stays critical,** the largest job is paused and continues on its own once memory frees up. Your agents themselves are never stopped, only the jobs they started.

| Free memory | What memguard does |
|---|---|
| under 6 GB | notifies you |
| under 3.5 GB | warns the agent, then pauses its job about 30 seconds later |
| under 2 GB | pauses the largest job immediately; if memory is still critical, stops it |
| back above 6 GB | resumes the paused job |

## Protect important programs

To make sure a program is never paused, add part of its name to the file `~/.config/memguard/exempt`, one name per line. For example:

```
redis-server
my_server.py
```

## Uninstall

Open the disk image again and double-click **Uninstall memguard**.

## License

memguard is licensed under the [Apache License 2.0](LICENSE).
