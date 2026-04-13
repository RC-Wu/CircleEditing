# Wave19 GC Launcher Contract

Date: 2026-04-13

Problem:
- `gc_job(...)` used the default launcher contract even when a queue was emitted with a different launcher module.
- `wave19` jobs include newer fields such as `resolution` and `epoch`, so cleanup crashed when the wrong `Experiment` dataclass was reconstructed.

Rule:
- The queue must persist the launcher module it was emitted with.
- `gc-job` must prefer the explicit launcher argument, then `queue_config.json`, then an existing `slot_*.sh` launcher reference before falling back to the default launcher.

Reason:
- Postprocess and GC must reconstruct the same run-name and dataclass contract as the original queue emission path.
