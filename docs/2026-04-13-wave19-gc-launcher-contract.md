# Wave19 Queue GC Launcher Contract

The `build_fixed_gpu_overnight_queue.py` helper originally passed the selected launcher module to `run-one` and `postprocess-job`, but `gc-job` reconstructed the experiment with its hardcoded `DEFAULT_LAUNCHER_MODULE`. That was safe only while every queue used the default launcher schema.

Wave19 `dev01` was emitted with `launch_carrier_probe_wave.py`, whose `Experiment` dataclass includes newer fields such as `resolution` and `epoch`. During cleanup, `gc_job(...)` loaded `launch_dev01_ttt3r_consistency_wave.py` instead, and the mismatch crashed with:

`TypeError: Experiment.__init__() got an unexpected keyword argument 'resolution'`

Fix:

- Persist the queue launcher contract in `queue_config.json` when emitting queue files.
- Pass `--launcher-module` to future `gc-job` commands in generated slot scripts.
- Keep a backward-compatible resolver in `gc_job(...)` that can recover the launcher from existing `slot_*.sh` files for already-emitted queues.

Practical rule: any queue helper that reconstructs `Experiment(**exp_kwargs)` must resolve the same launcher module that was used to emit and launch the queue, not assume the default launcher remains schema-compatible.
