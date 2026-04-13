# Continuous Ralph Loop

Date: 2026-04-13

Objective:
- Keep the active 3d-edit experiment wave moving without leaving `dev01 GPU0/1` or `dev02 GPU3` idle.
- Monitor live runs, review visual outputs, document real status, and schedule the next safe intervention.

Current compute contract:
- `dev-intern-01`: only `GPU0` and `GPU1` may be used by this workflow.
- `dev-intern-01 GPU2`: forbidden after user reclaim; never relaunch there.
- `dev-intern-01 GPU3-7`: unavailable.
- `dev-intern-02`: only the current `GPU3` retry wave is in scope.

Loop cadence:
1. Observe
   - read `nvidia-smi`, active wrapper PIDs, queue status JSONs, and the newest logs
   - inspect newly generated `panel_final_grid.png` / `panel_pipeline.png` when a job finishes
2. Judge
   - decide whether the run is healthy, stalled, visually failed, or finished-but-incomplete
   - compare against the known blocker: full-face prompt collapse after the 2D teacher stage
3. Act
   - only take safe actions: restart our own dead queue slot, regenerate a manifest, or requeue a later experiment
   - never kill unknown processes and never occupy forbidden GPUs
4. Document
   - append one timestamped note section into the heartbeat note
   - mirror important findings into the project docs before the session ends
5. Sleep
   - wait for the next heartbeat interval unless a just-finished job requires immediate follow-up

Current judged state:
- `dev02 GPU3 retry3`: all five status files now succeed at runtime, but direct visual review still shows semantic failure; no convincing bandage or gold-mask geometry is formed.
- `dev01 GPU1`: the invalid `resolution=8` validation head was replaced in-flight with `resolution=320`.
- `dev01 GPU2`: carrier-stability jobs are deferred for later rescheduling because the GPU was reclaimed.

Next interventions in priority order:
1. Finish monitoring the active `dev01 GPU0/1` jobs and collect the next completed visual panels.
2. Debug the broken/black lower-half panel export path so visual review can trust the grid format.
3. Repack the old GPU2 carrier-stability line into a future queue on an allowed GPU after one of the active slots frees up.
