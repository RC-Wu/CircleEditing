# Fix dev01 wave19 gc_job crash

Work only in `/dev_vepfs/rc_wu/edit/CircleEditing`.

Task:
- Fix the dev01 wave19 queue crash in `runtime/EditSplat/sandboxes/20260322_editsplat_ttt3r_flowedit_sam3_4567/scripts/build_fixed_gpu_overnight_queue.py`.
- The current failure is in `gc_job(...)`:
  - `TypeError: Experiment.__init__() got an unexpected keyword argument 'resolution'`
- Root cause hint:
  - `gc_job(...)` currently loads `DEFAULT_LAUNCHER_MODULE`, constructs `launcher.Experiment(**exp_kwargs)`, and the default launcher does not accept the newer wave19 fields such as `resolution` / `epoch`.
  - The active queue was launched with `launch_carrier_probe_wave.py`, so the cleanup path is using the wrong launcher contract.

Required outcome:
1. Find the exact root cause in the live code.
2. Make the minimal safe fix.
3. Add or update tests so the bug is covered.
4. Run the relevant tests.
5. If safe, relaunch only the stalled allowed queue slots for dev01 wave19.

Constraints:
- Never use or relaunch on `dev-intern-01` GPU2.
- Never touch `dev-intern-01` GPU3-7.
- Only operate on our own queue:
  - `/dev_vepfs/rc_wu/edit/EditSplat/sandboxes/20260322_editsplat_ttt3r_flowedit_sam3_4567/runtime/queues/20260413_wave19_dev01_three_gpu`
- Do not kill unknown processes.
- Stay inside the `CircleEditing` project root as the source of truth.
- Mirror any reusable finding into project docs if you make a meaningful fix.

Verification:
- Prefer repo tests first.
- Then confirm the relevant queue command no longer crashes on the `resolution` field path.
- If you relaunch, confirm only the allowed GPU slot is used.

When done:
- Print a concise summary with changed files, tests run, and whether a relaunch was performed.
