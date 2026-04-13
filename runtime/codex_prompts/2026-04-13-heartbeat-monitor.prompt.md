# Heartbeat Monitor

Work only in `/dev_vepfs/rc_wu/edit/CircleEditing`.

This is a long-running heartbeat monitor for a remote task. Stay in a loop until the stop conditions are met or until explicitly stopped.

Every 20 minutes:

- Monitor dev01 wave19 on GPU0/GPU1 only and keep forbidden GPU2 unused.
- Monitor dev02 GPU3 wave18 retry3 completion, collect status/result paths, and flag any new finished visual artifacts for review.
- If one of our active slots is dead and incomplete, only restart our own queue slot after confirming no equivalent wrapper is still alive.

Always inspect these paths when relevant:

- /dev_vepfs/rc_wu/edit/CircleEditing/docs/2026-04-13-continuous-ralph-loop.md
- /dev_vepfs/rc_wu/edit/EditSplat/sandboxes/20260322_editsplat_ttt3r_flowedit_sam3_4567/runtime/queues/20260413_wave19_dev01_three_gpu
- /dev_vepfs/rc_wu/_codex_staging/ttt3r_overnight/20260413_wave18_gpu3_five_rounds_retry3

Always inspect these processes or command fragments when relevant:

- 20260413_wave19_dev01_three_gpu
- run_sd35_ttt3r_sam3_wrapper.py
- 20260413_wave18_gpu3_five_rounds_retry3

Before any restart or repair, follow these intervention rules:

- Never touch dev01 GPU2 or dev01 GPU3-7.
- Never kill or restart a process unless its PID/path clearly belongs to this project queue.
- Before relaunching any slot, verify current status JSONs, current wrapper PIDs, and current GPU occupancy.

Append one timestamped monitoring section to `/dev_vepfs/rc_wu/edit/CircleEditing/docs/2026-04-13-heartbeat-monitor.md` after each pass.

Each section should include:
- current UTC time and local derived time if helpful
- current status summary
- exact PIDs or confirmation that no relevant process is alive
- key file counts or existence checks
- any intervention taken, with exact commands and paths
- the next expected milestone or blocker

Stop when all of these are true:

- Both dev01 active queue slots and dev02 GPU3 retry wave are complete, and the note records the final visual judgment plus next queue recommendation.
