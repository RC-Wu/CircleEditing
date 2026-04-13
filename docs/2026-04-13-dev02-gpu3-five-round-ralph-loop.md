# dev02 GPU3 Five-Round Ralph Loop

## Research Findings

- The latest human notes pin the scientific blocker on full-face prompt collapse after the 2D teacher stage.
- The current anchor regime remains `open_semboost_core`.
- The next experiments should prioritize carrier strength in this order: `A-baseline`, then `B-lite`, then PSRR-style interpretation from teacher-fit versus final render.
- Visual evidence is mandatory; panel and render artifacts must be generated and reviewed directly instead of relying on scalar metrics alone.

## Execution Design

- Target machine: `dev-intern-02`
- GPU: `3`
- Queue style: single-slot sequential runner that keeps advancing even if one round fails
- Launch path: `build_wave18_gpu3_five_round_manifest.py` -> `run_wave18_gpu3_five_rounds.py`
- Monitoring path: `monitor_wave18_gpu3_queue.py`

## Five Rounds

1. `bandage_wrap_open_semboost_core`
2. `bandage_wrap_open_semboost_core_a_baseline`
3. `bandage_wrap_open_semboost_core_blite`
4. `goldmask_structured_open_semboost_core_a_baseline`
5. `goldmask_structured_open_semboost_core_blite`

## Acceptance

- GPU `3` stays occupied by the queue or its immediate postprocess path until the five rounds finish.
- Each round writes a reproducible status JSON, log path, and result directory.
- Monitor writes a living note into AgentDoc and a mirrored project-side note in the runnable tree.
- Final review must inspect rendered visual artifacts before drawing conclusions.

## Live Status

- `retry3` cleared the earlier runtime failures: the first four status JSONs now report `returncode=0`.
- Direct visual review of the completed `panel_final_grid` outputs still shows semantic collapse: no convincing bandage or gold-mask structure appears, only mild blur / drift.
- The panel exporter is also incomplete for these runs: the lower half of the final grids is largely black, so panel generation itself needs follow-up debugging before relying on the collage format.
