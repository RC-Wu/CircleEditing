# Dev01 Three-GPU Ralph Loop

Date: 2026-04-13

Scope:
- Occupy `dev-intern-01` GPUs `0/1/2` only.
- Keep each card on a fixed sequential queue with at least 5 jobs.
- Stay within `/dev_vepfs/rc_wu/edit/CircleEditing` long-term migration direction and avoid loose temp trees.
- Keep incremental write growth on `vePFS` below `50GB`.

Research framing:
- Wave17 and wave18 still show full-face collapse for structured face-cover edits.
- `--resolution 8` was confirmed to downscale 512px views to ~64px, so it cannot remain the default for validation runs.
- The immediate dev01 objective is split across three lines:
  1. full-face prompt separation re-validation at sane render widths
  2. resolution sanity sweep under a fixed carrier method
  3. canonical/carrier stability after the latest fit-mask and tensor-device fixes

GPU plan:
- GPU0: prompt-separation line
- GPU1: resolution-sweep line (`320 / 256 / 384 / 512 / -1`; never `8`)
- GPU2: canonical/carrier-stability line

Visual-first outputs required for every job:
- `analysis/panel_final_grid.png`
- `analysis/panel_pipeline.png`
- `debug_intermediates/semantic_guidance/gaussian_mask_stats.json`
- representative `debug_intermediates/mfg_edit/view000/*`

Operational rules:
- Reuse cached weights only; no new bulk model prefetch.
- Keep `HF_HUB_OFFLINE=1` unless a targeted fix explicitly requires otherwise.
- Delete transient point clouds/debug step dumps through queue GC after panel generation.
- Mirror conclusions later into AgentDoc + project docs + Obsidian note after one line completes.
