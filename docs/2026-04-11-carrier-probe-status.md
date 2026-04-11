# 2026-04-11 Carrier Probe Status

## Scientific Status

- Latest human docs still agree on the same core conclusion: `SAM3 -> semantic GS guidance -> open GS` is real, but full-face prompt variants collapse in final 3D even when teacher-side differences are visible.
- The current anchor regime remains `open_semboost_core`.
- The next research priority is still carrier strength, not more generic prompt or semantic-weight sweeps.
- Immediate experiment priorities remain:
  - `A-baseline`: cheap warp/fuse residual carrier sanity probe
  - `B-lite`: stronger canonical edit-field probe
  - `PSRR`: track how much teacher separation survives final 3D

## Code Status

- Branch: `codex/20260411-carrier-probe`
- Latest synced commit: `f85d463 fix: restore cached base model defaults for carrier probe`
- This branch now contains:
  - prior `A-baseline` carrier helper and PSRR support
  - a launcher fix that restores the historical base-model default `cocktailpeanut/xulf-s`
  - a regression test for launcher env construction
- The staging worktree under `_codex_staging/20260411_circleediting_carrier_probe` is updated to `f85d463`, but it is still not a complete runnable EditSplat mirror, so live jobs continue to use the old runnable tree under `/dev_vepfs/rc_wu/edit/EditSplat`.

## Runtime Status

- The original dev02 wave failure root cause was confirmed:
  - queue path forced `HF_HUB_OFFLINE=1`
  - no `EDITSPLAT_BASE_MODEL_ID` override was present
  - wrapper fell back to `black-forest-labs/FLUX.1-dev`
  - that model was not cached locally, so startup died in `from_pretrained()`
- Historical successful runs show the intended base model was actually `cocktailpeanut/xulf-s`, not bare FLUX.
- A controlled prefetch wave is now live on `dev-intern-02` GPU `2`:
  - wave: `20260411_wave18_prefetch_xulfs_gpu2_debug`
  - purpose: repopulate `/dev_vepfs/rc_wu/cache/hf_home_dev02/hub/models--cocktailpeanut--xulf-s`
  - mode: `HF_HUB_OFFLINE=0`, `TRANSFORMERS_OFFLINE=0`
- A chained follow-up launcher is already armed:
  - once the prefetch wave exits with `returncode=0`, it auto-emits and auto-starts
  - wave: `20260411_wave18_a_baseline_xulfs_gpu2`
  - mode: offline again, base model pinned to `cocktailpeanut/xulf-s`
  - jobs: baseline + `A-baseline`, sequential on GPU `2`

## GPU Status

- `dev-intern-01` GPUs `4/5/6/7` are reserved by the user and must not be touched.
- At the time of this note there is no safe 4-GPU block available across allowed slots.
- Latest observed allowed-slot state:
  - `dev-intern-01`: GPU `0` still above idle threshold; GPUs `1/2/3` occupied
  - `dev-intern-02`: GPU `2` is the only practical free slot and is already assigned to the prefetch chain

## Next Actions

1. Let the `xulf-s` prefetch complete and verify the remaining `.incomplete` blobs disappear.
2. Confirm the chained offline wave `20260411_wave18_a_baseline_xulfs_gpu2` starts automatically on GPU `2`.
3. If the chained wave runs cleanly, inspect outputs and compute PSRR-style comparisons against the anchor baseline.
4. Keep monitoring `dev-intern-01` GPUs `0/1/2/3` and `dev-intern-02` spare slots for the first moment a true 4-GPU allowed window appears.
5. When a 4-GPU window appears, launch the next wider carrier wave on allowed slots only, still pinned to the restored `xulf-s` cache path.
