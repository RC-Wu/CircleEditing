# Frontier Fallback Grid Report (2026-03-27)

## Summary

This round moved the active CircleEditing frontier pipeline onto the new stable path:
- `frontier_seed1`
- `SAM3` mask backend on CPU
- `TTT3R` on CPU
- `AGT` disabled (`--skip_agt` / `EDITSPLAT_SKIP_AGT=1`)
- new `black-face fallback` enabled in `run_editing_flow.py`

The key outcome is that the previous black-face failure mode has been removed in the neighbor views. The fallback triggers correctly when the reprojected proxy face is effectively zero, and `mf_cond` now contains an editable clown-makeup face instead of a black blob.

## Code Changes

Modified runtime files:
- `/dev_vepfs/rc_wu/edit/CircleEditing/runtime/EditSplat_overlay_20260326/run_editing_flow.py`
- `/dev_vepfs/rc_wu/edit/CircleEditing/runtime/EditSplat_overlay_20260326/scene/gaussian_model.py`
- `/dev_vepfs/rc_wu/edit/CircleEditing/runtime/EditSplat_overlay_20260326/utils/frontier_fallback.py`
- `/dev_vepfs/rc_wu/edit/CircleEditing/runtime/EditSplat_overlay_20260326/tests/test_frontier_fallback.py`

Behavioral changes:
- added pure tensor helpers for black-face detection and anchor-face fallback compositing
- cached frontier anchor mask and used it for neighbor fallback composition
- exposed fallback metadata in `debug_intermediates/mfg_edit/*/stats.json`
- added `apply_weights()` tensor/device/shape assertions before the diffGS CUDA call
- fixed the `epoch=0` save path crash (`UnboundLocalError`)

## Smoke Validation

Successful smoke run:
- `/dev_vepfs/rc_wu/edit/CircleEditing/runs/frontier_seed1_epoch0_fallback_smoke2_seed211_20260327_102211`

Evidence:
- `frontier_fallback_triggered=true` for both neighbor views
- `frontier_proxy_mean=0.0` for both neighbor views
- `mf_cond` is no longer black after fallback

Local visual sheet:
- `F:/InformationAndCourses/Code/CircleEditing/assets/review/contact_sheet_fallback_smoke2.png`

## 4-GPU Grid

Run matrix:
- `/dev_vepfs/rc_wu/edit/CircleEditing/runs/frontier_seed1_grid_vel_seed131_fb_20260327_103126`
- `/dev_vepfs/rc_wu/edit/CircleEditing/runs/frontier_seed1_grid_vel_seed211_fb_20260327_103126`
- `/dev_vepfs/rc_wu/edit/CircleEditing/runs/frontier_seed1_grid_prox_seed131_fb_20260327_103126`
- `/dev_vepfs/rc_wu/edit/CircleEditing/runs/frontier_seed1_grid_prox_seed211_fb_20260327_103126`

Common settings:
- `epoch=0`
- `depth_mode=constant`
- `skip_agt=true`
- `EDITSPLAT_FRONTIER_BLACKFACE_FALLBACK=1`
- `EDITSPLAT_FRONTIER_BLACKFACE_MEAN_THR=0.08`
- `EDITSPLAT_FRONTIER_FALLBACK_FEATHER=9`

All four runs completed successfully and all neighbor views recorded:
- `frontier_fallback_triggered=true`
- `frontier_proxy_mean=0.0`
- non-black `mf_cond`

## Quantitative Triage

Face-mask mean intensity inside the target mask:

| run | view001 mf_cond | view001 output | view002 mf_cond | view002 output |
| --- | ---: | ---: | ---: | ---: |
| vel131 | 0.6488 | 0.6501 | 0.6381 | 0.6577 |
| vel211 | 0.6470 | 0.6528 | 0.6389 | 0.6783 |
| prox131 | 0.6489 | 0.6310 | 0.6384 | 0.6321 |
| prox211 | 0.6481 | 0.6326 | 0.6398 | 0.6488 |

Interpretation:
- `velocity` clearly preserves / amplifies the fallback-conditioned face better than `proximal`
- `vel211` is the strongest on `view002`
- `proximal` tends to wash out or under-edit the neighbor-face result compared with the conditioned face

## Visual Triage

Local comparison sheet:
- `F:/InformationAndCourses/Code/CircleEditing/assets/review/frontier_seed1_grid_compare_20260327_remote.jpg`

Visual conclusion:
- `vel211` is the best overall current setting
- `vel131` is the second-best and sometimes produces a slightly harsher front-face style
- both `proximal` runs are weaker than `velocity` on the actual edited neighbor outputs
- the black-face failure mode is gone in all four runs

## AGT / Memory Status

AGT remains disabled in the productive lane.

Current conclusion on the old illegal-memory path:
- the crash root is still the diffGS `apply_weights` CUDA path, not the high-level mask pipeline
- the productive route is to keep `skip_agt=true`
- current Python assertions in `GaussianModel.apply_weights()` are diagnostic guards only; they do not prove the CUDA kernel is fixed

## Next Runs Started

Follow-up `epoch=1` runs launched after the grid:
- `/dev_vepfs/rc_wu/edit/CircleEditing/runs/frontier_seed1_follow_vel_seed131_epoch1_fb_20260327_104449`
- `/dev_vepfs/rc_wu/edit/CircleEditing/runs/frontier_seed1_follow_vel_seed211_epoch1_fb_20260327_104449`

Purpose:
- test whether the `skip AGT + frontier fallback` route survives the 3DGS optimization stage
- compare `seed131` vs `seed211` under `epoch=1`
- keep partial artifacts if backward fails via `EDITSPLAT_SKIP_3DGS_BACKWARD_ON_ERROR=1`


## Follow-up Result: Epoch-1 Abort-Fix

Patched behavior:
- `run_editing_flow.py` now treats both `illegal memory access` and `invalid argument` as abort-worthy 3DGS backward errors when `EDITSPLAT_SKIP_3DGS_BACKWARD_ON_ERROR=1` is set.
- This preserves partial artifacts instead of killing the full run.

Verified run:
- `/dev_vepfs/rc_wu/edit/CircleEditing/runs/frontier_seed1_follow_vel_seed211_epoch1_fb_abortfix_20260327_105352`

Observed behavior:
- the run completes the `frontier_seed1 + fallback` MFG stage
- it enters `EPOCH 0: optimizing 3D Gaussian Splatting`
- `total_loss.backward()` still hits `RuntimeError: CUDA error: invalid argument`
- the new guard catches it, aborts optimization cleanly, and still saves:
  - `/dev_vepfs/rc_wu/edit/CircleEditing/runs/frontier_seed1_follow_vel_seed211_epoch1_fb_abortfix_20260327_105352/point_cloud/iteration_7004/point_cloud.ply`
  - `/dev_vepfs/rc_wu/edit/CircleEditing/runs/frontier_seed1_follow_vel_seed211_epoch1_fb_abortfix_20260327_105352/point_cloud/iteration_7004/chkpnt7004.pth`

Visual artifact:
- `F:/InformationAndCourses/Code/CircleEditing/assets/review/contact_sheet_epoch1_abortfix.jpg`

Interpretation:
- the productive route is now stable through the full pre-optimization pipeline and can preserve outputs even when the diffGS backward path fails
- the next low-level blocker is no longer AGT; it is the diffGS backward `invalid argument` path
- the best current practical recipe is still `velocity + seed211 + skip_agt + frontier fallback`
