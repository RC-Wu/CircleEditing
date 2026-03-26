# Frontier Seed1 Attempt 2026-03-26

## Scope

- Current machine scope: `shared`
- Primary execution machine: `dev-intern-01`
- Secondary diagnosis machine: `dev-intern-02`
- Local repo: `F:/InformationAndCourses/Code/CircleEditing`
- Remote overlay root: `/dev_vepfs/rc_wu/CircleEditing/runtime/EditSplat_overlay_20260326`

## What Was Implemented

### 1. Frontier scheduling in `run_editing_flow.py`

- Added `EDITSPLAT_MFG_MODE=frontier_seed1`.
- The new mode:
  - computes initial edits for the truncated training set,
  - selects one anchor view by initial-edit diff score,
  - expands only to first-ring neighboring views,
  - keeps downstream `edited_image_MFG_list[idx]` unchanged so the existing 3DGS fit loop still works.

### 2. Role-aware SAM3 support caching in `run_sd35_ttt3r_sam3_wrapper.py`

- `support_mask_cache` now stores per-role payloads instead of a single tensor overwrite.
- `_predict_langsam_mask(...)` accepts `mask_role`.
- Fit-loss fusion now explicitly uses `EDITSPLAT_SAM3_FIT_ROLE`, defaulting to `gt_view`.

### 3. Earlier SAM3 fusion in `run_sd35_ttt3r_proximal_wrapper.py`

- Added support-mask access in `_edit_image_mfg`.
- `edit_ratio` and `preserve_ratio` are modulated by the cached support mask before `edit_weight` / `preserve_weight` are formed.

## Runs

### A. Render-depth frontier run on `dev-intern-01`

- Run root:
  - `/dev_vepfs/rc_wu/CircleEditing/runs/frontier_seed1_sam3_local_dev01_20260326_20260326_122639`
- Result:
  - Reached `initial edit`, `frontier anchor` selection, and entered `Multi-view reprojection`.
  - Failed because `render depth` hit `CUDA illegal memory access`, poisoning the CUDA context before attention weighting.

### B. Constant-depth frontier run on `dev-intern-01`

- Run root:
  - `/dev_vepfs/rc_wu/CircleEditing/runs/frontier_seed1_sam3_local_constdepth_dev01_20260326_20260326_122905`
- Result:
  - Completed initial edits, frontier selection, two neighbor MFG passes, SAM3 support-mask generation, and one attention-weighting iteration.
  - Visual review showed the target mask on neighbor views was too small and locked onto a wrong blob.

### C. Constant-depth frontier run with fixed target-view mask

- Run root:
  - `/dev_vepfs/rc_wu/CircleEditing/runs/frontier_seed1_sam3_local_constdepth_fixmask_dev01_20260326_20260326_123724`
- Result:
  - Completed the same stages as run B.
  - Visual review showed the target-view mask became correct and face-local, but `mf_cond` / `proxy` still collapsed to a black filled face region instead of propagating the clown edit.
  - The next blocker remains downstream content transfer, not mask localization.

## Human Visual Review

Local thumbnails copied to:

- `F:/InformationAndCourses/Code/CircleEditing/assets/review/frontier_seed1_constdepth_dev01_20260326_122905/thumbs/contact_sheet.jpg`
- `F:/InformationAndCourses/Code/CircleEditing/assets/review/frontier_seed1_constdepth_fixmask_dev01_20260326_123724/thumbs/contact_sheet.jpg`

Observed with direct visual inspection:

- First constant-depth version:
  - neighbor masks latched onto an upper-left blob artifact instead of the face.
- Fixed-mask version:
  - neighbor masks are now face-aligned,
  - but the projected content inside the face is almost entirely black,
  - so the current bottleneck is content propagation / completion, not segmentation.

## Main Takeaways

1. The new frontier pipeline is no longer hypothetical; it runs through anchor selection, neighbor expansion, SAM3 support caching, and local TTT3R-weight construction.
2. `render depth` is still blocked by the old `diffGS illegal memory access` failure path.
3. Replacing the neighbor compositing mask with the target-view face mask fixed localization.
4. The next iteration should replace the current black-face content source with one of:
   - anchor-face masked color transfer plus feathering,
   - TTT3R / MASt3R-guided patch completion,
   - or direct multi-view completion on the masked target view.

## Files Patched In This Session

- `runtime/EditSplat/run_editing_flow.py`
- `runtime/EditSplat/sandboxes/20260319_aris_ttt3r_flowedit_45/scripts/run_sd35_ttt3r_proximal_wrapper.py`
- `runtime/EditSplat/sandboxes/20260322_editsplat_ttt3r_flowedit_sam3_4567/scripts/run_sd35_ttt3r_sam3_wrapper.py`
