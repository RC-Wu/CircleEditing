## 2026-03-27T20:45:18Z - Cycle 1: practical 3-view stability relaunch attempt (failed before frontier path)

- Experiment run path:
  - `/dev_vepfs/rc_wu/edit/CircleEditing/runs/frontier_seed1_grid3_editstrong_211_epoch1_patchstability_20260327_203857`
- Launch intent:
  - Validate patched runtime stability on practical 3-view `grid3_editstrong` route at `epoch=1` with `skip_agt=true` and fallback enabled, without using backward-abort guard as evidence.
- Failure signature (captured before code changes):
  - `OSError: Cannot load model black-forest-labs/FLUX.1-dev: model is not cached locally`
  - stack root in `run_sd35_ttt3r_proximal_wrapper.py` at `Editsplat_Pipeline.from_pretrained(...)`
  - this occurred under offline mode and before the frontier reprojection/optimization stages.
- Visual verdict:
  - No frontier outputs were produced (`debug_intermediates/mfg_edit/*` did not run), so this run yields no image-level evidence for patch stability.
- Concrete next move:
  - Relaunch same practical 3-view route with explicit known-working model surface:
    - `EDITSPLAT_BASE_MODEL_ID=cocktailpeanut/xulf-s`
    - `FLOWEDIT_SD35_MEDIUM_TURBO_OPEN_MODEL_ID=cocktailpeanut/xulf-s`
    - keep `--flow_method flowedit --flow_model_key sd35-medium-turbo-open`
    - keep `skip_agt=true` to isolate restore/optimization stability first.

## Rolling Summary

- Restore-to-CUDA patch is still only proven in the 1-view debug lane from prior evidence; practical 3-view proof is pending.
- First practical relaunch failed at model surface resolution (`FLUX.1-dev` offline cache miss), not kernel/runtime.
- Immediate unblock is to force the previously cached `xulf-s` surface and rerun practical 3-view `skip_agt=true`.

## 2026-03-27T20:51:24Z - Cycle 2: xulf override relaunch (adapter mismatch failure)

- Experiment run path:
  - `/dev_vepfs/rc_wu/edit/CircleEditing/runs/frontier_seed1_grid3_editstrong_211_epoch1_patchstability_xulf_20260327_204551`
- Launch intent:
  - Rerun practical 3-view `epoch=1` stability check on known cached surface by forcing:
    - `EDITSPLAT_BASE_MODEL_ID=cocktailpeanut/xulf-s`
    - `FLOWEDIT_SD35_MEDIUM_TURBO_OPEN_MODEL_ID=cocktailpeanut/xulf-s`
- Failure signature (captured before next relaunch):
  - adapter construction failed in `flowedit_adapters.py` while creating SD3 branch for `sd35-medium-turbo-open`
  - terminal error:
    - `RuntimeError: Error(s) in loading state_dict for SD3Transformer2DModel`
    - size mismatch for `context_embedder.weight` (`[3072, 4096]` checkpoint vs `[1152, 4096]` model)
- Visual verdict:
  - No frontier/neighbor outputs were produced; run failed during adapter/model init.
- Concrete next move:
  - Relaunch same run recipe with only:
    - `EDITSPLAT_BASE_MODEL_ID=cocktailpeanut/xulf-s`
  - and **remove** `FLOWEDIT_SD35_MEDIUM_TURBO_OPEN_MODEL_ID` override so the SD3 adapter returns to its known working cached model surface.

## Rolling Summary

- Practical 3-view stability is still pending because both new relaunch attempts failed before frontier execution.
- Failure 1 was base-pipeline FLUX offline miss; failure 2 was over-constrained SD3 adapter model-id override.
- Next relaunch will keep `skip_agt=true`, keep the xulf base pipeline override only, and restore SD3 adapter defaults to reach frontier/epoch-1 path.

## 2026-03-27T20:48:15Z - Visual review: front anchor vs propagated neighbors

- Artifact paths:
  - `/dev_vepfs/rc_wu/edit/CircleEditing/reports/ralph_loop_20260328/anchor_compare_front_20260327.png`
  - `/dev_vepfs/rc_wu/edit/CircleEditing/reports/ralph_loop_20260328/flow_grid3_neighbors_20260327.png`
- Visual verdict:
  - FlowEdit front anchor keeps the subject, fleece, and wall/background noticeably closer to the source than the DNAEdit strong front anchor, but its clown paint is weaker and less cleanly composed across the face. The DNAEdit strong front anchor gives the stronger, more centered makeup concept, but it drifts harder on identity/clothing and reads less source-faithful; the current `grid3_editstrong` propagated neighbors preserve pose reasonably well, yet the makeup style shifts away from the front anchor, especially in the eye shapes, mouth curve, and cheek color placement.
- Concrete next move:
  - Keep FlowEdit as the front anchor for fidelity, then strengthen propagation with tighter style anchoring from that front result and recheck neighbor outputs for eye/mouth consistency before promoting this run.

## 2026-03-27T21:00:15Z - Cycle 3: practical frontier route stability confirmed (skip_agt=true)

- Experiment run path:
  - `/dev_vepfs/rc_wu/edit/CircleEditing/runs/frontier_seed1_grid3_editstrong_211_epoch1_patchstability_tensorart_20260327_205251`
- Launcher surface used:
  - `EDITSPLAT_BASE_MODEL_ID=tensorart/stable-diffusion-3.5-medium-turbo`
  - `FLOWEDIT_SD35_MEDIUM_TURBO_OPEN_MODEL_ID=tensorart/stable-diffusion-3.5-medium-turbo`
  - `--model_key sd35-medium-turbo-open --flow_model_key sd35-medium-turbo-open`
  - `EDITSPLAT_SKIP_AGT=1`
- Success evidence:
  - initial editing completed on `6/6`
  - multi-view reprojection completed on `6/6`
  - entered and completed `EPOCH 0: optimizing 3D Gaussian Splatting`
  - wrote `[EPOCH 1] Saving Gaussians`
  - wrote `[EPOCH 1] Saving Checkpoint`
  - artifacts exist:
    - `/dev_vepfs/rc_wu/edit/CircleEditing/runs/frontier_seed1_grid3_editstrong_211_epoch1_patchstability_tensorart_20260327_205251/point_cloud/iteration_7010/chkpnt7010.pth`
    - `/dev_vepfs/rc_wu/edit/CircleEditing/runs/frontier_seed1_grid3_editstrong_211_epoch1_patchstability_tensorart_20260327_205251/point_cloud/iteration_7010/point_cloud.ply`
- Visual verdict:
  - Front-anchor reference remains: FlowEdit front is more identity-faithful than DNA strong.
  - In this successful run’s debug proxy sheet (`reports/ralph_loop_20260328/patchstability_tensorart_205251_proxies_sheet.png`), several neighbor proxy frames still show weak/fragmented clown transfer and local black-mask artifacts, indicating propagation quality remains the bottleneck even though runtime stability is now confirmed.
- Concrete next move:
  - Stop launcher debugging and run a focused 3-view propagation-quality cycle with FlowEdit baseline versus minimal DNA face-only anchor swap at the frontier-anchor seam, keeping `skip_agt=true`.

## Rolling Summary

- Restore-to-CUDA patch now has direct practical-route evidence under corrected launcher surface and `skip_agt=true`.
- Current bottleneck has shifted from kernel/launcher stability to neighbor propagation quality.
- Next cycle is anchor-quality propagation: keep FlowEdit baseline and test minimal frontier anchor swap (DNA face-only hybrid) without turning AGT back on.

## 2026-03-27T20:54:51Z - Cycle 3: prelaunch record for forced practical 3-view xulf rerun

- Selected task GPU:
  - `CUDA_VISIBLE_DEVICES=4` (idle at launch check; GPUs `6-7` were already occupied and `4-5` were idle)
- Planned experiment run path:
  - `/dev_vepfs/rc_wu/edit/CircleEditing/runs/frontier_seed1_grid3_editstrong_211_epoch1_patchstability_xulf_gpu4_20260327_205451`
- Planned launcher command:
  ```bash
  CUDA_VISIBLE_DEVICES=4 \
  EDITSPLAT_BASE_MODEL_ID=cocktailpeanut/xulf-s \
  FLOWEDIT_SD35_MEDIUM_TURBO_OPEN_MODEL_ID=cocktailpeanut/xulf-s \
  EDITSPLAT_SAM3_DEVICE=cpu \
  EDITSPLAT_TTT3R_FORCE_CPU=1 \
  EDITSPLAT_MAX_GAUSSIANS=70000 \
  EDITSPLAT_MAX_TRAIN_VIEWS=6 \
  /dev_vepfs/rc_wu/envs/editsplat_multimodel_v2/bin/python \
    /dev_vepfs/rc_wu/edit/CircleEditing/runtime/EditSplat_overlay_20260326/sandboxes/20260322_editsplat_ttt3r_flowedit_sam3_4567/scripts/run_sd35_ttt3r_sam3_wrapper.py \
    --model_key sd35-medium-turbo-open \
    --hf_home /dev_vepfs/rc_wu/cache/hf_home_dev02 \
    --adapter_resize_side 512 \
    --adapter_gpu 0 \
    --base_gpu 0 \
    --head_k 6 \
    --depth_mode constant \
    --skip_agt \
    --aux_models_cpu \
    --ttt3r_repo_root /dev_vepfs/rc_wu/edit/TTT3R \
    --ttt3r_checkpoint /dev_vepfs/rc_wu/edit/TTT3R/src/cut3r_512_dpt_4_64.pth \
    --ttt3r_support_views 2 \
    --ttt3r_support_stride 1 \
    --ttt3r_conf_power 1.0 \
    --ttt3r_conf_floor 0.0 \
    --ttt3r_geo_scale 1.0 \
    --ttt3r_edit_min_mass 0.0 \
    --ttt3r_preserve_min_mass 0.0 \
    --ttt3r_adaptive_max_scale 3.2 \
    --ttt3r_schedule_power 1.8 \
    --ttt3r_input_h 256 \
    --ttt3r_input_w 320 \
    --ttt3r_edit_mask_quantile 0.9 \
    --ttt3r_mode velocity \
    --ttt3r_gpu -1 \
    --dump_intermediates \
    --dump_max_per_stage 32 \
    --max_optimizer_steps 1 \
    --optimizer_lr_scale 0.6 \
    --fit_loss_mask_mode initial_edit \
    --fit_loss_mask_quantile 0.75 \
    --fit_loss_mask_bg 0.05 \
    --fit_view_topk -1 \
    --source_path /dev_vepfs/rc_wu/edit/EditSplat/dataset/dataset/face \
    --source_checkpoint /dev_vepfs/rc_wu/edit/EditSplat/sandboxes/20260322_editsplat_ttt3r_flowedit_sam3_4567/runtime/compat_pretrained_face/chkpnt7004.pth \
    --model_path /dev_vepfs/rc_wu/edit/CircleEditing/runs/frontier_seed1_grid3_editstrong_211_epoch1_patchstability_xulf_gpu4_20260327_205451 \
    --eval \
    --epoch 1 \
    --iterations 30000 \
    --debug \
    --target_prompt 'the same man in the same pose and camera framing, same background and clothes, with clear clown makeup: white face paint, a red clown nose, and colorful face paint' \
    --sampling_prompt 'the same man with clown makeup, same framing and identity' \
    --object_prompt face \
    --target_mask_prompt face \
    --filtering_ratio 0.85 \
    --flow_src_prompt 'a photo of a young man' \
    --flow_tar_prompt 'the same man in the same pose and camera framing, same background and clothes, with clear clown makeup: white face paint, a red clown nose, and colorful face paint' \
    --flow_model_key sd35-medium-turbo-open \
    --flow_method flowedit \
    --flow_hf_home /dev_vepfs/rc_wu/cache/hf_home_dev02 \
    --flow_adapter_resize_side 512 \
    --flow_adapter_gpu 0 \
    --flow_steps 24 \
    --flow_n_avg 1 \
    --flow_n_min 0 \
    --flow_n_max 10 \
    --enable \
    --resize 512 \
    --timesteps 28 \
    --n_min 0 \
    --n_max 24 \
    --time_weight one \
    --w_edit 1.1 \
    --w_id 0.15 \
    --src_guidance 1.4 \
    --tar_guidance 7.8 \
    --mask_bg 0.15 \
    --attn_thres 0.1 \
    --k_percent 0.15 \
    --text_guidance_scale 7.4 \
    --image_guidance_scale 1.6 \
    --source_guidance_scale 1.0 \
    --flow_src_guidance_scale 3.2 \
    --flow_tar_guidance_scale 7.8 \
    --flow_seed 211
  ```
- Prelaunch risk note:
  - The active code resolves `FLOWEDIT_SD35_MEDIUM_TURBO_OPEN_MODEL_ID` directly into the SD3 adapter repo id; the cached `cocktailpeanut/xulf-s` snapshot on this host is a `FluxPipeline`, so adapter init may still fail before frontier execution.

## 2026-03-27T21:07:39Z - Operator note: interrupted hybrid-anchor launch (non-runtime evidence)

- Experiment run path:
  - `/dev_vepfs/rc_wu/edit/CircleEditing/runs/frontier_seed1_grid3_editstrong_211_epoch1_anchorhybrid_tensorart_20260327_210648`
- Failure signature:
  - `KeyboardInterrupt` during pretrained pipeline load while monitoring the foreground process:
    - `param = param.to(casting_dtype)`
  - This was operator-caused and does **not** count as runtime evidence for or against the lane.
- Visual verdict:
  - No valid image evidence; the run was interrupted before frontier outputs completed.
- Concrete next move:
  - Relaunch the same hybrid-anchor recipe without touching the process and judge it only from completed logs and saved images.

## 2026-03-27T21:26:39Z - Cycle 4: raw DNA strong frontier-anchor override reached checkpoint but failed visually

- Experiment run path:
  - `/dev_vepfs/rc_wu/edit/CircleEditing/runs/frontier_seed1_grid3_editstrong_211_epoch1_anchorhybrid_tensorart_20260327_211522`
- Launch intent:
  - Use the existing frontier-anchor seam with the smallest available implementation:
    - `EDITSPLAT_FRONTIER_ANCHOR_OVERRIDE_IMAGE=/dev_vepfs/rc_wu/edit/CircleEditing/runs/dnafix_offline_strong211_staticproxy_20260327_171928/debug_intermediates/initial_edit/view000/edited.png`
    - `EDITSPLAT_FRONTIER_ANCHOR_OVERRIDE_MODE=face_on_existing`
    - keep the known-good productive lane:
      - `EDITSPLAT_BASE_MODEL_ID=tensorart/stable-diffusion-3.5-medium-turbo`
      - `FLOWEDIT_SD35_MEDIUM_TURBO_OPEN_MODEL_ID=tensorart/stable-diffusion-3.5-medium-turbo`
      - `skip_agt=true`
- Success evidence:
  - initial editing completed on `6/6`
  - multi-view reprojection completed on `6/6`
  - `EPOCH 0: optimizing 3D Gaussian Splatting` completed
  - wrote `[EPOCH 1] Saving Checkpoint`
  - loaded `iteration 7010` and completed train/test rendering
  - artifacts exist:
    - `/dev_vepfs/rc_wu/edit/CircleEditing/runs/frontier_seed1_grid3_editstrong_211_epoch1_anchorhybrid_tensorart_20260327_211522/point_cloud/iteration_7010/chkpnt7010.pth`
    - `/dev_vepfs/rc_wu/edit/CircleEditing/runs/frontier_seed1_grid3_editstrong_211_epoch1_anchorhybrid_tensorart_20260327_211522/point_cloud/iteration_7010/point_cloud.ply`
- Visual verdict:
  - By eye, the raw full-frame DNA strong override is **not** promotable. The MFG proxy sheet (`reports/ralph_loop_20260328/anchorhybrid_211522_proxies_sheet.png`) is effectively unchanged from the baseline weak-propagation pattern except for a detached clown-face fragment floating off the right side in one downward view, which indicates the override face is misregistered in absolute 2D image space rather than fused onto the frontier anchor seam.
  - The saved train/test render sheets (`reports/ralph_loop_20260328/anchorhybrid_train_renders_vs_gt_20260327_211522.png`, `reports/ralph_loop_20260328/anchorhybrid_test_renders_vs_gt_20260327_211522.png`) show severe black/speckled face collapse instead of usable clown makeup propagation, so this cycle is a visual regression despite clean runtime completion.
- Concrete next move:
  - Do **not** keep using the raw full-frame DNA strong image as the frontier override. The next high-value step is to prepare an aligned face-only DNA override in the frontier-anchor face box, then rerun the same recipe; if that still detaches, switch to the projected anchor-memory helper so covered pixels come from reprojected anchor memory instead of an absolute 2D crop placement.

## Rolling Summary

- Practical 3-view stability is confirmed on the productive `skip_agt=true` lane with the `tensorart` model surface.
- The first frontier-seam DNA hybrid attempt also completed end-to-end, so the remaining blocker is no longer launcher/runtime stability.
- Raw full-frame DNA strong override is visually wrong for this seam: it detaches into a floating face fragment and collapses final renders.
- Highest-value next move is an aligned face-only DNA override asset at the frontier-anchor bbox; if that still misses placement, implement the projected anchor-memory helper and rerun on the same `tensorart` lane.
