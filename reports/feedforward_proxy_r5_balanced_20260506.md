# Feed-forward Proxy R5 Balanced Result 2026-05-06

## Context

This run was executed during the urgent new-A100 CircleEditing sprint. The GitHub repository did not contain the full executable historical EditSplat stack, so the experiment focused on a small, reproducible proxy-first visual loop using committed review artifacts.

## Run

- Remote sprint root: `/mnt/beegfs/ruocheng/circleediting_sigasia_20260506`
- Repo: `repos/CircleEditing`
- Environment: `envs/cexp`
- GPU binding: `CUDA_VISIBLE_DEVICES=6`
- Final run root: `/mnt/beegfs/ruocheng/circleediting_sigasia_20260506/runs/feedforward_proxy/20260506_122412_r5_balanced`
- Final variant: `adaptive_final_balanced`
- Heavy checkpoints downloaded: none
- Sprint storage after run: about `2.5G`

## Visual Result

Committed visual artifacts:

- `assets/review/feedforward_proxy_r5_balanced_20260506/final_contact_sheet.jpg`
- `assets/review/feedforward_proxy_r5_balanced_20260506/all_variants_contact_sheet.jpg`
- `assets/review/feedforward_proxy_r5_balanced_20260506/summary.json`

Human visual inspection conclusion:

- R1 patch/warp variants showed that patch transfer can remove the old black-face failure but creates frontal-face paste artifacts.
- R2 semantic mask-coordinate variants removed most frontal-paste behavior but became too blurry/gray.
- R3 crisp semantic variants made the edit readable; `semantic_crisp` was good for normal neighbor view and `semantic_repaired_crisp` was good for the dark side proxy.
- R4 adaptive variants switched behavior by target-view condition and reached a dirty but usable fallback.
- R5 `adaptive_final_balanced` is the selected result. It keeps all three views visibly edited, preserves background/clothes, and avoids black-face collapse.

## Limitations

This is not a complete FlowEdit+TTT3R/EditSplat reconstruction. The side view is a repaired proxy face, not a true feed-forward 3D reconstruction. The method is a pragmatic fallback demonstrating that a proxy-first semantic propagation branch can produce an inspectable multi-view edit candidate under tight constraints.


## Paper-standard reassessment

After comparing against recent 3D editing papers, this result is rejected as a final method. It should be kept only as a debugging/fallback baseline. It fails the core paper-level requirement that the edit be represented in and rendered from an updated 3D representation. The next experiment must follow a DGE/EditSplat-like pipeline: consistent multi-view guidance followed by actual 3DGS/geometry fitting and held-out render inspection.

## Next Step

Promote the same output contract to a real feed-forward geometry backend: replace pseudo-depth/support-mask proxy with CUT3R/TTT3R geometry and keep `adaptive_final_balanced` as a fallback completion layer for dark/invalid projected content.
