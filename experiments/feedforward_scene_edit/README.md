# Feed-forward Scene Edit Proxy Experiment

This folder contains a compact, training-free experiment for the urgent 2026-05-06 CircleEditing sprint.

The GitHub repo currently contains a compact mirror of the historical EditSplat work, not a full executable checkout. In particular, the full renderer, argument stack, dataloader, and several utilities are absent. Before spending hours rebuilding FLUX/CUT3R/TTT3R dependencies, this experiment checks the simplest useful question:

Can an object-centric reconstruction/proxy-first scaffold propagate a successful anchor edit into neighboring views without black-fill collapse?

## Inputs

By default the script uses the committed visual review case:

`assets/review/frontier_seed1_constdepth_fixmask_dev01_20260326_123724`

That case has:

- `view000_input` and `view000_initial_edit`: the anchor source/edit pair.
- neighbor inputs or proxies.
- support masks from the previous frontier/SAM attempt.

## Run

From the repo root:

```bash
CUDA_VISIBLE_DEVICES=6 /mnt/beegfs/ruocheng/circleediting_sigasia_20260506/envs/cexp/bin/python \
  experiments/feedforward_scene_edit/run_proxy_edit.py \
  --repo-root . \
  --out-root /mnt/beegfs/ruocheng/circleediting_sigasia_20260506/runs/feedforward_proxy \
  --tag r1
```

The output directory contains per-variant images, pseudo-depth/mask debug images, `summary.json`, `visual_notes.md`, and `all_variants_contact_sheet.jpg`.

## Current variants

- `meanstd_color`: masked mean/std color transfer from the anchor edit.
- `lab_chroma`: target luminance with anchor-edit chroma statistics.
- `delta_warp`: resize/warp the anchor edit delta into the target mask box.
- `patch_warp`: resize/warp the anchor edited object crop into the target mask box.
- `hybrid_proxy`: mix patch warp, color transfer, and delta transfer.
- `seamless_clone`: OpenCV mixed seamless clone with the support mask.
- `semantic_paint`: normalized mask-coordinate procedural clown makeup, avoiding frontal face paste.
- `semantic_patchbase`: patch fill for black proxy faces, followed by semantic paint.
- `semantic_soft`: lower-strength semantic paint for cleaner boundaries.
- `semantic_crisp`: sharper procedural makeup with weaker gray foundation.
- `semantic_repaired_crisp`: patch repair for dark proxy interiors plus sharper procedural makeup.
- `adaptive_final`: identity-preserving semantic edit for normal views and patch-repaired semantic edit for dark proxy views.
- `adaptive_final_bold`: stronger version of the adaptive final candidate.
- `adaptive_final_clean`: conservative normal-view edit plus repaired side-proxy edit.
- `adaptive_final_balanced`: final default candidate balancing identity preservation and visible edit strength.

## Scope limits

This is not yet CUT3R/TTT3R. It is a deliberately small proxy experiment. A positive result means the reconstruction-first direction is worth promoting to a real feed-forward geometry backend. A negative result means the bottleneck is likely anchor content or mask/view correspondence, not just the old diffGS renderer failure.
