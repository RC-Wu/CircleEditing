# Frontier Grid3 + Memory Probe Report (2026-03-27)

## Summary

This round had two parallel goals:
1. keep pushing the productive `frontier_seed1 + skip_agt + fallback` line with more experiments
2. narrow the real root cause of the remaining diffGS backward crash

Main conclusions:
- The low-level 3DGS blocker is still the same diffGS backward kernel failure. Under `CUDA_LAUNCH_BLOCKING=1` it consistently collapses to `CUDA error: an illegal memory access was encountered` in `diff_gaussian_rasterization.__init__.py` at `_C.rasterize_gaussians_backward(*args)`.
- Two practical workaround probes failed:
  - reducing `EDITSPLAT_MAX_GAUSSIANS` from `70000` to `20000`
  - enabling both `--convert_SHs_python` and `--compute_cov3D_python`
- On the productive epoch-0 lane, the first batch of fine-grained frontier tweaks (`q93`, slightly stronger preserve, 4-view support) had only tiny output changes. They do not address the main failure mode.
- `flow_seed` still matters a lot. `seed131` under the tighter face-gated recipe changes the result visibly, but it degrades cross-view consistency versus `seed211`.
- Stronger guidance changes are the first non-seed levers that visibly move the current fallback-only pipeline. They still do not solve the core problem, but they matter more than the small TTT3R/SAM3 tweaks.

## Memory / Kernel Debugging

### Reproduction setup

Stable minimal repro:
- host: `dev-intern-01`
- 1 visible GPU
- `CUDA_LAUNCH_BLOCKING=1`
- `EDITSPLAT_MAX_TRAIN_VIEWS=1`
- `skip_agt=true`
- `epoch=1`
- first backward only

Runs:
- `/dev_vepfs/rc_wu/edit/CircleEditing/runs/frontier_seed1_debug_blocking_v1_20260327`
- `/dev_vepfs/rc_wu/edit/CircleEditing/runs/frontier_seed1_debug_mg20k_blocking_fb_20260327_150440`
- `/dev_vepfs/rc_wu/edit/CircleEditing/runs/frontier_seed1_debug_pyshcov_blocking_fb_20260327_151106`

### What was ruled out

1. `invalid argument` was not the real root signature.
   With launch blocking enabled, the crash becomes a consistent `illegal memory access`.

2. It is not just a point-count issue.
   - `70000` gaussians crashes
   - `20000` gaussians also crashes

3. It is not isolated to the SH / covariance-precompute branch.
   Even with both `--convert_SHs_python` and `--compute_cov3D_python`, the same kernel still crashes in backward.

### Current best debugging conclusion

The remaining blocker is still the core diffGS backward path itself, not AGT, not the current frontier fallback wrapper logic, and not a simple ?too many gaussians? resource issue.

## Experiment Batch A: Fine Frontier Tweaks

Completed runs:
- `/dev_vepfs/rc_wu/edit/CircleEditing/runs/frontier_seed1_grid2_vel211_q93_sam095_fb_20260327_150439`
- `/dev_vepfs/rc_wu/edit/CircleEditing/runs/frontier_seed1_grid2_vel211_pres003_p108_fb_20260327_150439`
- `/dev_vepfs/rc_wu/edit/CircleEditing/runs/frontier_seed1_grid2_vel211_v4_supp3_fb_20260327_150439`

Compared against baseline:
- `/dev_vepfs/rc_wu/edit/CircleEditing/runs/frontier_seed1_grid_vel_seed211_fb_20260327_103126`

Visual review assets:
- `/dev_vepfs/rc_wu/edit/CircleEditing/runs/view001_facecrop_grid2_compare_20260327.jpg`
- `/dev_vepfs/rc_wu/edit/CircleEditing/runs/view002_facecrop_grid2_compare_20260327.jpg`

Quantitative notes:
- all neighbor views still have `frontier_fallback_triggered=true`
- all neighbor views still have `frontier_proxy_mean=0.0`
- `mfg_output.mean` changes only slightly relative to baseline
- image-level differences exist, but the visible effect is small

Conclusion:
- these fine-grained tweaks are too weak to solve the current problem
- under the present pipeline, once the neighbor face is fully fallback-driven, these small mask / support / preserve changes do not produce the kind of qualitative shift we need

## Experiment Batch B: Seed Probe Under Tighter Face Gating

Completed run:
- `/dev_vepfs/rc_wu/edit/CircleEditing/runs/frontier_seed1_grid2_vel131_q93_sam095_fb_20260327_151403`

Visual review assets:
- `/dev_vepfs/rc_wu/edit/CircleEditing/runs/view000_seed_compare_20260327.jpg`
- `/dev_vepfs/rc_wu/edit/CircleEditing/runs/view001_seed_compare_20260327.jpg`
- `/dev_vepfs/rc_wu/edit/CircleEditing/runs/view002_seed_compare_20260327.jpg`

Manual conclusion:
- `seed131` still changes the result much more than the fine frontier tweaks
- the change is not a clean win
- front view drifts stylistically and the side views become less consistent than `seed211`
- for the current fallback-only path, `seed211` remains the more reliable default

## Experiment Batch C: Strong Guidance Levers

Completed runs:
- `/dev_vepfs/rc_wu/edit/CircleEditing/runs/frontier_seed1_grid3_idstable_211_20260327_153432`
- `/dev_vepfs/rc_wu/edit/CircleEditing/runs/frontier_seed1_grid3_editstrong_211_20260327_153432`

Visual review assets:
- `/dev_vepfs/rc_wu/edit/CircleEditing/runs/view000_grid3_compare_20260327.jpg`
- `/dev_vepfs/rc_wu/edit/CircleEditing/runs/view001_grid3_compare_20260327.jpg`
- `/dev_vepfs/rc_wu/edit/CircleEditing/runs/view002_grid3_compare_20260327.jpg`

Design intent:
- `idstable`: higher identity / image guidance, softer edit guidance, stronger preservation bias
- `editstrong`: stronger edit guidance, weaker preservation bias

Manual conclusion:
- these two runs finally move the outputs more clearly than the fine frontier tweaks
- `idstable` is a bit more conservative and identity-preserving
- `editstrong` pushes the makeup slightly harder
- the effect is still incremental, not fundamental
- both runs remain in the same fallback-dominated regime

## Overall Interpretation

The data from this round points to two strong conclusions:

1. The 3DGS backward bug should be treated as a separate low-level kernel issue.
   It is still unresolved after isolating point count and SH/cov branches.

2. The bigger modeling bottleneck is now the neighbor-view generation strategy itself.
   In the productive lane, the neighbor views are still driven by:
   - `proxy_rgb == 0`
   - fallback anchor-face compositing
   - per-view diffusion refinement

That makes the pipeline:
- too insensitive to small geometry-aware tuning
- too sensitive to diffusion seed / global guidance
- weak at ear / hair / occlusion consistency

## Practical Default After This Round

Best current default for stable editing success:
- `velocity`
- `seed211`
- `skip_agt=true`
- `frontier fallback enabled`

If a stronger visible change is desired without changing the paradigm yet:
- prefer the `grid3`-style stronger guidance changes over the `grid2` fine TTT3R/SAM3 tweaks

## Recommended Next Step

The evidence from this round supports moving to the next paradigm step instead of spending many more cycles on tiny frontier-fallback tweaks:
- keep the front-view anchor generation
- stop relying on zero-proxy fallback as the main neighbor-view content source
- replace it with a real anchor-conditioned multi-view completion stage around the anchor view
- continue to skip AGT unless diffGS is independently repaired
- treat the diffGS backward issue as a kernel debugging track, not the main productive editing track
