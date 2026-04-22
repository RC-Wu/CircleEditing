# Frontier Debug + DNAEdit Note (2026-03-28)

## Scope

This round focused on two blockers:

1. the remaining diffGS crash on the `epoch=1` 3DGS path
2. whether `SD3.5 + DNAEdit` can produce a meaningfully different front anchor than the current FlowEdit line

## Key Results

### 1) The current crash root cause was not just a generic kernel mystery

With raster debug enabled, the first failing minimal repro wrote:

- `/dev_vepfs/rc_wu/edit/CircleEditing/runs/frontier_seed1_debug_rasterdbg_clone_offline_dev02_20260327_165921/diffgs_debug/forward_001.json`

That dump showed the rasterizer forward inputs were mixed-device and mostly on CPU:

- `means3D/opacities/scales/rotations/sh/viewmatrix/projmatrix/campos/bg` were all on `cpu`
- empty optional tensors were on `cuda:0`

This directly explained the `illegal memory access` in rasterization.

### 2) Fix: explicit restore-to-CUDA rehydration

Code changes:

- `runtime/EditSplat_overlay_20260326/scene/gaussian_model.py`
  - added `_as_parameter_on_device`
  - added `_move_optimizer_state_to_device`
  - updated `restore(..., device=...)` to rebuild trainable parameters on the target device
  - moved optimizer state tensors onto the same device
  - made `training_setup()` allocate accumulators on `self.get_xyz.device`
- `runtime/EditSplat_overlay_20260326/run_editing_flow.py`
  - updated the checkpoint restore callsite to pass `device=torch.device("cuda" if torch.cuda.is_available() else "cpu")`

Verification:

- unit tests passed:
  - `tests.test_raster_safety`
  - `tests.test_gaussian_restore_device`
  - `tests.test_dnaedit_runtime_resolution`

### 3) The patched debug run now completes the previously crashing epoch-1 path

Validated run:

- `/dev_vepfs/rc_wu/edit/CircleEditing/runs/debugfix_rasterdbg_clone_offline_20260327_171928`

Evidence:

- it wrote both:
  - `diffgs_debug/forward_001.json`
  - `diffgs_debug/backward_001.json`
- the run reached:
  - `Saving Gaussians`
  - `Saving Checkpoint`
  - `point_cloud/iteration_7005/chkpnt7005.pth`

Most importantly, the debug dumps now show all relevant tensors on `cuda:0` in both forward and backward.

So the minimal `epoch=1` 3DGS optimization case no longer crashes after the restore-device fix.

## DNAEdit Integration

### 1) First blocker: mode mismatch

The first attempt failed because current wrapper logic only allows:

- `dnaedit` with `--ttt3r_mode static_proxy`

It does not accept `velocity`.

### 2) Second blocker: runtime path mismatch

The next attempt failed because `flowedit_multimodel/src/core_backend.py` only searched for the DNAEdit runtime under the migrated CircleEditing runtime tree, while the actual DNAEdit code still lived under the old EditSplat sandbox.

Fix:

- `runtime/EditSplat_overlay_20260326/flowedit_multimodel/src/core_backend.py`
  - added `_resolve_dnaedit_runtime_root`
  - search order:
    - `EDITSPLAT_DNAEDIT_RUNTIME_ROOT` if set
    - current project-local sandbox
    - fallback to `/dev_vepfs/rc_wu/edit/EditSplat/sandboxes/20260309_rfedit_dnaedit_flowalign_snredit/runtime/DNAEdit_code_http11`

### 3) After the path fix, DNAEdit runs successfully

Successful run:

- `/dev_vepfs/rc_wu/edit/CircleEditing/runs/dnafix_offline_strong211_staticproxy_20260327_171928`

Evidence:

- the run wrote:
  - `debug_intermediates/initial_edit/view000/edited.png`
  - `debug_intermediates/mfg_edit/view000/mfg_output.png`
- runtime summary:
  - `backend: wrapper_core_backend_dnaedit_static_proxy`
  - `flow_method: dnaedit`
  - `mode: static_proxy`

## Visual Read

### FlowEdit front-only anchor

Run:

- `/dev_vepfs/rc_wu/edit/CircleEditing/runs/frontonly_offline_flowedit_editstrong211_20260327_165921`

Observation:

- the front anchor is effectively the same as the earlier `grid3_editstrong` front edit
- making it `front-only` did not create a stronger anchor by itself

### DNAEdit strong static-proxy anchor

Run:

- `/dev_vepfs/rc_wu/edit/CircleEditing/runs/dnafix_offline_strong211_staticproxy_20260327_171928`

Observation from manual image inspection:

- clown makeup is stronger and more regular/symmetric than the FlowEdit anchor
- but geometry / identity drift is noticeably worse
- head orientation and shirt/collar appearance drift away from the source view

Interpretation:

- `DNAEdit` is a valid strong-anchor candidate
- but if promoted into the multi-view route, it will need stronger geometry/identity control than the current FlowEdit anchor

## Practical Takeaway

1. The main 3DGS crash on the patched minimal repro was fixed by moving restored Gaussian parameters and optimizer state back onto CUDA before rasterization.
2. `DNAEdit` is now runnable in the current migrated project, but only on the `static_proxy` branch right now.
3. `DNAEdit strong` produces a more aggressive front clown edit than FlowEdit, but also causes more pose/appearance drift.
4. The next productive comparison should be:
   - current FlowEdit front anchor
   - DNAEdit strong front anchor
   - same downstream neighbor propagation route


