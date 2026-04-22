# Frontier Fallback + AGT Bypass Grid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the frontier_seed1 pipeline produce non-black neighbor-face edits reliably, while removing AGT as a blocker and running a 4-GPU grid to validate the fallback.

**Architecture:** Keep the current frontier_seed1 skeleton, add a pure helper layer that detects black-face proxy failure and composes an anchor-based fallback into `MF_image`, and keep AGT off for the main experiment lane. For the AGT/memory lane, add a safe mode boundary so the pipeline either skips AGT cleanly or fails with explicit validation before poisoning the CUDA context.

**Tech Stack:** Python 3.9/3.10, PyTorch, EditSplat runtime overlay, pytest, remote GPU execution on `dev-intern-02`.

---

### Task 1: Add tested fallback helpers

**Files:**
- Modify: `runtime/EditSplat_overlay_20260326/run_editing_flow.py`
- Create: `runtime/EditSplat_overlay_20260326/tests/test_frontier_fallback.py`

- [ ] **Step 1: Write the failing tests**

```python
import torch
from run_editing_flow import _proxy_region_stats, _should_use_frontier_fallback, _compose_anchor_face_fallback


def test_should_use_frontier_fallback_triggers_on_dark_masked_proxy():
    proxy = torch.zeros(1, 3, 8, 8)
    mask = torch.zeros(1, 8, 8)
    mask[:, 2:6, 2:6] = 1.0
    stats = _proxy_region_stats(proxy, mask)
    assert _should_use_frontier_fallback(stats, mean_threshold=0.05, std_threshold=0.02)


def test_compose_anchor_face_fallback_writes_bright_anchor_crop_into_target_mask():
    gt = torch.zeros(1, 3, 8, 8)
    proxy = torch.zeros(1, 3, 8, 8)
    anchor = torch.zeros(1, 3, 8, 8)
    anchor[:, :, 2:6, 2:6] = 1.0
    anchor_mask = torch.zeros(1, 8, 8)
    anchor_mask[:, 2:6, 2:6] = 1.0
    target_mask = torch.zeros(1, 8, 8)
    target_mask[:, 1:7, 1:7] = 1.0
    out = _compose_anchor_face_fallback(proxy, gt, anchor, anchor_mask, target_mask, feather_radius=1)
    assert out[:, :, 2:6, 2:6].mean().item() > 0.4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /dev_vepfs/rc_wu/edit/CircleEditing/runtime/EditSplat_overlay_20260326 && pytest -q tests/test_frontier_fallback.py`
Expected: FAIL with missing helper imports.

- [ ] **Step 3: Write minimal implementation**

```python
# Add pure helper functions near the debug/helper section in run_editing_flow.py:
# - _proxy_region_stats(proxy, mask)
# - _should_use_frontier_fallback(stats, mean_threshold, std_threshold)
# - _compose_anchor_face_fallback(proxy, gt, anchor, anchor_mask, target_mask, feather_radius)
# Keep these helpers tensor-only and side-effect free so pytest can exercise them without running the pipeline.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /dev_vepfs/rc_wu/edit/CircleEditing/runtime/EditSplat_overlay_20260326 && pytest -q tests/test_frontier_fallback.py`
Expected: PASS.

- [ ] **Step 5: Record validation command**

```bash
cd /dev_vepfs/rc_wu/edit/CircleEditing/runtime/EditSplat_overlay_20260326
pytest -q tests/test_frontier_fallback.py
```

### Task 2: Wire fallback into frontier_seed1 and make AGT safe to bypass

**Files:**
- Modify: `runtime/EditSplat_overlay_20260326/run_editing_flow.py`
- Modify: `runtime/EditSplat_overlay_20260326/scene/gaussian_model.py`

- [ ] **Step 1: Add the failing safety/behavior assertions in tests**

```python
def test_compose_anchor_face_fallback_leaves_proxy_unchanged_when_target_mask_empty():
    proxy = torch.rand(1, 3, 8, 8)
    gt = torch.rand(1, 3, 8, 8)
    anchor = torch.rand(1, 3, 8, 8)
    empty = torch.zeros(1, 8, 8)
    out = _compose_anchor_face_fallback(proxy, gt, anchor, empty, empty, feather_radius=1)
    assert torch.allclose(out, proxy)
```

- [ ] **Step 2: Run test to verify it fails before wiring**

Run: `cd /dev_vepfs/rc_wu/edit/CircleEditing/runtime/EditSplat_overlay_20260326 && pytest -q tests/test_frontier_fallback.py -k empty`
Expected: FAIL until the edge case is handled.

- [ ] **Step 3: Write minimal implementation**

```python
# In run_editing_flow.py, cache the frontier anchor image and anchor gt mask.
# In the neighbor branch, compute proxy stats over the frontier target mask.
# If the stats fall below thresholds, replace the masked region in MF_image with the composed anchor fallback.
# Add debug metadata such as fallback_triggered, proxy_mean, proxy_std, fallback_mode.
# Keep AGT off in the main experiment lane via EDITSPLAT_SKIP_AGT=1.
# In gaussian_model.py, add cheap tensor-shape/device assertions at apply_weights() entry so AGT fails early instead of poisoning CUDA when someone enables it later.
# Also guard the epoch-save prints in run_editing_flow.py so epoch=0 smoke runs do not crash with UnboundLocalError.
```

- [ ] **Step 4: Run targeted validation**

Run: `cd /dev_vepfs/rc_wu/edit/CircleEditing/runtime/EditSplat_overlay_20260326 && pytest -q tests/test_frontier_fallback.py`
Expected: PASS.

- [ ] **Step 5: Run a CPU-only import smoke**

Run: `cd /dev_vepfs/rc_wu/edit/CircleEditing/runtime/EditSplat_overlay_20260326 && python3 -m py_compile run_editing_flow.py scene/gaussian_model.py`
Expected: exit code 0.

### Task 3: Launch 4-way frontier grid with AGT disabled

**Files:**
- Modify: `runtime/EditSplat_overlay_20260326/run_editing_flow.py`
- Create: `runs/frontier_*` outputs only

- [ ] **Step 1: Define the grid**

```bash
# Run A: baseline no fallback, seed 211
# Run B: fallback threshold 0.08, feather 9, seed 211
# Run C: fallback threshold 0.12, feather 9, seed 211
# Run D: fallback threshold 0.12, feather 15, seed 211
```

- [ ] **Step 2: Launch on 4 GPUs**

Run pattern:
```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=/dev_vepfs/rc_wu/edit/CircleEditing/runtime/diffgs_patch_20260326 EDITSPLAT_BASE_MODEL_ID=cocktailpeanut/xulf-s EDITSPLAT_MFG_MODE=frontier_seed1 EDITSPLAT_MASK_BACKEND=sam3 EDITSPLAT_SAM3_DEVICE=cpu EDITSPLAT_MAX_TRAIN_VIEWS=3 EDITSPLAT_MAX_GAUSSIANS=70000 EDITSPLAT_DEPTH_MODE=constant EDITSPLAT_SKIP_AGT=1 EDITSPLAT_DUMP_INTERMEDIATES=1 python <wrapper> ...
```

- [ ] **Step 3: Verify launch**

Run: `nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv`
Expected: 4 experiment processes mapped onto GPUs 4,5,6,7 only.

- [ ] **Step 4: Early triage**

Run: inspect each `run.log`, `debug_intermediates/mfg_edit/*/stats.json`, and rendered contact sheet; compare masked face brightness and visible edit quality.

- [ ] **Step 5: Save summary**

```bash
# Write a markdown note under /dev_vepfs/rc_wu/edit/CircleEditing/reports/
# with run paths, fallback settings, and visual conclusions.
```
