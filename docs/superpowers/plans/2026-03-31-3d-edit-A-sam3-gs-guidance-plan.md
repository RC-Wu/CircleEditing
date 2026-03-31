# 3D Edit A SAM3 GS Guidance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe, opt-in path that lets SAM3 support masks influence Gaussian color and opacity updates without reopening the full AGT path.

**Architecture:** Introduce a small pure-Python semantic-guidance helper that merges support masks into a per-view guidance plan, then wire `run_editing_flow.py` to use that plan when building the mask passed into `gaussians.apply_grad_mask(...)`. Keep geometry frozen by default and make the feature fully opt-in with environment variables so the current stable lane is unchanged unless explicitly enabled.

**Tech Stack:** Python, PyTorch, pytest-style unit tests, CircleEditing runtime/EditSplat

---

### Task 1: Add Pure Semantic-Guidance Helpers

**Files:**
- Create: `runtime/EditSplat/utils/semantic_guidance.py`
- Create: `tests/test_semantic_guidance.py`

- [ ] Add a failing unit test for mask fusion behavior.
- [ ] Add a failing unit test for color-vs-position guidance scaling.
- [ ] Implement minimal pure helper functions to make those tests pass.
- [ ] Run the focused tests and keep them green.

### Task 2: Wire A Guidance Into `run_editing_flow.py`

**Files:**
- Modify: `runtime/EditSplat/run_editing_flow.py`
- Test: `tests/test_semantic_guidance.py`

- [ ] Add environment parsing for opt-in semantic GS guidance.
- [ ] Use the helper to combine support masks with the existing selected mask path.
- [ ] Ensure the new path can drive `apply_grad_mask(..., l_color=..., l_position=...)` while leaving the current stable default unchanged.
- [ ] Run the focused tests again.

### Task 3: Add One Small Runtime-Safe Regression Check

**Files:**
- Modify: `runtime/EditSplat/run_editing_flow.py`
- Modify: `tests/test_semantic_guidance.py`

- [ ] Add a test that proves the default configuration is a no-op.
- [ ] Add a test that proves semantic support can increase color guidance while keeping position guidance zeroed.
- [ ] Run the focused tests again.

### Task 4: Document A Controls

**Files:**
- Modify: `README.md`

- [ ] Document the new env flags and the intended “quick win only” scope.
- [ ] Note that AGT remains disabled by default and this path is meant to be the safer bridge.

