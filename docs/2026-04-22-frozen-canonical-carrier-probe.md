# 2026-04-22 Frozen Canonical Carrier Probe

## Goal
Validate the simplest approved pivot: keep the source scene frozen, inject teacher-visible edit residuals into a canonical carrier, and avoid more complex bidirectional coupling.

## What Changed
- `runtime/EditSplat/utils/canonical_edit_field.py`
  - added `build_frozen_canonical_carrier(...)`
  - the carrier now uses a weighted blend of teacher residual (`initial_edit - source`) and flow proxy residual (`proxy - source`)
- `runtime/EditSplat/utils/ttt3r_elite_blite.py`
  - upgraded `SourceCanonicalPrior` from stats-only bookkeeping to reusable per-view tensor priors
  - added `build_source_canonical_prior_mask(...)`
- `.legacy_pre_migration/.../run_sd35_ttt3r_sam3_wrapper.py`
  - `B-lite` path now consumes frozen teacher residuals plus reusable prior masks instead of only the current proxy residual
- queue manifests now include:
  - `EDITSPLAT_FROZEN_CARRIER_TEACHER_WEIGHT=0.72`
  - `EDITSPLAT_CANONICAL_PRIOR_SUPPORT_FLOOR=0.08`
  - `EDITSPLAT_CANONICAL_PRIOR_CONFIDENCE_FLOOR=0.12`

## Verification
- `py_compile` passed for the updated utils, wrapper, and manifest builder.
- `unittest` passed:
  - `tests.test_canonical_edit_field`
  - `tests.test_carrier_baseline`

## Runtime State
- GPU check on `dev-intern-02` at implementation time:
  - GPU0: occupied
  - GPU1: occupied
  - GPU2: still holds an existing ~13.9 GiB context
  - GPU3-7: occupied
- No truly free GPU was available under the user's latest constraint "only use a free card".
- Result: code path is ready, but the first remote sanity launch is deferred until a genuinely free GPU appears.

## Ready-to-Run Entry
When one card is actually free, use the sequential single-card runner under:
- `.legacy_pre_migration/runtime/EditSplat_overlay_20260326/sandboxes/20260322_editsplat_ttt3r_flowedit_sam3_4567/scripts/run_wave18_gpu3_five_rounds.py`

Recommended next invocation shape:
```bash
/dev_vepfs/rc_wu/envs/editsplat_multimodel_v2/bin/python   .legacy_pre_migration/runtime/EditSplat_overlay_20260326/sandboxes/20260322_editsplat_ttt3r_flowedit_sam3_4567/scripts/run_wave18_gpu3_five_rounds.py   --slot-gpu <FREE_GPU>   --wave-name 20260422_wave20_frozen_carrier_probe
```
