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

## 2026-04-23 Mirror Recovery And Relaunch Attempt
- Development machine direct access to `huggingface.co` still timed out, and the PC reverse proxy port `17890` was not listening.
- `hf-mirror.com` was directly reachable from `dev-intern-02`; the CUT3R release file resolved successfully via:
  - `https://hf-mirror.com/datasets/zhangify/CUT3R_release/resolve/main/cut3r_512_dpt_4_64.pth?download=true`
- Recovered checkpoint to the canonical soft-link target:
  - `/dev_vepfs/rc_wu/cache/models/ttt3r/cut3r_512_dpt_4_64.pth`
  - size: `3173761006` bytes (`~3.0G`)
- Relaunch status:
  - launcher and overlay-wrapper fixes are already in `codex/20260422-canonical-carrier-freeze`
  - a fresh `GPU0` relaunch was attempted after the checkpoint recovery
  - before the run could safely proceed, `GPU0` and `GPU2` became occupied by other jobs (`~42.2 GiB`, `100% util`)
  - the new wave runner was terminated immediately to respect the free-GPU-only constraint
- Net effect:
  - the network / checkpoint blocker is resolved
  - the only remaining blocker is waiting for a truly free allowed GPU before restarting the five-round probe

