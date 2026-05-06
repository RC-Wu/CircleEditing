# 2026-05-06 DGE/EditSplat Paper-Standard Recovery Plan

## Problem

R1-R5 proxy edits are visually and methodologically below recent 3D editing papers. They are rejected baselines.

## Target

Build a paper-standard minimum viable experiment:

1. An explicit 3D representation exists before and after editing.
2. Multi-view guidance is generated and inspected before fitting.
3. Final output is rendered from the updated 3D representation, including held-out/novel views.
4. Visual quality is judged against DGE/EditSplat/VcEdit/TIP-Editor-style standards.

## Immediate steps

1. Clone and inspect DGE under this sandbox, without committing third-party code unless explicitly needed.
2. Check whether DGE provides demo data or scripts that can run within the 80GB budget.
3. If no demo scene/pretrained GS is available, create a small explicit scene pipeline inside CircleEditing:
   - small camera orbit;
   - explicit point/Gaussian-like object representation;
   - consistent edited guidance;
   - representation update;
   - novel-view render sheet.
4. Keep R5 only as a rejected/fallback baseline.

## Hard constraints

- Sprint root: `/mnt/beegfs/ruocheng/circleediting_sigasia_20260506`.
- Storage: under 80GB.
- GPU: only 6/7 if GPU is needed.
- No uncontrolled checkpoint downloads.
