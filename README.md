# CircleEditing

CircleEditing is the working repository for the multi-object synchronized 3D editing line.

Current contents:

- `runtime/EditSplat/`: locally mirrored runtime files pulled from the live EditSplat branch and patched in this session.
- `assets/review/`: local visual review artifacts copied back from remote runs.
- `reports/`: experiment reports and handoff notes.

Current focus:

1. Replace global MFG-first scheduling with a key-view / frontier-expansion pipeline.
2. Push SAM3 signals earlier into local MFG and local Gaussian optimization.
3. Keep TTT3R active as a light geometry prior instead of the main driver.
4. Make every run produce human-checkable visual artifacts before scaling up.

## A-Line Semantic GS Guidance

The current A-line adds a smaller bridge before reopening full AGT:

- `EDITSPLAT_ENABLE_SEMANTIC_GS_GUIDANCE=1` enables semantic loss gating during 3DGS optimization.
- `EDITSPLAT_SEMANTIC_BG_WEIGHT` controls how much background still contributes to the loss.
- `EDITSPLAT_SEMANTIC_COLOR_SCALE` and `EDITSPLAT_SEMANTIC_POSITION_SCALE` split color/opacity guidance from geometry guidance.
- `EDITSPLAT_SEMANTIC_FREEZE_GEOMETRY=1` forces the semantic guidance path to keep position-side updates at zero.

This path is opt-in and intentionally does not re-enable the full AGT path by default.
