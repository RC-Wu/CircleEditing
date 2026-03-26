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
