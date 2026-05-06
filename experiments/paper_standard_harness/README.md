# Paper-Standard 3D Editing Harness

This folder contains diagnostic harnesses created after the R1-R5 proxy image edits failed recent 3D editing paper standards.

The harnesses enforce basic output contracts that DGE/EditSplat/GaussCtrl-like methods require:

1. Build or load an explicit 3D representation.
2. Apply the edit in object/geometry space, not by independent per-view PNG compositing.
3. Render source, mask, guidance, edited, and held-out views from the same representation.
4. Save visual artifacts for manual inspection.
5. Clearly mark all toy/procedural outputs as diagnostics, not final research success.

## Scripts

- `run_explicit_3d_edit.py`: surfel representation harness. R1-R2 face runs are rejected; R3 mug run is a cleaner contract test but still toy quality.
- `run_mesh_texture_edit.py`: mesh/texture contract harness. R4 is visually cleaner and saves OBJ/MTL/texture, but it is procedural rather than a reconstructed scene.

## Example runs

```bash
/mnt/beegfs/ruocheng/circleediting_sigasia_20260506/envs/cexp/bin/python   experiments/paper_standard_harness/run_explicit_3d_edit.py   --out-root /mnt/beegfs/ruocheng/circleediting_sigasia_20260506/runs/paper_standard_harness   --tag r3_paper_gate_mug --scene paper_gate_mug --points 220000 --size 720

/mnt/beegfs/ruocheng/circleediting_sigasia_20260506/envs/cexp/bin/python   experiments/paper_standard_harness/run_mesh_texture_edit.py   --out-root /mnt/beegfs/ruocheng/circleediting_sigasia_20260506/runs/paper_standard_harness   --tag r4_fast_mesh_texture --size 720
```

## Visual verdict so far

- R1-R2 surfel face: rejected. Too sparse/noisy, weak semantic edit.
- R3 surfel mug: passes representation/novel-view contract but remains visibly toy and point-rendered.
- R4 mesh/texture mug: cleaner and more stable, but still procedural; it is a gate harness only.

The next non-toy path must use a real reconstructed object/scene or a real feed-forward reconstruction bootstrap.
