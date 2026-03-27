# diffGS Debug Harness

This directory holds a tiny repro for the `diff_gaussian_rasterization` illegal-memory path.

The harness does one synthetic forward/backward pass on a small Gaussian batch with a fixed camera and a deterministic target image. It is meant to stay isolated from the main runtime code.

## Files

- `probe_diffgs_harness.py`: CLI entry point for the repro.

## Usage

CircleEditing overlay:

```powershell
python F:/InformationAndCourses/Code/CircleEditing/runtime/debug_harness/probe_diffgs_harness.py `
  --source-root F:/InformationAndCourses/Code/CircleEditing `
  --gaussians 64 `
  --width 64 `
  --height 64 `
  --launch-blocking
```

Original EditSplat tree:

```powershell
python F:/InformationAndCourses/Code/CircleEditing/runtime/debug_harness/probe_diffgs_harness.py `
  --source-root /dev_vepfs/rc_wu/edit/EditSplat `
  --gaussians 4096 `
  --width 384 `
  --height 384 `
  --launch-blocking
```

If you have a separate built wheel or DSA build, prepend its directory with `--extra-path`:

```powershell
python F:/InformationAndCourses/Code/CircleEditing/runtime/debug_harness/probe_diffgs_harness.py `
  --source-root /dev_vepfs/rc_wu/edit/EditSplat `
  --extra-path /path/to/your/diff_gaussian_rasterization_build
```

## Notes

- The harness does not depend on the repo's camera classes or training pipeline.
- It expects CUDA for the actual rasterizer pass.
- `--launch-blocking` is useful when you want the traceback to land closer to the failing kernel.
- `DIFFGS_SOURCE_ROOT` or `EDITSPLAT_ROOT` can be used instead of `--source-root` if you prefer an environment variable.
