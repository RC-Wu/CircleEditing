# Paper-Standard Reset Report - 2026-05-06

## Summary

The current visual effect is far below recent 3D editing papers. The earlier proxy line should be treated as rejected diagnostics, not success.

## Visual inspections completed

- R2 face surfel harness: inspected train and held-out contact sheets. It has a persistent representation, but the face is toy-like, noisy, and the edit reads as colored blobs.
- R3 dense mug surfel harness: inspected train/held-out sheets and a single center view. It is more consistent but visibly point-rendered and still procedural.
- R4 mesh/texture harness: inspected train/held-out sheets and center view. It is clean and stable, but it is a procedural mug and therefore not a real 3D editing result.

## Paper-standard conclusion

Recent papers require at least one of:

- multi-view-consistent guidance followed by direct 3DGS fitting (DGE/GaussCtrl/EditSplat);
- attention/progressive localization of target Gaussians (GaussianEditor/GSEditPro/InterGSEdit);
- native Gaussian attribute changes or variation prediction (VF-Editor direction).

Our current artifacts do not satisfy these requirements on a real reconstructed scene.

## Immediate next step

Proceed to a real-input bootstrap: convert available anchor edit assets into a persistent 2.5D/mesh/GS-ready representation, render held-out views, and use the visual failure modes to decide whether to invest in full DGE/3DGS reconstruction under the 80GB cap.


## Real-anchor bootstrap R5/R6

### R5 `20260506_133913_r5_real_anchor_bootstrap`

- Output: `bootstrap_representation.npz`, `bootstrap_face_patch.ply`, train/held-out contact sheets, held-out turntable.
- Visual verdict: rejected. It avoids a complete black-face collapse, but the persistent patch captured black/corrupt lower pixels from the historical PNG and produced a severe black semicircle artifact under the face.
- Diagnosis: old committed PNG assets are partially truncated / contain black padding; naive patch extraction will turn that corruption into persistent 3D texture.

### R6 `20260506_134722_r6_real_anchor_cleanpatch`

- Output: `bootstrap_representation.npz`, `bootstrap_face_patch.ply`, train/held-out contact sheets, held-out turntable.
- Visual verdict: diagnostic partial pass, not paper quality. The black semicircle is removed and the clown edit remains visible across yaw views from one persistent representation.
- Remaining failure: it still looks like a 2.5D pasted face patch; side geometry is not physically correct; v2 source/proxy asset is still broken/black; it lacks real depth/3DGS fitting.
- Scientific takeaway: persistent representation plus completion can avoid the old black-face hole, but a heuristic patch is not enough. The next real experiment must use real depth/mesh/3DGS reconstruction and DGE/GaussCtrl/EditSplat-like fitting.

## Recommended next experiment

Use the allowed A100 GPU only after we have a concrete real 3D representation path. The next run should be:

1. Build a small real scene/object reconstruction from complete source views or a small downloaded demo scene under the 80GB cap.
2. Train or load a compact 3DGS `.ply` plus cameras.
3. Render source/depth/mask views.
4. Generate multi-view-consistent edited guidance using either DGE or a minimal GaussCtrl/EditSplat-style reference-view alignment.
5. Fit/update the 3D representation.
6. Inspect 8-16 held-out views and a turntable by eye.

If tonight cannot obtain the full 3DGS input, keep R6 as the honest lower bound: it is better than black-face collapse but not remotely enough for a SIGGRAPH Asia target.


## Subagent literature review matrix integration

A separate literature-review subagent independently reached the same conclusion: the current outputs must be named `lightweight multi-view proxy/harness`, not 3DGS editing or SOTA reproduction.

Minimum future evidence package, adapted from DGE/GaussCtrl/EditSplat/GSEditPro/InterGSEdit/VF-Editor:

- Use one real multi-view scene, at least anchor plus three neighbors on both sides when available.
- Show source views, edited outputs, support masks, confidence/hole maps, outside-mask difference maps, and failure views.
- Mark anchor view, accepted neighbors, rejected views, and why.
- Require visual success in at least 5/7 adjacent views before calling a proxy visually usable.
- If no Gaussian-level labels / 3DGS fitting exist, explicitly call the result an image-space or 2.5D proxy.
- If no depth/geometry-conditioned guidance exists, do not claim geometry consistency.
