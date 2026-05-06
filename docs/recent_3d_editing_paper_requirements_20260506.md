# 2026-05-06 Recent 3D Editing Paper Requirements

## Current verdict

The R1-R5 feed-forward proxy experiments and the R1-R4 paper-standard harness runs are **not** paper-quality 3D editing results. They are useful diagnostics only.

- `feedforward_proxy/R1-R5`: rejected because they are view-wise/proxy image edits and do not update a persistent 3D representation.
- `paper_standard_harness/R1-R2`: rejected visually because the face surfel scene is sparse, noisy, and semantically weak.
- `paper_standard_harness/R3`: cleaner object-space edit, but it is a toy surfel mug and still visibly point-rendered.
- `paper_standard_harness/R4`: cleaner mesh/texture contract, but it is procedural and not a reconstructed real scene.

A successful result for this project must be judged against recent 3D Gaussian/3D editing papers, not against our own earlier broken baselines.

## Paper-derived requirements

| Paper | Core signal | Requirement for CircleEditing | Current failure mode |
|---|---|---|---|
| DGE, Chen et al., ECCV 2024 / arXiv 2404.18929 | Two-stage editing: multi-view-consistent 2D guidance using 3D geometry, then direct 3DGS fitting. | We need a real or GS-ready 3D representation, consistent edited sequence, and an optimized edited representation. | Proxy PNG propagation and toy harnesses do not fit or update a real 3DGS. |
| GaussCtrl, Wu et al., ECCV 2024 / arXiv 2403.08733 | Depth-conditioned editing plus attention-based latent alignment across views. | Use depth/geometry conditioning and cross-view/reference-view alignment before fitting. | Our frontier run used constant-depth shortcuts; black-face collapse proves geometry/content transfer is inadequate. |
| EditSplat, Lee et al., CVPR 2025 / arXiv 2412.11520 | Multi-view fusion guidance plus attention-guided optimization of 3DGS. | Use source views, 3DGS geometry, and attention/localization to restrict optimization to target Gaussians. | Current GitHub runtime has partial EditSplat mirror only; no full runnable scene/checkpoint path yet. |
| GaussianEditor, Chen et al., CVPR 2024 / arXiv 2311.14521 | Gaussian semantic tracing and hierarchical GS for controllable local edits. | Target Gaussians must be explicitly localized and background must remain stable. | Current proxies localize 2D masks but never trace/update Gaussians. |
| GSEditPro, Sun et al., arXiv 2411.10033 | Attention-based progressive localization labels Gaussians by prompt relevance. | Need progressive target localization, not broad full-image or full-face painting. | R1-R5 edits often overpaint and rely on heuristic image masks. |
| InterGSEdit, Wen et al., arXiv 2507.04961 | 3D geometry-consistent attention prior, selected key views, and reference-view consistency screening. | Key-view selection and reference views must be used to constrain diffusion/attention, not one-shot prompt edits. | We have an anchor/frontier idea, but no robust attention prior or reference-view screening. |
| VF-Editor, Qin et al., ICLR 2026 / arXiv 2602.11638 | Indirect 2D-edit-then-project pipelines inherently create cross-view inconsistency; native Gaussian attribute variation is stronger. | Long-term direction should be native/GS-attribute edits or feed-forward variation prediction, not only 2D projection. | Current fast harnesses are indirect and procedural, so they cannot be final research contribution. |
| TIP-Editor / Instruct-GS2GS family | Local text/image prompted edits preserve unedited regions and render novel views from the edited 3D representation. | Need target prompt/reference, masks, final 3D representation, and held-out turntable review. | Current toy harness has these gates only in synthetic form. |

## Hard acceptance gates

A result can be called `usable` only if it passes **all** gates by direct visual inspection:

1. `Real representation gate`: final output is a real scene/object representation (`3DGS .ply`, mesh/texture from reconstruction, or feed-forward 3D model output), not only edited PNGs.
2. `Source fidelity gate`: source scene/object identity, background, camera geometry, and unedited parts remain recognizable.
3. `Prompt/reference gate`: the requested edit is visually obvious and semantically correct from front, side, and held-out views.
4. `Multi-view guidance gate`: edited guidance views are mutually consistent before fitting; no independent per-view style drift.
5. `Geometry gate`: side views and occlusion boundaries follow 3D geometry; no frontal texture pasted onto side views.
6. `Localization gate`: target object/part only; background, clothing/hair/other objects remain stable unless the prompt asks otherwise.
7. `Novel-view gate`: at least 8-16 held-out views or a turntable video rendered from the same edited representation.
8. `Failure disclosure gate`: toy/proxy/compositing results must be labeled as diagnostics, not successes.
9. `Storage/runtime gate`: sprint root stays under 80GB; no uncontrolled checkpoint downloads.

## Immediate minimum viable path

The next acceptable experiment is not another procedural toy. It should be one of these, in priority order:

1. **Real 3DGS/DGE path**: obtain or reconstruct a small 3DGS scene, render source views/depth, create multi-view-consistent edited guidance, run DGE/EditSplat-style 3D fitting, and inspect held-out renders.
2. **Feed-forward reconstruction bootstrap**: use a streaming/feed-forward reconstruction model or a lightweight 2.5D/mesh bootstrap on a real object image/edit pair, save a persistent representation, render held-out views, and mark exactly where it fails relative to DGE/GaussCtrl/EditSplat.
3. **Harness-only fallback**: maintain toy surfel/mesh harnesses only as automated gates for representation/novel-view/localization checks.

## Current practical blocker

The GitHub `CircleEditing` checkout contains useful runtime fragments and previous visual artifacts, but not a complete ready-to-run 3DGS scene/checkpoint plus training data. The DGE clone is present under `sandboxes/20260506_dge_paper_standard/DGE`, but DGE requires:

- `data.source`: training dataset with cameras/images;
- `system.gs_source`: pretrained 3DGS model;
- a heavy env with torch/diffusers/threestudio/3DGS dependencies;
- deliberate GPU use on only allowed `a100-2` GPUs 6/7.

Until those inputs exist, DGE cannot honestly produce paper-standard results from the GitHub repo alone.
