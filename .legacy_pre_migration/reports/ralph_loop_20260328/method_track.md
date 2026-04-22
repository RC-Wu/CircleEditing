## 2026-03-27 20:38:11Z - DNA strong anchor constraints, frontier seam, minimal world-memory step

### 1. Constrain `DNAEdit strong` before propagation, not after propagation

The current runtime confirms that `dnaedit` in this project is only supported on the `static_proxy` branch, and that branch directly edits the proxy image without the proximal per-pixel preserve/edit weighting path (`runtime/EditSplat_overlay_20260326/sandboxes/20260319_aris_ttt3r_flowedit_45/scripts/run_sd35_ttt3r_proximal_wrapper.py:1173-1176`, `:1375-1403`). That means the practical control surface for the front anchor is currently:

- the `flow_dna_*` hyperparameters exposed in `arguments/__init__.py`
- any post-edit compositing we do on the resulting front image before it becomes the frontier anchor

The DNAEdit code itself is useful here. In `DNAEdit_utils.py`, `T_start` is the jump point before target-prompt integration begins, and `mvg` mixes the target velocity with a reference-correction term (`.../DNAEdit_utils.py:112-124`, `:247-280`, `:287-301`, `:459-470`). My read from the code is:

- larger `flow_dna_t_start` should preserve source layout more strongly because the target branch starts later
- larger `flow_dna_src_guidance_scale` should hold the source trajectory more tightly
- smaller `flow_dna_tar_guidance_scale` should reduce over-edit pressure
- smaller `flow_dna_mvg` should lean less on pure target velocity and more on the reference-correction term

That aligns with the manual report: DNA gives a stronger clown edit than FlowEdit, but the drift is mostly pose / head / collar drift rather than "not enough clown". So the first fix should be to localize DNA's effect, not to weaken the whole route uniformly.

The highest-signal minimal constraint is a face-only hybrid anchor:

- run the strong DNA front anchor exactly as now
- predict the front target mask on the original front view
- composite `DNA face` over either the original front view or the current FlowEdit front anchor with a soft feather

For this task, I would prefer `DNA face on source` first, because the drift that was observed is explicitly outside the target region. If the result looks too disconnected from the prompt, the second variant should be `DNA face on FlowEdit strong front`.

This is also consistent with the method literature:

- FlowEdit's gain is that it follows a direct source-to-target ODE instead of a full inversion path, which is one reason it preserves scene layout reasonably well ([FlowEdit](https://matankleiner.github.io/flowedit/)).
- FlowAlign's main idea is to regularize the trajectory with a terminal source-similarity term to improve source consistency ([FlowAlign](https://arxiv.org/abs/2505.23145)).
- DNAEdit explicitly introduces MVG to balance background preservation and target editability, which is exactly the tradeoff failing in the current strong front anchor ([DNAEdit](https://xiechenxi99.github.io/DNAEdit/)).

Action items for the lead loop:

- Run a 4-run front-only DNA sweep with `seed211` fixed: `flow_dna_t_start in {13, 16}` crossed with `flow_dna_mvg in {0.8, 0.65}`, while also changing `flow_dna_src_guidance_scale=1.5` and `flow_dna_tar_guidance_scale=3.0`.
- Inspect those four fronts by eye only for three failure modes: head pose drift, shirt/collar drift, and whether the clown pattern still materially beats the FlowEdit front.
- If one of the four still has the stronger clown effect, promote only that winner into a face-only hybrid anchor instead of pushing raw full-frame DNA into neighbors.
- If none of the four materially reduces drift, stop tuning raw DNA and move immediately to the face-only hybrid anchor, because the problem is spatial scope more than edit strength.

### 2. The simplest way to feed a stronger front anchor into the existing frontier neighbor route is to overwrite one tensor

The current `frontier_seed1` route is already much closer to "anchor propagation" than the run names imply:

- the anchor is selected once by `_pick_frontier_anchor(...)` (`runtime/EditSplat_overlay_20260326/run_editing_flow.py:358-365`, `:1599-1608`)
- frontier neighbors use only that anchor as their source list: `src_cam_idx_list = [frontier_anchor_idx]` (`.../run_editing_flow.py:1700-1701`)
- if the proxy face is dark, the fallback stage also pulls only `edited_image_list[frontier_anchor_idx]` (`.../run_editing_flow.py:1795-1813`)

So the smallest implementation seam is not "rewrite the neighbor route". It is:

1. finish the initial per-view edit pass
2. choose `frontier_anchor_idx`
3. replace `edited_image_list[frontier_anchor_idx]` with the chosen stronger anchor variant
4. let the current neighbor reprojection + fallback path run unchanged

That means the most informative experiment matrix is anchor-swap only, with no other route changes:

- `A`: current FlowEdit strong front anchor baseline
- `B`: raw DNA strong front anchor
- `C`: `DNA face on source` hybrid anchor
- `D`: `DNA face on FlowEdit strong` hybrid anchor

I would not spend runs on front-only FlowEdit again. The report already says it does not materially improve over the existing `grid3_editstrong` front result.

For implementation burden, `C` is the best first move. It can reuse the existing mask and feather logic from `utils/frontier_fallback.py`; the only new behavior needed is "do this once to the anchor view before neighbors consume it". No change is required to the neighbor branch logic itself.

Action items for the lead loop:

- Implement the anchor swap at the seam after `frontier_anchor_idx` is known and before neighbor MFG begins; keep the rest of `frontier_seed1` untouched.
- Run the minimal 3-run propagation matrix: `A` FlowEdit baseline, `C` DNA-on-source hybrid, `D` DNA-on-FlowEdit hybrid.
- Keep `velocity + seed211 + skip_agt + frontier fallback` fixed for these runs so the only variable is the anchor.
- Judge the result on side views only after checking the front anchor by eye; if the front hybrid already looks spatially incoherent, do not waste a neighbor run on it.

### 3. The next minimal world-memory step should be projected anchor memory, not a full MASt3R/TTT3R rewrite

The current fallback is still a 2D crop-resize paste:

- it finds the anchor-mask bbox
- resizes the cropped anchor image into the target-mask bbox
- blends it into `MF_image`

That is good enough to avoid black faces, but it throws away the one thing this route already has: camera geometry and a depth map for the anchor. The runtime already exposes all of the needed geometry utilities in `utils/rgbd_warping.py`:

- `depth_to_points`
- `camera_to_world`
- `world_to_camera`
- `project_points`
- `reproject_rgbd`

So the smallest world-grounded stage for this week is:

1. build a front-anchor memory from masked anchor RGBD points in world coordinates
2. for each frontier neighbor, project only those memory points into the destination camera
3. splat them into a target-face canvas with depth ordering
4. use that projected face canvas as the first replacement for the current 2D anchor crop
5. keep the existing diffusion refinement after `MF_image_cond`

This is not a full memory model yet. It is a one-anchor explicit pointer memory. But it is the right shape:

- Spann3R's gain comes from querying an external spatial memory in a global frame instead of solving pairwise alignment every time ([Spann3R](https://arxiv.org/abs/2408.16061)).
- TTT3R's gain comes from confidence-aware memory updates that balance retention and adaptation over long sequences ([TTT3R](https://rover-xingyu.github.io/TTT3R/)).
- Point3R makes the memory explicit by attaching state to 3D positions in a global coordinate system ([Point3R](https://arxiv.org/abs/2507.02863)).

We do not need their full machinery yet. We only need the minimal projection-memory habit:

- store edit evidence in world space once
- project it forward instead of resize-pasting it in image space

After that works, MASt3R / Test3R / TTT3R become clearer insertion points:

- MASt3R can replace weak per-view geometry with pointmap-based correspondences or a sparse global alignment prior ([MASt3R](https://github.com/naver/mast3r)).
- Test3R gives the simplest "make pairwise predictions agree through a common view" idea, which translates well to "front-anchor memory projected into two side views should be cross-view consistent around the shared face region" ([Test3R](https://arxiv.org/abs/2506.13750)).
- TTT3R suggests confidence-weighted memory updates when more than one edited view is later added to the memory bank.

Action items for the lead loop:

- Implement a new helper beside `utils/frontier_fallback.py`, for example `utils/frontier_memory.py`, that builds masked anchor world points from `edited_image_list[anchor]`, `rendered_depth_list[anchor]`, and `camera_list[anchor]`.
- In the neighbor branch, project that anchor memory into the destination camera and use it as the first face canvas before the current fallback logic; keep the current 2D fallback as a second-stage hole filler for uncovered pixels only.
- Run a 2-run comparison on the same anchor: current 2D fallback versus projected-anchor-memory fallback. Do not change prompts or guidance for this comparison.
- If the projected memory improves ear / hairline / cheek placement even slightly, keep it and postpone MASt3R integration; if it does not, inspect the projected canvas images directly before changing anything else.

### 4. Minimal method bridge for next week

The method bridge that now looks most practical is:

- use FlowEdit-style stability for the route skeleton
- use DNAEdit only as a stronger local face editor for the anchor
- use world-space projection as the propagation primitive
- reserve MASt3R / Test3R / TTT3R for improving geometry and memory confidence once the one-anchor projected-memory stage proves useful

That is a much smaller step than "build a full world model", but it is also the shortest path that actually changes the current failure mode. The reports already show that tiny grid tweaks on the fallback-only regime do not move the result enough, while stronger front guidance does move the front. The missing bridge is therefore not more tuning. It is "make the stronger front edit enter neighbors through geometry instead of through a resized 2D crop".

Recommended run order from here:

1. front-only DNA parameter sweep
2. hybrid-anchor propagation sweep
3. projected-anchor-memory ablation on the winning anchor

Only after one of those three stages shows a real side-view gain should the project spend time integrating MASt3R / Test3R / TTT3R more deeply.

Action items for the lead loop:

- Do not launch more fine-grained `grid2`-style tuning against the current pure fallback route; the last report already showed those levers are too weak.
- Prioritize one code edit only this week: anchor swap plus projected-anchor-memory helper, because both reuse the current route instead of replacing it.
- Save debug images for `anchor_hybrid`, `projected_anchor_memory`, `mf_cond`, and final `mfg_output` for every neighbor so the next note can inspect the failure by eye instead of inferring from means.
- If projected-anchor-memory works, the next implementation step should be confidence weighting for memory updates; if it fails, the next step should be MASt3R pointmap substitution for anchor geometry, not more DNA hyperparameter search.

## 2026-03-27 20:43:09Z - Decision ladder for the next productive week

### 5. Decision ladder: constrain DNA spatially first, then swap the anchor, then ground propagation in projection

The code and run history point to a clean ordering. `DNAEdit` is not failing because the route lacks another global guidance sweep; it is failing because the current `static_proxy` path directly edits the proxy image and therefore has no per-pixel preserve branch to stop full-frame drift (`runtime/EditSplat_overlay_20260326/sandboxes/20260319_aris_ttt3r_flowedit_45/scripts/run_sd35_ttt3r_proximal_wrapper.py:1173-1176`, `:1375-1394`). At the same time, `frontier_seed1` already consumes one anchor everywhere that matters: anchor passthrough, neighbor reprojection source, and dark-proxy fallback all read from `edited_image_list[frontier_anchor_idx]` (`runtime/EditSplat_overlay_20260326/run_editing_flow.py:1635-1671`, `:1700-1749`, `:1787-1814`).

That means the next method step should not be "find a better all-view editor". The next method step should be:

- localize DNA to the front face region
- make that localized result the only anchor override
- let the current neighbor route consume it unchanged
- only then replace 2D crop fallback with projected anchor memory

I would treat the decision gates as follows.

Gate 1: raw DNA can stay in the loop only if a modest parameter retreat keeps the clown pattern stronger than FlowEdit while materially reducing collar / pose drift. The exposed DNA controls already pass straight through `FlowBackendConfig` into the backend call (`runtime/EditSplat_overlay_20260326/flowedit_multimodel/src/core_backend.py:32-45`, `:173-204`), so there is no implementation burden here.

Gate 2: if drift remains mostly outside the face region after that retreat, stop tuning raw DNA immediately and move to a hybrid anchor. The existing soft-mask logic in `utils/frontier_fallback.py` is already enough to build a "DNA inside mask, source or FlowEdit outside mask" anchor without inventing new segmentation machinery.

Gate 3: if the hybrid front anchor looks correct by eye but the side views still lose ear / cheek / hairline placement, the bottleneck has moved from anchor quality to propagation geometry. That is the point where projected anchor memory becomes the shortest-path fix, because `utils/rgbd_warping.py` already exposes `depth_to_points`, `camera_to_world`, `world_to_camera`, `project_points`, and `reprojected2img`.

The main thing to avoid is spending more time on frontier-wide guidance nudges before answering those three gates. The reports already show that the current fallback-dominated route is insensitive to small tuning and sensitive to seed. The productive change is therefore structural but still small: better anchor in, better projection primitive out.

Action items for the lead loop:

- Use this stop rule on raw DNA sweeps: if front-view drift is still clearly visible in hairline, head orientation, or collar after one small retreat sweep, stop raw DNA tuning and move directly to a hybrid anchor.
- Implement the anchor override at the single seam that matters: replace `edited_image_list[frontier_anchor_idx]` after anchor selection and before neighbor reprojection, while keeping `velocity + seed211 + skip_agt + frontier fallback` fixed.
- For the first hybrid test, prefer `DNA face on source` over `DNA face on FlowEdit`, because the observed failure is mostly out-of-mask drift rather than under-editing inside the face.
- Only start the projected-memory helper after one hybrid anchor has passed front-view visual inspection; otherwise the experiment will confound anchor quality with propagation quality.

## 2026-03-27 20:45:19Z - Post-device-fix ranking: projected anchor memory before new flow backends

### 1. What changed after the three cited reports

The method picture is a bit different after `frontier_debug_dna_20260328.md` than it was when the previous note was written:

- The black-face failure mode is fixed on the productive `epoch=0` lane, and the best stable recipe is still `velocity + seed211 + skip_agt + frontier fallback`.
- The minimal `epoch=1` 3DGS path is no longer blocked by the earlier mixed-device rasterization bug. The restore-to-CUDA fix means the low-level track is no longer the main reason to avoid trying better neighbor propagation ideas.
- The real modeling bottleneck is still the same one identified in `frontier_grid3_memory_probe_20260327.md`: the productive neighbor views are mostly generated from a zero-proxy regime plus anchor fallback, so small TTT3R/SAM3 tuning barely moves the result while seed and global guidance still matter a lot.
- `DNAEdit strong` is now a real option in this repo, but the manual read in `frontier_debug_dna_20260328.md` says its gain is stronger clown makeup at the cost of head / collar / identity drift outside the target area.

That changes the practical priority. The next experiments should attack the neighbor-content source itself, not keep spending cycles on tiny per-view tuning.

### 2. Repo-state constraints that should drive method choice

There are two repo facts that matter for implementation burden:

- `FlowAlign` is partially present in-tree. `runtime/EditSplat_overlay_20260326/utils/flow_utils.py` contains `flowalign_flux3d_teacher_grad(...)` and `flowalign_flux3d_step(...)`, but I do not see an active callsite in the current CircleEditing runtime.
- The current productive wrappers still expose only `flowedit` and `dnaedit`. `run_editing_flow.py` warns that other `flow_method` values are not wired into the core multimodel backend yet, and `run_sd35_ttt3r_proximal_wrapper.py` explicitly rejects anything outside `flowedit` / `dnaedit`.
- I do not see an active `RF-Edit` integration inside this tree. The only `rfedit` hit is an old sandbox path string referenced as a fallback location for DNAEdit runtime resolution.

So the realistic burden ordering inside this repo is:

1. reuse the current `FlowEdit` / `DNAEdit` anchor machinery
2. change how the anchor reaches neighbors
3. only then expose new flow backends such as `FlowAlign`
4. treat `RF-Edit` as a later comparison track, not the next implementation

### 3. Method synthesis around the requested families

#### FlowEdit / FlowAlign / RF-Edit

`FlowEdit` remains the right baseline because it already works in the current stable lane and the reports show that stronger guidance does move the anchor when we need it to.

`FlowAlign` is attractive here for one specific reason: its main claim is trajectory regularization for better source consistency during inversion-free editing, which is exactly the kind of drift control that the `DNAEdit strong` anchor is currently missing. Primary source: [FlowAlign](https://arxiv.org/abs/2505.23145).

But the cheapest credible `FlowAlign` experiment in this repo is not "replace the whole productive lane." It is:

- front-only anchor generation
- no neighbor-route changes
- judge whether the front anchor preserves pose / collar / silhouette better than current `FlowEdit` or tuned `DNAEdit`

Anything larger than that will confound backend changes with the still-unfixed neighbor propagation issue.

`RF-Edit` is lower priority here. Its value proposition is feature sharing between inversion and edit to improve source preservation under rectified flow. Primary source: [Taming Rectified Flow for Inversion and Editing](https://arxiv.org/abs/2411.04746). That is interesting in principle, but in this repo it is currently expensive for two reasons:

- there is no active `RF-Edit` path in CircleEditing
- the current productive lane is already inversion-free and is failing more from weak neighbor propagation than from front-anchor inversion quality

So I would treat `RF-Edit` as a benchmark target to compare against later, not as the next engineering task.

#### MASt3R / Test3R / TTT3R / world-memory-style propagation

`TTT3R` is still useful, but the reports argue it has been mispositioned as the primary lever. In the current zero-proxy fallback regime, small `TTT3R` hyperparameter changes were too weak. The better role for `TTT3R` now is confidence weighting and memory update control, not primary edit generation. Primary source: [TTT3R](https://arxiv.org/abs/2509.26645).

`Test3R` is more immediately useful as an evaluation principle than as a full integration. Its core idea is enforcing consistency between reconstructions that share a common view. That translates almost directly to this repo: if two side-view edits both come from one front anchor, they should agree when projected back into the anchor face region. Primary source: [Test3R](https://arxiv.org/abs/2506.13750).

`MASt3R` is the geometry-upgrade option if the projected-memory route still fails because anchor depth or correspondences are too weak around profile views, ears, or hairline boundaries. Primary source: [MASt3R](https://arxiv.org/abs/2406.09756).

For world-memory-style propagation, the right minimal reference is not "train a new recurrent memory model." It is the design habit from external spatial-memory methods: keep edited evidence in a global frame and query it from new views. Primary source: [Spann3R / 3D Reconstruction with Spatial Memory](https://arxiv.org/abs/2408.16061).

That means the next world-memory step in this repo should still be the small one:

- build a masked anchor-face memory in world coordinates from the anchor RGBD
- project it into each neighbor camera
- use that projection as `mf_cond` initialization before diffusion
- keep the current 2D fallback only as a hole filler, not the main content source

### 4. Concrete low-burden experiments I would try next

#### A. Hybrid-anchor propagation sweep

This is still the highest-signal / lowest-burden test.

Keep fixed:

- `velocity`
- `seed211`
- `skip_agt=true`
- frontier fallback enabled
- same prompts and same downstream neighbor route

Change only the front anchor:

- `A1`: current FlowEdit strong anchor baseline
- `A2`: `DNA face on source`
- `A3`: `DNA face on FlowEdit strong`

I would skip raw full-frame `DNAEdit strong` as a primary propagation candidate because the latest report already says the full-frame anchor drifts in exactly the regions we do not want to propagate.

Why this should go first:

- it directly tests whether stronger but spatially constrained front content helps side views
- it needs only one seam edit at the anchor replacement point
- it does not confound the result with a new geometry stack

#### B. Projected-anchor-memory ablation

This is the next experiment if `A2` or `A3` shows any side-view promise at all.

Implementation should stay minimal and reuse existing geometry helpers:

- build masked anchor world points from `edited_image_list[anchor]`, anchor depth, and anchor camera
- project those points into each neighbor with `rgbd_warping.py`
- rasterize them into a projected face canvas
- use that canvas as first-pass neighbor condition
- run the current 2D fallback only where projected memory leaves holes

Compare only three variants:

- `B1`: current 2D bbox-resize fallback
- `B2`: projected-anchor-memory only
- `B3`: projected-anchor-memory plus current 2D hole fill

This is the smallest experiment that actually changes the current failure mode described in `frontier_grid3_memory_probe_20260327.md`.

#### C. Test3R-style consistency scorer for run selection

I would add this before deeper method integration because the current notes still rely too much on visual inspection and crude means.

Minimal version:

- take two edited side views from the same run
- project them back into the anchor view with existing geometry
- compare agreement inside the anchor face mask

Useful outputs:

- masked L1 or LPIPS between the two back-projections
- coverage ratio of valid projected pixels
- simple per-part scores for cheek / hairline / ear region if the face mask can be split

This is not full `Test3R`, but it captures the same common-view consistency idea with very low burden and gives a better run-ranking signal for `A*` and `B*`.

#### D. FlowAlign front-anchor side branch

This is the first new-backend experiment worth doing, but only after `A` and ideally after `C`.

The scope should stay intentionally narrow:

- expose one front-only `FlowAlign` anchor path
- do not change the stable neighbor route yet
- compare front-anchor drift against FlowEdit on the same prompt and seed

The reason to keep this narrow is that `FlowAlign` currently looks partially implemented but not productized in this tree. It is promising, but it is still a bigger engineering branch than anchor swap or projected memory.

#### E. MASt3R local face-correspondence probe

This is the right geometry escalation if projected-anchor-memory still lands the edit in the wrong place on profile views.

The cheap version is not a whole-pipeline MASt3R integration. It is:

- crop the anchor face region and one target side-view region
- estimate stronger correspondences there
- use them only to improve anchor-memory projection around profile features

If the projected-memory ablation already works reasonably, I would postpone this. If projected-memory fails specifically because the anchor depth or camera reprojection is too brittle, then this becomes the next geometry lever.

#### F. Defer full RF-Edit integration

I do not recommend spending the next repo cycle on a full `RF-Edit` migration. If someone wants an `RF-Edit` comparison soon, the cheapest path is an external front-only quality benchmark, not wiring it into frontier propagation first.

### 5. Short ranked proposal list for the next experiments in this repo

1. Run the 3-way hybrid-anchor propagation sweep: FlowEdit baseline vs `DNA face on source` vs `DNA face on FlowEdit strong`.
2. On the best anchor from step 1, run the projected-anchor-memory ablation: current 2D fallback vs projected memory vs projected memory plus 2D hole fill.
3. Add a Test3R-style common-anchor consistency scorer so side-view runs can be ranked by geometry-aware agreement instead of mean intensity and manual inspection alone.
4. If anchor-memory still underperforms, expose a front-only `FlowAlign` anchor branch and compare drift against FlowEdit before touching neighbor propagation again.
5. Escalate to a local face-region MASt3R probe only if projected memory fails because reprojection geometry is visibly the limiting factor.
6. Keep full `RF-Edit` integration out of the immediate loop; treat it as a later benchmark or migration track, not the next experiment.
