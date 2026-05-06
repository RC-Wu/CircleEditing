# 2026-05-06 Feed-forward Proxy Sprint Note

## Goal

Produce a visually inspectable single-object multi-view edit under the new A100 sprint constraints:

- all files under `/mnt/beegfs/ruocheng/circleediting_sigasia_20260506`;
- total sprint root under 80GB;
- only GPUs 6 and 7 may be used;
- no training;
- visual review is mandatory.

## Repository state

The GitHub `RC-Wu/CircleEditing` checkout is clean on `main`, but it is a compact project mirror rather than a full executable EditSplat checkout. Missing pieces include the renderer entrypoint, argument stack, dataloader, and several utility modules imported by `runtime/EditSplat/run_editing_flow.py`.

This makes the full historical FlowEdit/EditSplat stack a high-latency path for tonight. The fast path is to keep the committed historical visual artifacts and build a small proxy experiment around them.

## R1 design

Use the previous frontier/SAM run as a minimal multi-view case. The old report already says segmentation/localization was mostly fixed while content propagation collapsed to black. R1 therefore tests deterministic propagation variants:

- anchor diff mask from `view000_input` vs `view000_initial_edit`;
- support masks for neighbor views;
- pseudo-depth / object proxy overlays for geometry sanity;
- masked color/chroma/patch/delta/seamless-clone transfer variants;
- one contact sheet containing all variants for direct human review.

## What to look for

A useful result must show a visible object-level edit in non-anchor views, keep the edit inside the target object, and avoid black filled regions or obvious rectangular paste seams. Metrics in `summary.json` are only triage signals; the contact sheet is the real evidence.

## Next escalation if R1 is promising

Replace the proxy geometry block with a real feed-forward model, preferably a small CUT3R/TTT3R path, while keeping the same run/output contract. Do not introduce a large model family until R1 proves the visual propagation problem is worth heavier setup.


## R2 design update

The first proxy run showed that direct anchor patch warping can fix the black side-face failure in one view, but it pastes a frontal face into the neighbor view. R2 therefore adds semantic mask-coordinate variants. These variants treat the anchor clown edit as object-internal makeup primitives and redraw those primitives in each target mask's normalized coordinate frame. This is a pragmatic visual fallback, not a replacement for real geometry.


## R3 design update

R2 reduced frontal-paste artifacts but produced soft gray-mask edits. R3 adds crisp semantic variants with smaller face-coordinate primitives, stronger red/black/blue makeup, weaker full-face foundation, and optional dark-proxy patch repair. The goal is an inspectable fallback result that is visually clearer, not a claim of solved 3D geometry.


## R4 design update

R3 showed that different views need different treatment. Normal input views should preserve identity/lighting and only add crisp makeup primitives, while dark proxy views need patch repair before semantic drawing. R4 adds `adaptive_final` and `adaptive_final_bold`, plus a `final_selected/final_contact_sheet.jpg` export for direct inspection.


## R5 design update

R4 reached a usable fallback but the normal neighbor view was too heavy. R5 adds `adaptive_final_clean` and `adaptive_final_balanced`. The default final export now uses `adaptive_final_balanced`, which keeps normal views more identity-preserving while still repairing the dark proxy view.
