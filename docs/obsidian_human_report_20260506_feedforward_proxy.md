# CircleEditing 2026-05-06 Feed-forward Proxy Sprint Summary

## What happened

I rebuilt the GitHub `RC-Wu/CircleEditing` project inside `/mnt/beegfs/ruocheng/circleediting_sigasia_20260506` on `a100-2`, kept all durable files under that root, and used only the allowed GPU binding. The GitHub checkout was a compact mirror rather than a full executable EditSplat runtime, so I implemented a lightweight proxy-first visual experiment instead of spending the night reconstructing the full old stack.

## Main result

After five experiment rounds, the selected result is R5 `adaptive_final_balanced`:

- remote run root: `/mnt/beegfs/ruocheng/circleediting_sigasia_20260506/runs/feedforward_proxy/20260506_122412_r5_balanced`
- repo visual artifact: `assets/review/feedforward_proxy_r5_balanced_20260506/final_contact_sheet.jpg`
- repo report: `reports/feedforward_proxy_r5_balanced_20260506.md`

Visual judgment: dirty but usable as a fallback demo. The final contact sheet shows clown edit in all three views, preserves background/clothes, and fixes the previous black-face side-view collapse.

## What this does and does not prove

It supports the hypothesis that a reconstruction/proxy-first pipeline plus adaptive semantic completion is a useful emergency path when direct FlowEdit/EditSplat propagation collapses. It does not prove solved 3D consistency, does not replace CUT3R/TTT3R, and should not be overclaimed.

## Next concrete step

Keep this exact input/output contract and replace the pseudo-depth/support-mask proxy with a real feed-forward geometry backend. Use the R5 adaptive semantic completion as the fallback layer for dark/invalid projected content.
