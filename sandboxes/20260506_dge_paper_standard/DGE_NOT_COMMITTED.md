# DGE Sandbox Not Committed

DGE was cloned during the 2026-05-06 sprint under this sandbox to inspect feasibility. It is an embedded upstream git repository and is intentionally not committed into CircleEditing.

Recreate if needed:

```bash
git clone --depth 1 https://github.com/silent-chen/DGE.git sandboxes/20260506_dge_paper_standard/DGE
```

Current conclusion: DGE is the right paper-standard direction, but it requires a pretrained 3DGS model plus its training data and a heavier torch/diffusers/threestudio environment. The current GitHub CircleEditing checkout does not yet include those inputs.
