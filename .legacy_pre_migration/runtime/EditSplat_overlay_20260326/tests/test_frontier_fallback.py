import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.frontier_fallback import (
    _blend_face_override,
    _compose_anchor_face_fallback,
    _proxy_region_stats,
    _should_use_frontier_fallback,
)


def test_should_use_frontier_fallback_triggers_on_dark_masked_proxy():
    proxy = torch.zeros(1, 3, 8, 8)
    mask = torch.zeros(1, 8, 8)
    mask[:, 2:6, 2:6] = 1.0
    stats = _proxy_region_stats(proxy, mask)
    assert _should_use_frontier_fallback(stats, mean_threshold=0.05, std_threshold=0.02)


def test_should_use_frontier_fallback_skips_when_mask_has_no_coverage():
    proxy = torch.zeros(1, 3, 8, 8)
    mask = torch.zeros(1, 8, 8)
    stats = _proxy_region_stats(proxy, mask)
    assert not _should_use_frontier_fallback(stats, mean_threshold=0.05, std_threshold=0.02)


def test_compose_anchor_face_fallback_writes_bright_anchor_crop_into_target_mask():
    gt = torch.zeros(1, 3, 8, 8)
    proxy = torch.zeros(1, 3, 8, 8)
    anchor = torch.zeros(1, 3, 8, 8)
    anchor[:, :, 2:6, 2:6] = 1.0
    anchor_mask = torch.zeros(1, 8, 8)
    anchor_mask[:, 2:6, 2:6] = 1.0
    target_mask = torch.zeros(1, 8, 8)
    target_mask[:, 1:7, 1:7] = 1.0
    out = _compose_anchor_face_fallback(proxy, gt, anchor, anchor_mask, target_mask, feather_radius=1)
    assert out[:, :, 2:6, 2:6].mean().item() > 0.4


def test_compose_anchor_face_fallback_leaves_proxy_unchanged_when_target_mask_empty():
    proxy = torch.rand(1, 3, 8, 8)
    gt = torch.rand(1, 3, 8, 8)
    anchor = torch.rand(1, 3, 8, 8)
    empty = torch.zeros(1, 8, 8)
    out = _compose_anchor_face_fallback(proxy, gt, anchor, empty, empty, feather_radius=1)
    assert torch.allclose(out, proxy)


def test_blend_face_override_replaces_only_masked_region():
    base = torch.zeros(1, 3, 8, 8)
    override = torch.ones(1, 3, 8, 8)
    mask = torch.zeros(1, 8, 8)
    mask[:, 2:6, 2:6] = 1.0
    out = _blend_face_override(base, override, mask, feather_radius=1)
    assert out[:, :, 2:6, 2:6].mean().item() > 0.9
    assert out[:, :, :2, :2].abs().max().item() < 1e-6


def test_blend_face_override_resizes_override_to_base_shape():
    base = torch.zeros(1, 3, 8, 8)
    override = torch.ones(1, 3, 4, 4)
    mask = torch.ones(1, 8, 8)
    out = _blend_face_override(base, override, mask, feather_radius=1)
    assert out.shape == base.shape
    assert out.mean().item() > 0.9
