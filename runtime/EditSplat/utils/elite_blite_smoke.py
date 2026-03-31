#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.ttt3r_elite_blite import (
    ELiteCorrectionConfig,
    SourceCanonicalPrior,
    apply_elite_correction_weights,
    update_source_canonical_prior,
)


def main() -> None:
    edit = np.array([[[[0.1, 0.6], [0.2, 0.8]]]], dtype=np.float32)
    preserve = np.array([[[[0.7, 0.2], [0.6, 0.1]]]], dtype=np.float32)
    support = np.array([[[[1.0, 0.5], [0.0, 1.0]]]], dtype=np.float32)
    confidence = np.array([[[[0.2, 0.9], [0.4, 1.0]]]], dtype=np.float32)

    cfg = ELiteCorrectionConfig(enabled=True, support_alpha=0.4, edit_alpha=0.5, confidence_alpha=0.6)
    edit_out, preserve_out, combo = apply_elite_correction_weights(
        edit_weight=edit,
        preserve_weight=preserve,
        support_weight=support,
        edit_mask=edit,
        confidence_weight=confidence,
        cfg=cfg,
    )
    assert bool(np.all(edit_out <= edit + 1e-6))
    assert bool(np.all(preserve_out <= preserve + 1e-6))
    assert float(np.mean(combo)) > 0.0

    prior = SourceCanonicalPrior()
    update_source_canonical_prior(
        prior=prior,
        view_idx=0,
        edit_weight=edit_out,
        preserve_weight=preserve_out,
        confidence_weight=confidence,
        support_weight=support,
        metadata={"tag": "smoke"},
    )
    serialized = prior.to_serializable()
    assert serialized["schema"] == "source_canonical_prior_v0"
    assert "0" in serialized["views"]


if __name__ == "__main__":
    main()
