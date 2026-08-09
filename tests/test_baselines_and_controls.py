from __future__ import annotations

import numpy as np

from visionlm_fov_matching.baselines.classical import gradient_ncc
from visionlm_fov_matching.controls import constant_center_box, expected_random_box_iou, image_center_box


def test_gradient_ncc_recovers_a_synthetic_translation() -> None:
    rng = np.random.default_rng(4)
    template = rng.random((12, 16), dtype=np.float32)
    reference = np.zeros((60, 80), dtype=np.float32)
    reference[21:33, 37:53] = template
    result = gradient_ncc(reference, template)
    assert result.box == (37, 21, 16, 12)
    assert result.score > 0.25


def test_geometric_controls_are_deterministic() -> None:
    assert image_center_box((100, 80), (20, 10)) == (40, 35, 20, 10)
    assert constant_center_box((100, 80), (20, 10), (0.25, 0.75)) == (15, 55, 20, 10)
    first = expected_random_box_iou((30, 20, 20, 10), (100, 80), (20, 10), seed=7)
    second = expected_random_box_iou((30, 20, 20, 10), (100, 80), (20, 10), seed=7)
    assert first == second
    assert 0.0 <= first <= 1.0
