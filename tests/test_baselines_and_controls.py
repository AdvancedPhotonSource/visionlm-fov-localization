from __future__ import annotations

import numpy as np

from visionlm_fov_matching.baselines.classical import gradient_ncc
from visionlm_fov_matching.baselines.loftr import boxes_from_matches
from visionlm_fov_matching.baselines.owlv2 import boxes_from_detection
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


def test_loftr_boxes_undo_the_downscaling_and_the_similarity_transform() -> None:
    rng = np.random.default_rng(3)
    template_points = rng.random((40, 2)) * [40, 20]
    angle, zoom, shift = np.deg2rad(90.0), 2.0, np.array([100.0, 50.0])
    rotation = zoom * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    reference_points = template_points @ rotation.T + shift
    match = boxes_from_matches(template_points, reference_points, template_size=(40, 20), box_size=(160, 80), scale=0.5)
    assert match.status == "ok" and match.inlier_count == 40
    # The 40 x 20 template turns a quarter and doubles to 40 x 80 at half scale, so 80 x 160 in full pixels.
    assert match.free_size_box == (120, 100, 80, 160)
    assert match.fixed_size_box == (80, 140, 160, 80)


def test_owlv2_boxes_use_the_long_side_of_the_padded_square() -> None:
    free, fixed = boxes_from_detection((0.5, 0.25, 0.2, 0.1), reference_size=(200, 100), box_size=(30, 10))
    assert free == (80.0, 40.0, 40.0, 20.0)
    assert fixed == (85.0, 45.0, 30.0, 10.0)
