from __future__ import annotations


def test_optional_baseline_modules_import_without_loading_weights() -> None:
    from visionlm_fov_matching.baselines import dinov2, loftr, multigradicon, owlv2

    assert dinov2.DINO_MODEL_ID == "facebook/dinov2-base"
    assert callable(multigradicon.centered_moving_canvas)
    assert callable(loftr.load_loftr) and callable(owlv2.load_owlv2)
