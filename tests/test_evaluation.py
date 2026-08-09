from __future__ import annotations

import numpy as np

from visionlm_fov_matching.evaluation import (
    bootstrap_summary,
    exact_sign_flip_p,
    ground_truth_mean_normalized_center,
    leave_one_out_policy_selection,
)


def test_bootstrap_is_deterministic() -> None:
    first = bootstrap_summary([0.1, 0.2, 0.3], resamples=1_000, seed=5)
    second = bootstrap_summary([0.1, 0.2, 0.3], resamples=1_000, seed=5)
    assert first == second
    assert np.isclose(first.mean, 0.2)


def test_exact_sign_flip_probabilities() -> None:
    assert exact_sign_flip_p([1.0, 1.0], alternative="greater") == 0.25
    assert exact_sign_flip_p([1.0, 1.0], alternative="two-sided") == 0.5
    assert exact_sign_flip_p([0.0, 0.0], alternative="two-sided") == 1.0


def test_leave_one_out_policy_selection() -> None:
    matrix = np.asarray([[0.9, 0.1], [0.8, 0.2], [0.1, 0.7]])
    held_out, selected = leave_one_out_policy_selection(matrix, ("first", "second"))
    assert selected == ("first", "first", "first")
    assert np.array_equal(held_out, np.asarray([0.9, 0.8, 0.1]))


def test_ground_truth_constant_position_is_explicit() -> None:
    center = ground_truth_mean_normalized_center(
        ((0, 0, 20, 10), (40, 30, 20, 10)),
        ((100, 50), (100, 50)),
    )
    assert np.allclose(center, (0.3, 0.4))
