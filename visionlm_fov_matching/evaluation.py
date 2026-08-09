from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .geometry import box_iou
from .types import BoxXYWH


@dataclass(frozen=True)
class SummaryStatistics:
    mean: float
    median: float
    confidence_interval: tuple[float, float]
    count: int


def ground_truth_mean_normalized_center(
    ground_truth: Sequence[BoxXYWH],
    image_sizes: Sequence[tuple[int, int]],
) -> tuple[float, float]:
    """Compute the ground-truth-derived location used by the constant prior."""

    if len(ground_truth) != len(image_sizes) or len(ground_truth) == 0:
        raise ValueError("ground truth and image sizes must be non-empty and aligned")
    centers = [
        ((x + width / 2.0) / image_width, (y + height / 2.0) / image_height)
        for (x, y, width, height), (image_width, image_height) in zip(ground_truth, image_sizes)
    ]
    return tuple(float(value) for value in np.mean(np.asarray(centers), axis=0))


def evaluate_boxes(predictions: Sequence[BoxXYWH], ground_truth: Sequence[BoxXYWH]) -> np.ndarray:
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground truth must be aligned")
    return np.asarray(
        [box_iou(prediction, target) for prediction, target in zip(predictions, ground_truth)],
        dtype=float,
    )


def bootstrap_summary(
    values: Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int = 0,
) -> SummaryStatistics:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("values must be a non-empty finite one-dimensional sequence")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(resamples, array.size))
    means = array[indices].mean(axis=1)
    return SummaryStatistics(
        mean=float(array.mean()),
        median=float(np.median(array)),
        confidence_interval=(
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)),
        ),
        count=int(array.size),
    )


def exact_sign_flip_p(differences: Sequence[float], *, alternative: str = "greater") -> float:
    array = np.asarray(differences, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("differences must be non-empty")
    observed = float(array.mean())
    null = np.asarray(
        [float(np.mean(array * signs)) for signs in itertools.product((-1.0, 1.0), repeat=array.size)]
    )
    if alternative == "greater":
        return float(np.mean(null >= observed - 1e-12))
    if alternative == "two-sided":
        return float(np.mean(np.abs(null) >= abs(observed) - 1e-12))
    raise ValueError("alternative must be 'greater' or 'two-sided'")


def leave_one_out_policy_selection(
    iou_matrix: np.ndarray,
    policy_names: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    values = np.asarray(iou_matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != len(policy_names):
        raise ValueError("IoU matrix must be pairs-by-policies with at least two pairs")
    held_out = np.empty(values.shape[0], dtype=float)
    selected: list[str] = []
    for pair_index in range(values.shape[0]):
        training = np.delete(values, pair_index, axis=0)
        policy_index = int(np.argmax(training.mean(axis=0)))
        held_out[pair_index] = values[pair_index, policy_index]
        selected.append(str(policy_names[policy_index]))
    return held_out, tuple(selected)
