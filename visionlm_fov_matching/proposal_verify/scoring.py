from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ..geometry import crop_box
from ..preprocessing import gradient_magnitude, normalize_unit_interval, percentile_normalize


@dataclass(frozen=True)
class VerificationImages:
    reference_raw: np.ndarray
    reference_percentile: np.ndarray
    reference_gradient: np.ndarray
    template_raw: np.ndarray
    template_percentile: np.ndarray
    template_gradient: np.ndarray


def normalized_mutual_information(first: np.ndarray, second: np.ndarray, *, bins: int = 32) -> float:
    """Return mutual information normalized by the geometric mean entropy."""

    if first.shape != second.shape or first.size == 0:
        return 0.0
    first_values = np.asarray(first, dtype=np.float32).reshape(-1)
    second_values = np.asarray(second, dtype=np.float32).reshape(-1)
    histogram, _, _ = np.histogram2d(
        first_values,
        second_values,
        bins=bins,
        range=((0.0, 1.0), (0.0, 1.0)),
    )
    joint = histogram / max(float(histogram.sum()), 1.0)
    first_probability = joint.sum(axis=1)
    second_probability = joint.sum(axis=0)
    nonzero = joint > 0
    product = np.outer(first_probability, second_probability)
    mutual_information = float(
        np.sum(joint[nonzero] * np.log(np.maximum(joint[nonzero], 1e-12) / np.maximum(product[nonzero], 1e-12)))
    )
    first_entropy = float(-np.sum(first_probability[first_probability > 0] * np.log(first_probability[first_probability > 0])))
    second_entropy = float(-np.sum(second_probability[second_probability > 0] * np.log(second_probability[second_probability > 0])))
    if first_entropy <= 1e-8 or second_entropy <= 1e-8:
        return 0.0
    return float(mutual_information / math.sqrt(first_entropy * second_entropy))


def zero_mean_normalized_correlation(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape or first.size == 0:
        return -1.0
    first_values = np.asarray(first, dtype=np.float32).reshape(-1)
    second_values = np.asarray(second, dtype=np.float32).reshape(-1)
    first_values -= float(first_values.mean())
    second_values -= float(second_values.mean())
    denominator = float(np.linalg.norm(first_values) * np.linalg.norm(second_values))
    return float(np.dot(first_values, second_values) / denominator) if denominator > 1e-8 else -1.0


def _edge_map(image: np.ndarray) -> np.ndarray:
    values = np.clip(percentile_normalize(image) * 255.0, 0.0, 255.0).astype(np.uint8)
    otsu, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low = max(0, int(round(0.5 * float(otsu))))
    high = max(low + 1, int(round(min(255.0, 1.5 * float(otsu) + 16.0))))
    return cv2.Canny(values, threshold1=low, threshold2=high) > 0


def edge_chamfer_similarity(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape or first.size == 0:
        return -1e9
    first_edges = _edge_map(first)
    second_edges = _edge_map(second)
    if not first_edges.any() and not second_edges.any():
        return 0.0
    if not first_edges.any() or not second_edges.any():
        return -1e6
    first_distance = cv2.distanceTransform((~first_edges).astype(np.uint8), cv2.DIST_L2, 3)
    second_distance = cv2.distanceTransform((~second_edges).astype(np.uint8), cv2.DIST_L2, 3)
    forward = float(np.mean(second_distance[first_edges]))
    backward = float(np.mean(first_distance[second_edges]))
    return float(-0.5 * (forward + backward))


def prepare_verification_images(reference: np.ndarray, template: np.ndarray) -> VerificationImages:
    """Prepare the full-image score representations used in the manuscript."""

    reference_raw = normalize_unit_interval(reference)
    template_raw = normalize_unit_interval(template)
    reference_percentile = percentile_normalize(reference_raw)
    template_percentile = percentile_normalize(template_raw)
    return VerificationImages(
        reference_raw=reference_raw,
        reference_percentile=reference_percentile,
        reference_gradient=gradient_magnitude(reference_percentile),
        template_raw=template_raw,
        template_percentile=template_percentile,
        template_gradient=gradient_magnitude(template_percentile),
    )


def _resized_crop(image: np.ndarray, box: tuple[int, int, int, int], target_size: tuple[int, int]) -> np.ndarray:
    patch = np.asarray(crop_box(image, box), dtype=np.float32)
    target_width, target_height = target_size
    if patch.shape[:2] == (target_height, target_width):
        return patch
    return cv2.resize(patch, target_size, interpolation=cv2.INTER_LINEAR)


def score_box(images: VerificationImages, box: tuple[int, int, int, int]) -> dict[str, float]:
    target_size = (int(images.template_raw.shape[1]), int(images.template_raw.shape[0]))
    raw_patch = _resized_crop(images.reference_raw, box, target_size)
    percentile_patch = _resized_crop(images.reference_percentile, box, target_size)
    gradient_patch = _resized_crop(images.reference_gradient, box, target_size)
    return {
        "nmi_raw": normalized_mutual_information(raw_patch, images.template_raw),
        "zncc_percentile": zero_mean_normalized_correlation(percentile_patch, images.template_percentile),
        "zncc_gradient": zero_mean_normalized_correlation(gradient_patch, images.template_gradient),
        "edge_chamfer": edge_chamfer_similarity(percentile_patch, images.template_percentile),
    }
