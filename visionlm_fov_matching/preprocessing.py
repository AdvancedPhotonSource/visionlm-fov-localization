from __future__ import annotations

import cv2
import numpy as np

from .types import LocalizationInput


def normalize_unit_interval(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("image contains NaN or infinite values")
    low = float(array.min())
    high = float(array.max())
    if high <= low:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - low) / (high - low)).astype(np.float32)


def percentile_normalize(
    image: np.ndarray,
    lower: float = 1.0,
    upper: float = 99.0,
) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(array, [lower, upper])
    if float(high) <= float(low):
        return normalize_unit_interval(array)
    return np.clip((array - low) / (high - low), 0.0, 1.0).astype(np.float32)


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    normalized = percentile_normalize(image)
    gradient_x = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
    return normalize_unit_interval(cv2.magnitude(gradient_x, gradient_y))


def to_grayscale(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[2] == 3:
        return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"unsupported image shape: {array.shape}")


def apply_orientation(
    image: np.ndarray,
    *,
    rotation_degrees: int = 0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
) -> np.ndarray:
    if rotation_degrees % 90 != 0:
        raise ValueError("the reference implementation accepts 90-degree rotations only")
    oriented = np.rot90(image, k=(rotation_degrees // 90) % 4)
    if flip_horizontal:
        oriented = np.fliplr(oriented)
    if flip_vertical:
        oriented = np.flipud(oriented)
    return np.ascontiguousarray(oriented)


def resize_template(template: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    width, height = (int(value) for value in target_size)
    if width <= 0 or height <= 0:
        raise ValueError("target template dimensions must be positive")
    return cv2.resize(np.asarray(template), (width, height), interpolation=cv2.INTER_LINEAR)


def prepare_localization_input(
    reference: np.ndarray,
    phosphorus_map: np.ndarray,
    *,
    fixed_box_size: tuple[int, int] | None = None,
) -> LocalizationInput:
    reference_gray = percentile_normalize(to_grayscale(reference))
    template = percentile_normalize(phosphorus_map)
    if fixed_box_size is not None:
        template = resize_template(template, fixed_box_size)
    return LocalizationInput(
        reference=reference_gray,
        template=template,
        fixed_box_size=fixed_box_size,
    )
