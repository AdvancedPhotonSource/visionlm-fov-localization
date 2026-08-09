from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from .types import BoxXYWH


def clip_box(
    box: BoxXYWH,
    image_width: int,
    image_height: int,
    *,
    preserve_size: bool = True,
) -> BoxXYWH:
    """Clip a half-open ``(x, y, width, height)`` box to an image."""

    x, y, width, height = (int(value) for value in box)
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    width = max(1, width)
    height = max(1, height)
    if preserve_size:
        width = min(width, image_width)
        height = min(height, image_height)
        x = min(max(x, 0), image_width - width)
        y = min(max(y, 0), image_height - height)
        return x, y, width, height
    x = min(max(x, 0), image_width - 1)
    y = min(max(y, 0), image_height - 1)
    width = min(width, image_width - x)
    height = min(height, image_height - y)
    return x, y, width, height


def box_iou(first: BoxXYWH, second: BoxXYWH) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    intersection_width = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    intersection_height = max(0, min(ay + ah, by + bh) - max(ay, by))
    intersection = intersection_width * intersection_height
    union = max(0, aw) * max(0, ah) + max(0, bw) * max(0, bh) - intersection
    return float(intersection / union) if union > 0 else 0.0


def polygon_iou(
    first: Sequence[Sequence[float]],
    second: Sequence[Sequence[float]],
) -> float:
    first_points = np.asarray(first, dtype=np.float32).reshape(-1, 2)
    second_points = np.asarray(second, dtype=np.float32).reshape(-1, 2)
    if len(first_points) < 3 or len(second_points) < 3:
        return 0.0
    first_hull = cv2.convexHull(first_points)
    second_hull = cv2.convexHull(second_points)
    first_area = float(cv2.contourArea(first_hull))
    second_area = float(cv2.contourArea(second_hull))
    if first_area <= 0.0 or second_area <= 0.0:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(first_hull, second_hull)
    union = first_area + second_area - float(intersection)
    return float(intersection / union) if union > 0.0 else 0.0


def crop_box(image: np.ndarray, box: BoxXYWH) -> np.ndarray:
    x, y, width, height = box
    return image[y : y + height, x : x + width]


def generate_offset_boxes(
    box: BoxXYWH,
    image_shape: tuple[int, int],
    offset_fraction: float,
) -> tuple[BoxXYWH, ...]:
    """Generate the center and eight neighboring offsets used for re-ranking."""

    image_height, image_width = image_shape
    x, y, width, height = box
    if float(offset_fraction) <= 0.0:
        return (clip_box(box, image_width, image_height),)
    delta_x = max(1, int(round(width * float(offset_fraction))))
    delta_y = max(1, int(round(height * float(offset_fraction))))
    boxes: list[BoxXYWH] = []
    for shift_y in (-delta_y, 0, delta_y):
        for shift_x in (-delta_x, 0, delta_x):
            candidate = clip_box(
                (x + shift_x, y + shift_y, width, height),
                image_width,
                image_height,
            )
            if candidate not in boxes:
                boxes.append(candidate)
    return tuple(boxes)
