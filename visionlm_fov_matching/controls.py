from __future__ import annotations

import numpy as np

from .geometry import box_iou, clip_box
from .types import BoxXYWH


def image_center_box(image_size: tuple[int, int], box_size: tuple[int, int]) -> BoxXYWH:
    image_width, image_height = image_size
    box_width, box_height = box_size
    return clip_box(
        (
            int(round((image_width - box_width) / 2.0)),
            int(round((image_height - box_height) / 2.0)),
            box_width,
            box_height,
        ),
        image_width,
        image_height,
    )


def constant_center_box(
    image_size: tuple[int, int],
    box_size: tuple[int, int],
    normalized_center: tuple[float, float],
) -> BoxXYWH:
    image_width, image_height = image_size
    box_width, box_height = box_size
    center_x = normalized_center[0] * image_width
    center_y = normalized_center[1] * image_height
    return clip_box(
        (
            int(round(center_x - box_width / 2.0)),
            int(round(center_y - box_height / 2.0)),
            box_width,
            box_height,
        ),
        image_width,
        image_height,
    )


def expected_random_box_iou(
    ground_truth: BoxXYWH,
    image_size: tuple[int, int],
    box_size: tuple[int, int],
    *,
    samples: int = 5_000,
    seed: int = 0,
) -> float:
    image_width, image_height = image_size
    box_width, box_height = box_size
    if box_width > image_width or box_height > image_height:
        raise ValueError("box must fit inside the image")
    rng = np.random.default_rng(seed)
    xs = rng.integers(0, image_width - box_width + 1, size=samples)
    ys = rng.integers(0, image_height - box_height + 1, size=samples)
    values = [
        box_iou((int(x), int(y), box_width, box_height), ground_truth)
        for x, y in zip(xs, ys)
    ]
    return float(np.mean(values))
