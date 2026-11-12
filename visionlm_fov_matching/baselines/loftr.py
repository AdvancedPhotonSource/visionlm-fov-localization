from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..types import BoxXYWH


LOFTR_PRETRAINED = "outdoor"
MAX_SIDE = 1024
MIN_TEMPLATE_SIDE = 16
MIN_MATCHES = 3
RANSAC_THRESHOLD = 4.0


@dataclass(frozen=True)
class LoFTRMatch:
    status: str
    fixed_size_box: BoxXYWH | None = None
    free_size_box: BoxXYWH | None = None
    match_count: int = 0
    inlier_count: int = 0


def load_loftr(*, device: str = "cpu"):
    """Load kornia's LoFTR with its outdoor weights; this may download model weights."""

    import kornia.feature as KF

    return KF.LoFTR(pretrained=LOFTR_PRETRAINED).eval().to(device)


def _unit_range(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    if values.ndim == 3:
        values = values[:, :, :3].mean(axis=2)
    low, high = float(values.min()), float(values.max())
    return (values - low) / (high - low) if high > low else np.zeros_like(values)


def _pad_to_multiple_of_8(image: np.ndarray) -> np.ndarray:
    height, width = image.shape
    return np.pad(image, ((0, (-height) % 8), (0, (-width) % 8)))


def boxes_from_matches(
    template_points: np.ndarray,
    reference_points: np.ndarray,
    *,
    template_size: tuple[int, int],
    box_size: tuple[int, int],
    scale: float,
) -> LoFTRMatch:
    """Fit a RANSAC similarity transform to matches and place both boxes in full-reference pixels.

    Points are in the downscaled images, ``template_size`` is the downscaled template ``(width, height)``
    and ``scale`` the downscaling factor. The fixed-size box centers ``box_size`` on the mapped template
    center, and the free-size box bounds the mapped template corners.
    """

    template_width, template_height = template_size
    template_points = np.asarray(template_points, dtype=np.float64).reshape(-1, 2)
    reference_points = np.asarray(reference_points, dtype=np.float64).reshape(-1, 2)
    inside = (template_points[:, 0] < template_width) & (template_points[:, 1] < template_height)
    template_points, reference_points = template_points[inside], reference_points[inside]
    if len(template_points) < MIN_MATCHES:
        return LoFTRMatch(status="too_few_matches", match_count=len(template_points))
    transform, inliers = cv2.estimateAffinePartial2D(
        template_points,
        reference_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESHOLD,
    )
    if transform is None:
        return LoFTRMatch(status="no_ransac_fit", match_count=len(template_points))
    corners = np.array(
        [[0, 0], [template_width, 0], [template_width, template_height], [0, template_height]],
        dtype=np.float64,
    )
    mapped = (corners @ transform[:, :2].T + transform[:, 2]) / scale
    center_x, center_y = mapped.mean(axis=0)
    box_width, box_height = box_size
    fixed_size_box = (
        int(round(center_x - box_width / 2)),
        int(round(center_y - box_height / 2)),
        int(box_width),
        int(box_height),
    )
    x0, y0 = mapped.min(axis=0)
    x1, y1 = mapped.max(axis=0)
    free_size_box = (
        int(round(x0)),
        int(round(y0)),
        max(1, int(round(x1 - x0))),
        max(1, int(round(y1 - y0))),
    )
    return LoFTRMatch(
        status="ok",
        fixed_size_box=fixed_size_box,
        free_size_box=free_size_box,
        match_count=len(template_points),
        inlier_count=int(inliers.sum()) if inliers is not None else 0,
    )


def loftr_localize(
    matcher,
    reference: np.ndarray,
    template: np.ndarray,
    *,
    box_size: tuple[int, int],
    device: str = "cpu",
) -> LoFTRMatch:
    """Match the XRF map, resized to the supplied box size, against the optical image with LoFTR.

    Both images are scaled together so the optical image's long side is at most ``MAX_SIDE`` pixels.
    A template whose short side falls below ``MIN_TEMPLATE_SIDE`` there is not matched.
    """

    import torch

    reference_unit = _unit_range(reference)
    template_unit = _unit_range(template)
    reference_height, reference_width = reference_unit.shape
    scale = min(1.0, MAX_SIDE / max(reference_height, reference_width))
    reference_scaled = cv2.resize(
        reference_unit,
        (max(1, round(reference_width * scale)), max(1, round(reference_height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    box_width, box_height = box_size
    template_size = (max(1, round(box_width * scale)), max(1, round(box_height * scale)))
    if min(template_size) < MIN_TEMPLATE_SIDE:
        return LoFTRMatch(status="template_too_small")
    template_scaled = cv2.resize(template_unit, template_size, interpolation=cv2.INTER_AREA)
    with torch.inference_mode():
        output = matcher(
            {
                "image0": torch.from_numpy(_pad_to_multiple_of_8(template_scaled))[None, None].float().to(device),
                "image1": torch.from_numpy(_pad_to_multiple_of_8(reference_scaled))[None, None].float().to(device),
            }
        )
    return boxes_from_matches(
        output["keypoints0"].cpu().numpy(),
        output["keypoints1"].cpu().numpy(),
        template_size=template_size,
        box_size=box_size,
        scale=scale,
    )
