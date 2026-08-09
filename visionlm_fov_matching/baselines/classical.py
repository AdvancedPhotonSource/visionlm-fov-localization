from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import cv2
import numpy as np

from ..preprocessing import gradient_magnitude, resize_template
from ..types import BoxXYWH


@dataclass(frozen=True)
class ClassicalMatch:
    box: BoxXYWH
    score: float
    template_size: tuple[int, int]


def gradient_ncc(
    reference: np.ndarray,
    template: np.ndarray,
    *,
    target_size: tuple[int, int] | None = None,
) -> ClassicalMatch:
    """Locate a template using gradient magnitude and normalized correlation."""

    matched_template = resize_template(template, target_size) if target_size is not None else template
    reference_feature = gradient_magnitude(reference)
    template_feature = gradient_magnitude(matched_template)
    template_height, template_width = template_feature.shape[:2]
    reference_height, reference_width = reference_feature.shape[:2]
    if template_width > reference_width or template_height > reference_height:
        raise ValueError("template must fit inside the reference image")
    response = cv2.matchTemplate(
        reference_feature.astype(np.float32),
        template_feature.astype(np.float32),
        cv2.TM_CCOEFF_NORMED,
    )
    response = np.nan_to_num(response, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    _, score, _, location = cv2.minMaxLoc(response)
    return ClassicalMatch(
        box=(int(location[0]), int(location[1]), template_width, template_height),
        score=float(score),
        template_size=(template_width, template_height),
    )


def multiscale_gradient_ncc(
    reference: np.ndarray,
    template: np.ndarray,
    *,
    short_side_sizes: Sequence[int] = (64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048),
) -> ClassicalMatch:
    """Run the manuscript's unconstrained multi-scale gradient + NCC search."""

    template_height, template_width = template.shape[:2]
    short_side = min(template_width, template_height)
    if short_side <= 0:
        raise ValueError("template dimensions must be positive")
    reference_height, reference_width = reference.shape[:2]
    matches: list[ClassicalMatch] = []
    for candidate_short_side in short_side_sizes:
        scale = float(candidate_short_side) / float(short_side)
        target_size = (
            max(1, int(round(template_width * scale))),
            max(1, int(round(template_height * scale))),
        )
        if target_size[0] > reference_width or target_size[1] > reference_height:
            continue
        matches.append(gradient_ncc(reference, template, target_size=target_size))
    if not matches:
        raise ValueError("no candidate scale fits inside the reference image")
    return max(matches, key=lambda match: match.score)
