from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..settings import PAPER_SETTINGS
from ..types import BoxXYWH, PolicyResult, Proposal
from .selection import most_diverse_top, rank_proposals


def build_zoom_crop(
    seed_boxes: Sequence[BoxXYWH],
    *,
    image_shape: tuple[int, int],
    template_size: tuple[int, int],
) -> BoxXYWH:
    if not seed_boxes:
        raise ValueError("at least one seed box is required")
    settings = PAPER_SETTINGS.unconstrained_zoom
    image_height, image_width = image_shape
    template_width, template_height = template_size
    left = min(box[0] for box in seed_boxes)
    top = min(box[1] for box in seed_boxes)
    right = max(box[0] + box[2] for box in seed_boxes)
    bottom = max(box[1] + box[3] for box in seed_boxes)
    margin = int(round(settings.zoom_margin_scale * max(template_width, template_height)))
    crop_width = max(
        int(round(settings.zoom_min_scale * template_width)),
        right - left + 2 * margin,
        template_width + 64,
    )
    crop_height = max(
        int(round(settings.zoom_min_scale * template_height)),
        bottom - top + 2 * margin,
        template_height + 64,
    )
    crop_width = min(crop_width, image_width)
    crop_height = min(crop_height, image_height)
    center_x = int(round((left + right) / 2.0))
    center_y = int(round((top + bottom) / 2.0))
    x = min(max(center_x - crop_width // 2, 0), image_width - crop_width)
    y = min(max(center_y - crop_height // 2, 0), image_height - crop_height)
    return x, y, crop_width, crop_height


def local_fraction_to_box(
    fraction: tuple[float, float],
    *,
    zoom_crop: BoxXYWH,
    template_size: tuple[int, int],
) -> BoxXYWH:
    crop_x, crop_y, crop_width, crop_height = zoom_crop
    template_width, template_height = template_size
    fraction_x = min(max(float(fraction[0]), 0.0), 1.0)
    fraction_y = min(max(float(fraction[1]), 0.0), 1.0)
    x = crop_x + int(round(fraction_x * max(crop_width - template_width, 0)))
    y = crop_y + int(round(fraction_y * max(crop_height - template_height, 0)))
    return x, y, template_width, template_height


def unconstrained_zoom_policy(
    reference: np.ndarray,
    template: np.ndarray,
    direct_proposals: Sequence[Proposal],
    local_fractions: Sequence[tuple[float, float]],
) -> tuple[PolicyResult, BoxXYWH]:
    """Assemble three direct and three local proposals, then re-rank all six."""

    settings = PAPER_SETTINGS.unconstrained_zoom
    if len(direct_proposals) != settings.direct_proposals:
        raise ValueError(f"expected {settings.direct_proposals} direct proposals")
    if len(local_fractions) != settings.zoom_repeats:
        raise ValueError(f"expected {settings.zoom_repeats} local responses")
    direct_ranked = rank_proposals(reference, template, direct_proposals, metric=settings.rerank_metric)
    seeds = most_diverse_top(direct_ranked, count=settings.zoom_seed_count)
    seed_boxes = [item.proposal.box for item in seeds] or [direct_proposals[0].box]
    template_size = (int(template.shape[1]), int(template.shape[0]))
    zoom_crop = build_zoom_crop(seed_boxes, image_shape=reference.shape[:2], template_size=template_size)
    local = tuple(
        Proposal(
            box=local_fraction_to_box(fraction, zoom_crop=zoom_crop, template_size=template_size),
            source="vlm_zoom",
            stage="zoom",
        )
        for fraction in local_fractions
    )
    ranked = rank_proposals(
        reference,
        template,
        tuple(direct_proposals) + local,
        metric=settings.rerank_metric,
    )
    return (
        PolicyResult(
            selected=ranked[0] if ranked else None,
            candidates=ranked,
            selection_rule="six_candidate_zoom_nmi_rerank",
        ),
        zoom_crop,
    )
