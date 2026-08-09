from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..geometry import box_iou, generate_offset_boxes
from ..settings import PAPER_SETTINGS
from ..types import PolicyResult, Proposal
from .selection import rank_proposals


def controlled_metadata_policy(
    reference: np.ndarray,
    template: np.ndarray,
    primary_proposals: Sequence[Proposal],
) -> PolicyResult:
    """NMI re-ranking with the manuscript's 10% translation jitter."""

    settings = PAPER_SETTINGS.controlled_metadata
    if len(primary_proposals) != settings.primary_repeats:
        raise ValueError(f"expected {settings.primary_repeats} primary proposals")
    expanded: list[Proposal] = []
    for proposal in primary_proposals:
        for box in generate_offset_boxes(proposal.box, reference.shape[:2], settings.offset_fraction):
            expanded.append(Proposal(box=box, source=proposal.source, stage="offset"))
    ranked = rank_proposals(reference, template, expanded, metric=settings.primary_metric)
    return PolicyResult(
        selected=ranked[0] if ranked else None,
        candidates=ranked,
        selection_rule="controlled_metadata_nmi_with_10_percent_offsets",
    )


def challenging_metadata_policy(
    reference: np.ndarray,
    template: np.ndarray,
    primary_proposals: Sequence[Proposal],
    supplemental_proposals: Sequence[Proposal],
) -> PolicyResult:
    """Edge-chamfer policy with small-template NMI fallback and overlap guard."""

    settings = PAPER_SETTINGS.challenging_metadata
    if len(primary_proposals) != settings.primary_repeats:
        raise ValueError(f"expected {settings.primary_repeats} primary proposals")
    if len(supplemental_proposals) != settings.supplemental_repeats:
        raise ValueError(f"expected {settings.supplemental_repeats} supplemental proposals")
    template_area = int(template.shape[0] * template.shape[1])
    metric = settings.primary_metric
    if template_area < settings.fallback_template_area:
        metric = str(settings.fallback_metric)
    primary_ranked = rank_proposals(reference, template, primary_proposals, metric=metric)
    supplemental_ranked = rank_proposals(reference, template, supplemental_proposals, metric=metric)
    selected = primary_ranked[0] if primary_ranked else None
    selection_rule = f"primary_{metric}"
    if selected is not None and supplemental_ranked:
        supplemental = supplemental_ranked[0]
        overlap = box_iou(selected.proposal.box, supplemental.proposal.box)
        if (
            supplemental.primary_score > selected.primary_score
            and overlap >= settings.supplemental_min_overlap_iou
        ):
            selected = supplemental
            selection_rule = f"supplemental_{metric}_overlap_guard"
    return PolicyResult(
        selected=selected,
        candidates=primary_ranked + supplemental_ranked,
        selection_rule=selection_rule,
    )
