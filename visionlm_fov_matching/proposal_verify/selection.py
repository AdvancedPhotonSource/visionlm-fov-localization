from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..geometry import clip_box
from ..types import Proposal, ScoredProposal
from .scoring import VerificationImages, prepare_verification_images, score_box


SUPPORTED_METRICS = frozenset({"nmi_raw", "zncc_percentile", "zncc_gradient", "edge_chamfer"})


def _score_prepared_proposal(
    images: VerificationImages,
    proposal: Proposal,
    *,
    image_shape: tuple[int, int],
    metric: str,
) -> ScoredProposal:
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"unsupported verification metric: {metric!r}")
    reference_height, reference_width = image_shape
    box = clip_box(proposal.box, reference_width, reference_height, preserve_size=False)
    clipped = Proposal(box=box, source=proposal.source, stage=proposal.stage, metadata=proposal.metadata)
    scores = score_box(images, box)
    return ScoredProposal(proposal=clipped, primary_score=float(scores[metric]), scores=scores)


def score_proposal(reference: np.ndarray, template: np.ndarray, proposal: Proposal, *, metric: str) -> ScoredProposal:
    images = prepare_verification_images(reference, template)
    return _score_prepared_proposal(images, proposal, image_shape=reference.shape[:2], metric=metric)


def rank_proposals(
    reference: np.ndarray,
    template: np.ndarray,
    proposals: Sequence[Proposal],
    *,
    metric: str,
) -> tuple[ScoredProposal, ...]:
    images = prepare_verification_images(reference, template)
    scored = [
        _score_prepared_proposal(images, proposal, image_shape=reference.shape[:2], metric=metric)
        for proposal in proposals
    ]
    scored.sort(
        key=lambda item: (
            item.primary_score,
            item.scores["nmi_raw"],
            item.scores["zncc_percentile"],
            item.scores["zncc_gradient"],
            item.scores["edge_chamfer"],
        ),
        reverse=True,
    )
    return tuple(scored)


def most_diverse_top(
    ranked: Sequence[ScoredProposal],
    *,
    count: int,
    maximum_iou: float = 0.8,
) -> tuple[ScoredProposal, ...]:
    from ..geometry import box_iou

    selected: list[ScoredProposal] = []
    for candidate in ranked:
        if any(box_iou(candidate.proposal.box, prior.proposal.box) > maximum_iou for prior in selected):
            continue
        selected.append(candidate)
        if len(selected) == count:
            break
    return tuple(selected)
