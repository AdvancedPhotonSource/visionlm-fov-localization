from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..types import BoxXYWH, Proposal


def proposals_from_boxes(
    boxes: Sequence[BoxXYWH],
    *,
    source: str,
    stage: str,
) -> tuple[Proposal, ...]:
    return tuple(Proposal(box=tuple(int(value) for value in box), source=source, stage=stage) for box in boxes)


def consensus_proposal(proposals: Sequence[Proposal]) -> Proposal:
    if not proposals:
        raise ValueError("at least one proposal is required")
    values = np.asarray([proposal.box for proposal in proposals], dtype=float)
    box = tuple(int(round(value)) for value in np.median(values, axis=0))
    return Proposal(
        box=box,
        source="consensus",
        stage=proposals[0].stage,
        metadata={"member_count": len(proposals)},
    )
