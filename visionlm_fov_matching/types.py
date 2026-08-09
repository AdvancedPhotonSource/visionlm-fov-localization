from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, TypeAlias

import numpy as np


BoxXYWH: TypeAlias = tuple[int, int, int, int]


@dataclass(frozen=True)
class LocalizationInput:
    """Prepared optical reference and XRF template used by a localizer."""

    reference: np.ndarray
    template: np.ndarray
    fixed_box_size: tuple[int, int] | None = None


@dataclass(frozen=True)
class Demonstration:
    """One completed in-context example for a VLM prompt."""

    reference_data_url: str
    template_data_urls: tuple[str, ...]
    answer: str


@dataclass(frozen=True)
class Proposal:
    """One candidate field of view in optical-image coordinates."""

    box: BoxXYWH
    source: str
    stage: str = "direct"
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredProposal:
    proposal: Proposal
    primary_score: float
    scores: Mapping[str, float]


@dataclass(frozen=True)
class PolicyResult:
    selected: ScoredProposal | None
    candidates: tuple[ScoredProposal, ...]
    selection_rule: str
