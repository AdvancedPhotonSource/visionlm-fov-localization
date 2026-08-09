from __future__ import annotations

import numpy as np

from visionlm_fov_matching.proposal_verify import policies
from visionlm_fov_matching.proposal_verify.policies import (
    challenging_metadata_policy,
    controlled_metadata_policy,
)
from visionlm_fov_matching.proposal_verify.scoring import normalized_mutual_information
from visionlm_fov_matching.proposal_verify.selection import rank_proposals
from visionlm_fov_matching.proposal_verify.zoom import local_fraction_to_box, unconstrained_zoom_policy
from visionlm_fov_matching.types import Proposal, ScoredProposal


def _proposal(box, source="primary") -> Proposal:
    return Proposal(tuple(box), source=source)


def test_image_scores_rank_the_exact_crop_first() -> None:
    rng = np.random.default_rng(9)
    template = rng.random((12, 10), dtype=np.float32)
    reference = rng.random((60, 70), dtype=np.float32) * 0.02
    reference[20:32, 30:40] = template
    ranked = rank_proposals(
        reference,
        template,
        (_proposal((30, 20, 10, 12)), _proposal((3, 4, 10, 12))),
        metric="nmi_raw",
    )
    assert ranked[0].proposal.box == (30, 20, 10, 12)
    assert set(ranked[0].scores) == {"nmi_raw", "zncc_percentile", "zncc_gradient", "edge_chamfer"}

    edge_ranked = rank_proposals(
        reference,
        template,
        (_proposal((30, 20, 10, 12)), _proposal((3, 4, 10, 12))),
        metric="edge_chamfer",
    )
    assert edge_ranked[0].proposal.box == (30, 20, 10, 12)


def test_nmi_uses_geometric_mean_entropy_normalization() -> None:
    image = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    assert np.isclose(normalized_mutual_information(image, image, bins=2), 1.0)


def test_controlled_policy_generates_ten_percent_offsets() -> None:
    rng = np.random.default_rng(11)
    reference = rng.random((80, 80), dtype=np.float32)
    template = reference[30:40, 30:40].copy()
    proposals = tuple(_proposal((30, 30, 10, 10)) for _ in range(3))
    result = controlled_metadata_policy(reference, template, proposals)
    boxes = {item.proposal.box for item in result.candidates}
    assert len(boxes) == 9
    assert (29, 29, 10, 10) in boxes
    assert (31, 31, 10, 10) in boxes
    assert result.selected is not None


def test_challenging_policy_uses_small_template_nmi_fallback(monkeypatch) -> None:
    metrics: list[str] = []

    def fake_rank(reference, template, proposals, *, metric):
        metrics.append(metric)
        return tuple(
            ScoredProposal(proposal, 1.0, {"nmi_raw": 1.0, "zncc_percentile": 0.0, "edge_chamfer": 0.0})
            for proposal in proposals[:1]
        )

    monkeypatch.setattr(policies, "rank_proposals", fake_rank)
    primary = tuple(_proposal((10 + index, 10, 5, 5)) for index in range(3))
    supplemental = tuple(_proposal((10 + index, 10, 5, 5), "supplemental") for index in range(3))
    result = challenging_metadata_policy(np.zeros((30, 30)), np.zeros((10, 10)), primary, supplemental)
    assert metrics == ["nmi_raw", "nmi_raw"]
    assert result.selection_rule == "primary_nmi_raw"

    metrics.clear()
    challenging_metadata_policy(np.zeros((400, 400)), np.zeros((200, 300)), primary, supplemental)
    assert metrics == ["edge_chamfer", "edge_chamfer"]


def test_challenging_supplement_requires_better_score_and_half_iou(monkeypatch) -> None:
    calls = 0

    def fake_rank(reference, template, proposals, *, metric):
        nonlocal calls
        calls += 1
        proposal = proposals[0]
        score = 1.0 if calls == 1 else 2.0
        return (ScoredProposal(proposal, score, {"nmi_raw": score, "zncc_percentile": 0.0, "edge_chamfer": score}),)

    monkeypatch.setattr(policies, "rank_proposals", fake_rank)
    primary = tuple(_proposal((10, 10, 10, 10)) for _ in range(3))
    supplemental = tuple(_proposal((12, 10, 10, 10), "supplemental") for _ in range(3))
    result = challenging_metadata_policy(np.zeros((400, 400)), np.zeros((250, 250)), primary, supplemental)
    assert result.selected is not None
    assert result.selected.proposal.source == "supplemental"
    assert result.selection_rule == "supplemental_edge_chamfer_overlap_guard"


def test_challenging_supplement_rejects_unrelated_better_candidate(monkeypatch) -> None:
    calls = 0

    def fake_rank(reference, template, proposals, *, metric):
        nonlocal calls
        calls += 1
        proposal = proposals[0]
        score = 1.0 if calls == 1 else 2.0
        return (ScoredProposal(proposal, score, {"nmi_raw": score, "zncc_percentile": 0.0, "edge_chamfer": score}),)

    monkeypatch.setattr(policies, "rank_proposals", fake_rank)
    primary = tuple(_proposal((10, 10, 10, 10)) for _ in range(3))
    supplemental = tuple(_proposal((30, 30, 10, 10), "supplemental") for _ in range(3))
    result = challenging_metadata_policy(np.zeros((400, 400)), np.zeros((250, 250)), primary, supplemental)
    assert result.selected is not None
    assert result.selected.proposal.source == "primary"
    assert result.selection_rule == "primary_edge_chamfer"


def test_zoom_maps_local_fractions_and_reranks_six_candidates() -> None:
    rng = np.random.default_rng(21)
    reference = rng.random((120, 140), dtype=np.float32)
    template = reference[50:60, 60:72].copy()
    direct = (
        _proposal((45, 40, 12, 10), "direct"),
        _proposal((60, 50, 12, 10), "direct"),
        _proposal((80, 70, 12, 10), "direct"),
    )
    result, crop = unconstrained_zoom_policy(reference, template, direct, ((0.1, 0.1), (0.5, 0.5), (0.9, 0.9)))
    assert len(result.candidates) == 6
    assert result.selected is not None
    mapped = local_fraction_to_box((0.5, 0.5), zoom_crop=crop, template_size=(12, 10))
    assert crop[0] <= mapped[0] <= crop[0] + crop[2] - 12
    assert crop[1] <= mapped[1] <= crop[1] + crop[3] - 10
