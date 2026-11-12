from __future__ import annotations

import numpy as np

from visionlm_fov_matching.proposal_verify.judge import (
    deduplicate_candidates,
    judge_orders,
    judge_prompts,
    judge_requests,
    judge_votes,
    model_judge_policy,
    select_judged,
)
from visionlm_fov_matching.types import Proposal, ScoredProposal


def _scored(box, score) -> ScoredProposal:
    return ScoredProposal(Proposal(tuple(box), source="direct"), primary_score=score, scores={})


def test_a_tie_goes_to_the_better_image_score_and_no_vote_falls_back_to_all_candidates() -> None:
    weak, strong, unjudged = _scored((0, 0, 10, 10), -5.0), _scored((40, 0, 10, 10), -1.0), _scored((80, 0, 10, 10), 3.0)
    ranked = (unjudged, strong, weak)
    judged = (weak.proposal, strong.proposal)
    tie = select_judged(ranked, judged, (1, 1))
    assert tie.selected.proposal.box == (40, 0, 10, 10)
    assert tie.selection_rule == "judge_tie_image_score"
    assert select_judged(ranked, judged, (2, 1)).selected.proposal.box == (0, 0, 10, 10)
    # Without a usable vote the image check over every candidate decides, including dropped duplicates.
    assert select_judged(ranked, judged, (0, 0)).selected.proposal.box == (80, 0, 10, 10)


def test_duplicates_are_dropped_at_the_threshold_itself() -> None:
    first = Proposal((0, 0, 10, 10), source="direct")
    at_threshold = Proposal((0, 0, 10, 8), source="direct")  # IoU 0.8 with the first
    below = Proposal((0, 0, 10, 7), source="direct")  # IoU 0.7
    kept = deduplicate_candidates((first, at_threshold, below), maximum_iou=0.8)
    assert kept == (first, below)


def test_votes_follow_each_call_order_and_ignore_unusable_replies() -> None:
    orders = ((2, 0, 1), (1, 2, 0), (0, 1, 2))
    assert judge_votes(3, orders, ("2", "Candidate 3", "none")) == (2, 0, 0)
    assert judge_votes(3, orders, ("4", "", "0")) == (0, 0, 0)


def test_prompt_variants_drop_the_claims_that_would_be_false() -> None:
    controlled_system, _ = judge_prompts(same_section=True, fixed_size=True)
    _, free_instruction = judge_prompts(same_section=False, fixed_size=False)
    assert "adjacent" not in controlled_system and "same tissue section" in controlled_system
    assert "size and orientation" not in free_instruction and "orientation of that rectangle" in free_instruction


def test_requests_show_the_crops_in_each_call_order() -> None:
    display = np.zeros((60, 80, 3), dtype=np.uint8)
    display[:, 40:] = 200
    template = np.full((10, 10), 128, dtype=np.uint8)
    candidates = (Proposal((5, 5, 10, 10), source="direct"), Proposal((55, 5, 10, 10), source="direct"))
    orders = ((1, 0), (0, 1))
    requests = judge_requests(display, template, candidates, orders, model="m", same_section=False, fixed_size=True)
    plain = judge_requests(display, template, candidates, ((0, 1),), model="m", same_section=False, fixed_size=True)[0]
    crop_urls = [item["image_url"]["url"] for item in plain["messages"][1]["content"][3::2]]
    for order, request in zip(orders, requests):
        sent = [item["image_url"]["url"] for item in request["messages"][1]["content"][3::2]]
        assert sent == [crop_urls[index] for index in order]


def test_policy_selects_the_voted_candidate_over_the_image_score() -> None:
    rng = np.random.default_rng(5)
    template = rng.random((10, 10), dtype=np.float32)
    reference = rng.random((60, 80), dtype=np.float32) * 0.02
    reference[5:15, 5:15] = template
    boxes = [(5, 5, 10, 10), (5, 5, 10, 10), (5, 6, 10, 10), (50, 40, 10, 10), (50, 40, 10, 10), (20, 30, 10, 10)]
    proposals = tuple(Proposal(box, source="direct") for box in boxes)
    # Three candidates survive deduplication; every call names the third in its own order.
    orders = ((2, 1, 0), (0, 2, 1), (1, 0, 2))
    result = model_judge_policy(
        reference,
        template,
        proposals,
        metric="nmi_raw",
        orders=orders,
        response_texts=("1", "2", "3"),
    )
    assert result.selected.proposal.box == (20, 30, 10, 10)
    assert result.candidates[0].proposal.box == (5, 5, 10, 10)


def test_orders_reproduce_the_manuscript_run_seeds() -> None:
    # Recorded orders of the controlled collection, free-size search, for one map in the manuscript run.
    orders = judge_orders(6, calls=3, seed="gpt5-controlled_free-85e863790b5c41fba65c556a9c83f38a")
    assert orders == ((2, 5, 4, 3, 1, 0), (3, 4, 2, 0, 5, 1), (2, 5, 3, 4, 1, 0))
