from __future__ import annotations

import random
import re
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

from ..geometry import box_iou
from ..prompting import encode_png_data_url
from ..settings import PAPER_SETTINGS
from ..types import BoxXYWH, PolicyResult, Proposal, ScoredProposal
from .selection import rank_proposals


JUDGE_SYSTEM_PROMPT = (
    "You compare an X-ray fluorescence (XRF) elemental map with crops of an optical microscope image "
    "of an adjacent tissue section from the same specimen. Answer with a number only."
)
JUDGE_INSTRUCTION = (
    "The first image is an XRF phosphorus map of a scanned region. The following images are candidate "
    "crops of the optical microscope image. In each crop a rectangle outlines a proposed scan area; the "
    "XRF map has the size and orientation of that rectangle. Which candidate's outlined area shows the "
    "region in the XRF map? Answer with the candidate number only."
)
_ADJACENT_SECTION = "of an adjacent tissue section from the same specimen"
_SAME_SECTION = "of the same tissue section"
_FIXED_SIZE = "the XRF map has the size and orientation of that rectangle"
_FREE_SIZE = "the XRF map has the orientation of that rectangle"


def judge_prompts(*, same_section: bool, fixed_size: bool) -> tuple[str, str]:
    """System prompt and instruction for one dataset and search setting.

    The controlled collection images the same section, and in free-size search a candidate box need
    not have the XRF map's size, so those two phrases change and nothing else.
    """

    system_prompt = JUDGE_SYSTEM_PROMPT.replace(_ADJACENT_SECTION, _SAME_SECTION) if same_section else JUDGE_SYSTEM_PROMPT
    instruction = JUDGE_INSTRUCTION if fixed_size else JUDGE_INSTRUCTION.replace(_FIXED_SIZE, _FREE_SIZE)
    return system_prompt, instruction


def deduplicate_candidates(proposals: Sequence[Proposal], *, maximum_iou: float) -> tuple[Proposal, ...]:
    """Keep each candidate unless it overlaps an earlier kept one at ``maximum_iou`` or more."""

    kept: list[Proposal] = []
    for proposal in proposals:
        if all(box_iou(proposal.box, prior.box) < maximum_iou for prior in kept):
            kept.append(proposal)
    return tuple(kept)


def outlined_crop(display: np.ndarray, box: BoxXYWH, *, margin_fraction: float) -> np.ndarray:
    """The box enlarged by ``margin_fraction`` of its size on each side, clipped, with the box outlined.

    ``display`` is the whole optical image as sent in the proposal step (8-bit BGR), so every crop keeps
    the contrast the model saw there.
    """

    x, y, width, height = (int(value) for value in box)
    image_height, image_width = display.shape[:2]
    x0 = max(0, int(x - margin_fraction * width))
    y0 = max(0, int(y - margin_fraction * height))
    x1 = min(image_width, int(x + width + margin_fraction * width))
    y1 = min(image_height, int(y + height + margin_fraction * height))
    crop = display[y0:y1, x0:x1].copy()
    thickness = max(2, int(0.006 * max(crop.shape[:2])))
    cv2.rectangle(crop, (x - x0, y - y0), (x - x0 + width - 1, y - y0 + height - 1), (0, 255, 255), thickness)
    return crop


def judge_image_data_url(image: np.ndarray, *, max_side: int) -> str:
    """Lossless PNG data URL, downscaled with area interpolation to at most ``max_side`` pixels."""

    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return encode_png_data_url(image)


def build_judge_payload(
    *,
    model: str,
    template_data_url: str,
    crop_data_urls: Sequence[str],
    system_prompt: str,
    instruction: str,
) -> dict[str, Any]:
    """Construct one OpenAI-compatible judge request without performing network I/O."""

    content: list[dict[str, Any]] = [
        {"type": "text", "text": instruction},
        {"type": "image_url", "image_url": {"url": template_data_url}},
    ]
    for number, url in enumerate(crop_data_urls, start=1):
        content.append({"type": "text", "text": f"Candidate {number}:"})
        content.append({"type": "image_url", "image_url": {"url": url}})
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
    }


def judge_orders(candidate_count: int, *, calls: int, seed: str) -> tuple[tuple[int, ...], ...]:
    """Candidate orders for the judge calls, shuffled with the fixed seed ``f"{seed}-{call}"`` per call.

    The manuscript's seeds were ``"<model>-<map>"`` for the challenging collection in fixed-size search
    and ``"<model>-<collection>_<setting>-<map>"`` in the other three cells, for example
    ``"gpt5-controlled_free-85e863790b5c41fba65c556a9c83f38a"``.
    """

    orders = []
    for call in range(calls):
        order = list(range(candidate_count))
        random.Random(f"{seed}-{call}").shuffle(order)
        orders.append(tuple(order))
    return tuple(orders)


def judge_requests(
    display: np.ndarray,
    template_display: np.ndarray,
    candidates: Sequence[Proposal],
    orders: Sequence[Sequence[int]],
    *,
    model: str,
    same_section: bool,
    fixed_size: bool,
) -> tuple[dict[str, Any], ...]:
    """One request per call, showing the XRF map and then the outlined crops in that call's order."""

    settings = PAPER_SETTINGS.model_judge
    system_prompt, instruction = judge_prompts(same_section=same_section, fixed_size=fixed_size)
    template_url = judge_image_data_url(template_display, max_side=settings.image_max_side)
    crop_urls = [
        judge_image_data_url(
            outlined_crop(display, candidate.box, margin_fraction=settings.crop_margin_fraction),
            max_side=settings.image_max_side,
        )
        for candidate in candidates
    ]
    return tuple(
        build_judge_payload(
            model=model,
            template_data_url=template_url,
            crop_data_urls=[crop_urls[index] for index in order],
            system_prompt=system_prompt,
            instruction=instruction,
        )
        for order in orders
    )


def parse_judge_choice(text: str, candidate_count: int) -> int | None:
    """The 1-based candidate number a reply names, or ``None`` if it names none."""

    found = re.search(r"\d+", text or "")
    if found is None or not 1 <= int(found.group()) <= candidate_count:
        return None
    return int(found.group())


def judge_votes(
    candidate_count: int,
    orders: Sequence[Sequence[int]],
    response_texts: Sequence[str],
) -> tuple[int, ...]:
    """Votes per candidate from the replies to the calls made in ``orders``."""

    if len(orders) != len(response_texts):
        raise ValueError("one response is required per judge call")
    votes = [0] * candidate_count
    for order, text in zip(orders, response_texts):
        choice = parse_judge_choice(text, len(order))
        if choice is not None:
            votes[order[choice - 1]] += 1
    return tuple(votes)


def select_judged(
    ranked: Sequence[ScoredProposal],
    judged: Sequence[Proposal],
    votes: Sequence[int],
) -> PolicyResult:
    """Most votes wins, a tie goes to the best image score, and without any vote the top image score
    over all candidates is selected."""

    if len(judged) != len(votes):
        raise ValueError("one vote count is required per judged candidate")
    by_box = {item.proposal.box: item for item in ranked}
    top = max(votes, default=0)
    if top == 0:
        return PolicyResult(
            selected=ranked[0] if ranked else None,
            candidates=tuple(ranked),
            selection_rule="judge_fallback_image_score",
        )
    leaders = [candidate for candidate, count in zip(judged, votes) if count == top]
    selected = max(leaders, key=lambda candidate: by_box[candidate.box].primary_score)
    return PolicyResult(
        selected=by_box[selected.box],
        candidates=tuple(ranked),
        selection_rule="judge_votes" if len(leaders) == 1 else "judge_tie_image_score",
    )


def model_judge_policy(
    reference: np.ndarray,
    template: np.ndarray,
    proposals: Sequence[Proposal],
    *,
    metric: str,
    orders: Sequence[Sequence[int]],
    response_texts: Sequence[str],
) -> PolicyResult:
    """Select among the six direct answers with the proposing model's votes.

    ``proposals`` are the answers with demonstrations followed by those without, and ``orders`` index
    the candidates kept after deduplication, as passed to :func:`judge_requests`. ``metric`` is the
    image check's metric: in fixed-size search on the challenging collection, the one
    :func:`challenging_metadata_policy` picks for the template's area, and ``"nmi_raw"`` otherwise.
    """

    settings = PAPER_SETTINGS.model_judge
    expected = len(settings.candidate_demonstrations) * settings.candidate_repeats
    if len(proposals) != expected:
        raise ValueError(f"expected {expected} candidate proposals")
    ranked = rank_proposals(reference, template, proposals, metric=metric)
    judged = deduplicate_candidates(proposals, maximum_iou=settings.duplicate_iou)
    votes = judge_votes(len(judged), orders, response_texts)
    return select_judged(ranked, judged, votes)
