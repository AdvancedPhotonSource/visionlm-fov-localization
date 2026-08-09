from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .prompting import build_chat_payload, encode_png_data_url, parse_prediction
from .settings import PAPER_SETTINGS
from .types import BoxXYWH, Demonstration, LocalizationInput


def build_direct_payload(
    inputs: LocalizationInput,
    *,
    model: str,
    strategy: str,
    demonstrations: Sequence[Demonstration] = (),
) -> dict[str, object]:
    return build_chat_payload(
        model=model,
        strategy=strategy,
        reference_data_url=encode_png_data_url(inputs.reference),
        template_data_urls=(encode_png_data_url(inputs.template),),
        demonstrations=demonstrations,
    )


def parse_direct_response(
    inputs: LocalizationInput,
    *,
    strategy: str,
    response_text: str,
) -> BoxXYWH | None:
    reference_height, reference_width = inputs.reference.shape[:2]
    template_height, template_width = inputs.template.shape[:2]
    return parse_prediction(
        strategy,
        response_text,
        reference_size=(reference_width, reference_height),
        template_size=(template_width, template_height),
    )


def mean_prediction(predictions: Sequence[BoxXYWH]) -> tuple[float, float, float, float]:
    if not predictions:
        raise ValueError("at least one prediction is required")
    values = np.asarray(predictions, dtype=float)
    return tuple(float(value) for value in values.mean(axis=0))


def ablation_grid() -> tuple[tuple[str, int], ...]:
    """Return the unique output-format/few-shot cells used in the appendix."""

    cells = [(strategy, 3) for strategy in PAPER_SETTINGS.ablation_strategies]
    cells.extend(("frac_xy", count) for count in PAPER_SETTINGS.ablation_demonstration_counts)
    return tuple(dict.fromkeys(cells))
