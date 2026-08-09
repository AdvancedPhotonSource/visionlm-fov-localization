from __future__ import annotations

import numpy as np

from visionlm_fov_matching.direct_localization import ablation_grid
from visionlm_fov_matching.prompting import build_chat_payload, parse_prediction
from visionlm_fov_matching.types import Demonstration


def test_four_coordinate_formats_parse_to_boxes() -> None:
    arguments = {"reference_size": (100, 80), "template_size": (20, 10)}
    assert parse_prediction("top_left_xy", "30,25", **arguments) == (30, 25, 20, 10)
    assert parse_prediction("center_xy", "40,30", **arguments) == (30, 25, 20, 10)
    assert parse_prediction("bbox_xywh", "30,25,20,10", **arguments) == (30, 25, 20, 10)
    assert parse_prediction("frac_xy", "0.375,0.357142857", **arguments) == (30, 25, 20, 10)


def test_prediction_parsing_clips_to_valid_image_range() -> None:
    assert parse_prediction(
        "frac_xy",
        "2,-1",
        reference_size=(100, 80),
        template_size=(20, 10),
    ) == (80, 0, 20, 10)
    assert parse_prediction(
        "bbox_xywh",
        "95,75,20,20",
        reference_size=(100, 80),
        template_size=(20, 10),
    ) == (95, 75, 5, 5)


def test_payload_contains_ordered_demonstrations_without_transport_fields() -> None:
    demonstration = Demonstration("data:ref", ("data:tpl",), "0.2,0.3")
    payload = build_chat_payload(
        model="paper-model",
        strategy="frac_xy",
        reference_data_url="data:query-ref",
        template_data_urls=("data:query-tpl",),
        demonstrations=(demonstration,),
    )
    assert [message["role"] for message in payload["messages"]] == ["system", "user", "assistant", "user"]
    assert payload["messages"][2]["content"] == "0.2,0.3"
    assert set(payload) == {"model", "messages"}


def test_ablation_grid_has_six_unique_cells() -> None:
    grid = ablation_grid()
    assert len(grid) == 6
    assert ("top_left_xy", 3) in grid
    assert ("center_xy", 3) in grid
    assert [("frac_xy", count) in grid for count in range(4)] == [True] * 4
