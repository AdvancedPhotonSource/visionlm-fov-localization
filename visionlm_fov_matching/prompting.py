from __future__ import annotations

import base64
import re
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

from .geometry import clip_box
from .types import BoxXYWH, Demonstration


SYSTEM_PROMPT = (
    "You are an expert in multimodal microscopy image localization. "
    "You will receive one visible-light microscopy reference image first, "
    "followed by one or more X-ray fluorescence template images that correspond "
    "to the same specimen. Always treat the first image as the reference and all "
    "remaining images as templates. Return only the requested numeric values."
)


def prompt_instruction(strategy: str) -> str:
    instructions = {
        "top_left_xy": (
            "Return the pixel coordinates of the top-left corner of the template "
            "location. Return only two integers 'x,y'."
        ),
        "center_xy": (
            "Return the pixel coordinates of the center of the template location. "
            "Return only two integers 'x,y'."
        ),
        "bbox_xywh": (
            "Return the template bounding rectangle in reference-image pixels. "
            "Return only four integers 'x,y,w,h'."
        ),
        "frac_xy": (
            "Return the relative top-left position over the valid placement range "
            "as two fractions in [0,1]. Return only 'fx,fy'."
        ),
    }
    try:
        return instructions[strategy]
    except KeyError as exc:
        raise ValueError(f"unsupported prompt strategy: {strategy!r}") from exc


def encode_png_data_url(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", np.asarray(image))
    if not success:
        raise ValueError("could not encode image as PNG")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _image_content(reference_url: str, template_urls: Sequence[str], instruction: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": instruction},
        {"type": "image_url", "image_url": {"url": reference_url}},
    ]
    content.extend(
        {"type": "image_url", "image_url": {"url": template_url}}
        for template_url in template_urls
    )
    return content


def build_chat_payload(
    *,
    model: str,
    strategy: str,
    reference_data_url: str,
    template_data_urls: Sequence[str],
    demonstrations: Sequence[Demonstration] = (),
    system_prompt: str = SYSTEM_PROMPT,
) -> dict[str, Any]:
    """Construct an OpenAI-compatible payload without performing network I/O."""

    instruction = prompt_instruction(strategy)
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for demonstration in demonstrations:
        messages.append(
            {
                "role": "user",
                "content": _image_content(
                    demonstration.reference_data_url,
                    demonstration.template_data_urls,
                    instruction,
                ),
            }
        )
        messages.append({"role": "assistant", "content": demonstration.answer})
    messages.append(
        {
            "role": "user",
            "content": _image_content(reference_data_url, template_data_urls, instruction),
        }
    )
    return {"model": model, "messages": messages}


def _numbers(text: str) -> list[float]:
    matches = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", text)
    return [float(value) for value in matches]


def parse_prediction(
    strategy: str,
    text: str,
    *,
    reference_size: tuple[int, int],
    template_size: tuple[int, int],
) -> BoxXYWH | None:
    """Parse one of the four manuscript coordinate formats into ``xywh``."""

    values = _numbers(text or "")
    reference_width, reference_height = reference_size
    template_width, template_height = template_size
    if strategy == "bbox_xywh":
        if len(values) < 4:
            return None
        raw = tuple(int(round(value)) for value in values[:4])
        return clip_box(raw, reference_width, reference_height, preserve_size=False)
    if len(values) < 2:
        return None
    first, second = values[:2]
    if strategy == "top_left_xy":
        x, y = int(round(first)), int(round(second))
    elif strategy == "center_xy":
        x = int(round(first - template_width / 2.0))
        y = int(round(second - template_height / 2.0))
    elif strategy == "frac_xy":
        fraction_x = min(max(float(first), 0.0), 1.0)
        fraction_y = min(max(float(second), 0.0), 1.0)
        x = int(round(fraction_x * max(reference_width - template_width, 0)))
        y = int(round(fraction_y * max(reference_height - template_height, 0)))
    else:
        raise ValueError(f"unsupported prompt strategy: {strategy!r}")
    return clip_box(
        (x, y, template_width, template_height),
        reference_width,
        reference_height,
    )


def parsed_fraction(text: str) -> tuple[float, float] | None:
    values = _numbers(text or "")
    if len(values) < 2:
        return None
    return min(max(values[0], 0.0), 1.0), min(max(values[1], 0.0), 1.0)
