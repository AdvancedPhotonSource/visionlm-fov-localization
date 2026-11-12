from __future__ import annotations

from collections.abc import Sequence

import numpy as np


OWLV2_MODEL_ID = "google/owlv2-base-patch16-ensemble"


def load_owlv2(*, revision: str | None = None):
    """Load the OWLv2 detector used for image-guided detection; this may download model weights."""

    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    processor = Owlv2Processor.from_pretrained(OWLV2_MODEL_ID, revision=revision)
    model = Owlv2ForObjectDetection.from_pretrained(OWLV2_MODEL_ID, revision=revision)
    return model.eval(), processor


def boxes_from_detection(
    center_size_fraction: Sequence[float],
    *,
    reference_size: tuple[int, int],
    box_size: tuple[int, int],
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Free-size and fixed-size boxes from one OWLv2 box.

    OWLv2 returns ``(cx, cy, w, h)`` as fractions of the square it pads the image to, whose side is the
    image's long side. The free-size box is that box clipped to the image, and the fixed-size box centers
    ``box_size`` on it.
    """

    width, height = reference_size
    side = float(max(width, height))
    center_x, center_y, box_width, box_height = (float(value) * side for value in center_size_fraction)
    x0, y0 = max(0.0, center_x - box_width / 2), max(0.0, center_y - box_height / 2)
    x1, y1 = min(float(width), center_x + box_width / 2), min(float(height), center_y + box_height / 2)
    free_size_box = (x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0))
    fixed_width, fixed_height = box_size
    fixed_size_box = (center_x - fixed_width / 2, center_y - fixed_height / 2, float(fixed_width), float(fixed_height))
    return free_size_box, fixed_size_box


def owlv2_localize(
    model,
    processor,
    reference_rgb: np.ndarray,
    template_rgb: np.ndarray,
    *,
    box_size: tuple[int, int],
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float], float]:
    """Image-guided detection with the XRF map as the query and the optical image as the target.

    Returns the free-size box, the fixed-size box and the score of the highest-scoring target box.
    """

    import torch
    from PIL import Image

    reference = Image.fromarray(np.asarray(reference_rgb, dtype=np.uint8)).convert("RGB")
    template = Image.fromarray(np.asarray(template_rgb, dtype=np.uint8)).convert("RGB")
    inputs = processor(images=reference, query_images=template, return_tensors="pt")
    with torch.inference_mode():
        output = model.image_guided_detection(**inputs)
    scores = output.logits[0].reshape(-1)
    boxes = output.target_pred_boxes[0]
    if boxes.shape[0] != scores.shape[0]:
        raise ValueError(f"{boxes.shape[0]} boxes but {scores.shape[0]} scores")
    best = int(torch.argmax(scores))
    free_size_box, fixed_size_box = boxes_from_detection(
        boxes[best].tolist(),
        reference_size=reference.size,
        box_size=box_size,
    )
    return free_size_box, fixed_size_box, float(scores[best])
