from __future__ import annotations

import numpy as np

from ..types import BoxXYWH


DINO_MODEL_ID = "facebook/dinov2-base"
PATCH_SIZE = 14


def load_dinov2(*, device: str = "cpu", revision: str | None = None):
    """Load the manuscript DINOv2 baseline; this may download model weights."""

    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(DINO_MODEL_ID, revision=revision)
    model = AutoModel.from_pretrained(DINO_MODEL_ID, revision=revision)
    return model.eval().to(device), processor


def dense_feature_window_match(
    reference_features: np.ndarray,
    template_features: np.ndarray,
    *,
    box_size_pixels: tuple[int, int],
    patch_size: int = PATCH_SIZE,
) -> tuple[BoxXYWH, float]:
    """Match dense ``(channels, height, width)`` feature maps by cosine score."""

    reference = np.asarray(reference_features, dtype=np.float32)
    template = np.asarray(template_features, dtype=np.float32)
    if reference.ndim != 3 or template.ndim != 3 or reference.shape[0] != template.shape[0]:
        raise ValueError("feature maps must be channels-by-height-by-width with aligned channels")
    mean_template = template.mean(axis=(1, 2))
    mean_template /= max(float(np.linalg.norm(mean_template)), 1e-12)
    reference_norm = reference / np.maximum(np.linalg.norm(reference, axis=0, keepdims=True), 1e-12)
    similarity = np.einsum("chw,c->hw", reference_norm, mean_template)
    template_height, template_width = template.shape[1:]
    if template_height > similarity.shape[0] or template_width > similarity.shape[1]:
        raise ValueError("template feature map must fit inside the reference feature map")
    integral = cv_integral(similarity)
    window_sums = (
        integral[template_height:, template_width:]
        - integral[:-template_height, template_width:]
        - integral[template_height:, :-template_width]
        + integral[:-template_height, :-template_width]
    )
    row, column = np.unravel_index(int(np.argmax(window_sums)), window_sums.shape)
    score = float(window_sums[row, column] / (template_height * template_width))
    box_width, box_height = box_size_pixels
    return (int(column * patch_size), int(row * patch_size), box_width, box_height), score


def cv_integral(array: np.ndarray) -> np.ndarray:
    """Integral image with a zero top row and left column."""

    values = np.asarray(array, dtype=np.float64)
    return np.pad(values.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
