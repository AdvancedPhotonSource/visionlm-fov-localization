from __future__ import annotations

import hashlib
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..preprocessing import normalize_unit_interval, resize_template
from ..types import BoxXYWH


def centered_moving_canvas(
    reference: np.ndarray,
    template: np.ndarray,
    *,
    box_size: tuple[int, int],
    seed_label: str,
) -> tuple[np.ndarray, BoxXYWH]:
    """Create the centered moving image used for the multiGradICON adaptation."""

    reference_height, reference_width = reference.shape[:2]
    box_width, box_height = box_size
    if box_width > reference_width or box_height > reference_height:
        raise ValueError("box must fit inside the reference image")
    resized = normalize_unit_interval(resize_template(template, box_size))
    seed = int.from_bytes(hashlib.sha256(seed_label.encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed % (2**31))
    canvas = rng.random((reference_height, reference_width), dtype=np.float32) * 1e-3
    x = (reference_width - box_width) // 2
    y = (reference_height - box_height) // 2
    canvas[y : y + box_height, x : x + box_width] = resized
    return canvas, (x, y, box_width, box_height)


def warped_content_box(
    warped: np.ndarray,
    *,
    box_size: tuple[int, int],
    fallback: BoxXYWH,
) -> BoxXYWH:
    values = np.asarray(warped, dtype=np.float32).squeeze()
    threshold = max(0.05, float(values.max()) * 0.20)
    rows, columns = np.where(values > threshold)
    if rows.size == 0:
        return fallback
    box_width, box_height = box_size
    center_x = (float(columns.min()) + float(columns.max())) / 2.0
    center_y = (float(rows.min()) + float(rows.max())) / 2.0
    reference_height, reference_width = values.shape
    x = min(max(int(round(center_x - box_width / 2.0)), 0), reference_width - box_width)
    y = min(max(int(round(center_y - box_height / 2.0)), 0), reference_height - box_height)
    return x, y, box_width, box_height


def run_multigradicon(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    executable: str | Path,
    timeout_seconds: int = 900,
) -> np.ndarray:
    """Run ``unigradicon-register`` in an isolated temporary directory."""

    import nibabel as nib

    with tempfile.TemporaryDirectory(prefix="multigradicon-") as directory:
        root = Path(directory)
        fixed_path = root / "fixed.nii.gz"
        moving_path = root / "moving.nii.gz"
        transform_path = root / "transform.hdf5"
        warped_path = root / "warped.nii.gz"
        nib.save(nib.Nifti1Image(reference[:, :, None].astype(np.float32), np.eye(4)), fixed_path)
        nib.save(nib.Nifti1Image(moving[:, :, None].astype(np.float32), np.eye(4)), moving_path)
        command = [
            str(executable),
            "--fixed", str(fixed_path),
            "--moving", str(moving_path),
            "--fixed_modality", "mri",
            "--moving_modality", "mri",
            "--transform_out", str(transform_path),
            "--warped_moving_out", str(warped_path),
            "--model", "multigradicon",
            "--io_iterations", "30",
        ]
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout_seconds)
        return np.asarray(nib.load(warped_path).get_fdata(), dtype=np.float32).squeeze()
