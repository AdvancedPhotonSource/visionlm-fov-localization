from __future__ import annotations

from pathlib import Path

import cv2
import h5py
import numpy as np


def load_optical_image(path: str | Path, *, grayscale: bool = True) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(f"could not read optical image: {path}")
    return image


def _decode_channel_names(raw: np.ndarray) -> list[str]:
    names: list[str] = []
    for value in np.asarray(raw).reshape(-1):
        if isinstance(value, bytes):
            names.append(value.decode("utf-8", errors="replace").strip())
        else:
            names.append(str(value).strip())
    return names


def load_xrf_channel(
    path: str | Path,
    *,
    channel: str = "P",
    dataset_key: str = "MAPS/XRF_fits",
    channel_names_key: str = "MAPS/channel_names",
    channel_axis: int = 0,
) -> np.ndarray:
    """Load one named XRF channel from a MAPS-style HDF5 file."""

    with h5py.File(path, "r") as handle:
        if dataset_key not in handle:
            raise KeyError(f"missing XRF dataset {dataset_key!r}")
        if channel_names_key not in handle:
            raise KeyError(f"missing channel names dataset {channel_names_key!r}")
        names = _decode_channel_names(np.asarray(handle[channel_names_key]))
        matches = [index for index, name in enumerate(names) if name.casefold() == channel.casefold()]
        if not matches:
            raise KeyError(f"channel {channel!r} is not present; available channels: {names}")
        cube = np.asarray(handle[dataset_key])
        if cube.ndim != 3:
            raise ValueError(f"expected a three-dimensional XRF cube, got shape {cube.shape}")
        image = np.take(cube, matches[0], axis=int(channel_axis))
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"selected XRF channel is not two-dimensional: {image.shape}")
    if not np.isfinite(image).all():
        raise ValueError("selected XRF channel contains NaN or infinite values")
    return image
