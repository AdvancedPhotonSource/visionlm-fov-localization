from __future__ import annotations

import h5py
import numpy as np

from visionlm_fov_matching.data import load_xrf_channel
from visionlm_fov_matching.geometry import box_iou, polygon_iou
from visionlm_fov_matching.preprocessing import prepare_localization_input


def test_axis_aligned_and_polygon_iou() -> None:
    assert box_iou((0, 0, 10, 10), (5, 0, 10, 10)) == 1.0 / 3.0
    first = ((0, 0), (10, 0), (10, 10), (0, 10))
    second = ((5, 0), (15, 0), (15, 10), (5, 10))
    assert np.isclose(polygon_iou(first, second), 1.0 / 3.0)


def test_phosphorus_channel_loading_and_fixed_size_preparation(tmp_path) -> None:
    path = tmp_path / "maps.h5"
    cube = np.stack((np.zeros((3, 4)), np.arange(12).reshape(3, 4), np.ones((3, 4))))
    with h5py.File(path, "w") as handle:
        handle.create_dataset("MAPS/XRF_fits", data=cube)
        handle.create_dataset("MAPS/channel_names", data=np.asarray([b"Ca", b"P", b"Zn"]))
    phosphorus = load_xrf_channel(path)
    assert phosphorus.shape == (3, 4)
    assert np.array_equal(phosphorus, cube[1].astype(np.float32))

    reference = np.arange(100, dtype=np.uint8).reshape(10, 10)
    prepared = prepare_localization_input(reference, phosphorus, fixed_box_size=(6, 5))
    assert prepared.reference.shape == (10, 10)
    assert prepared.template.shape == (5, 6)
    assert prepared.fixed_box_size == (6, 5)
    assert prepared.reference.dtype == np.float32
