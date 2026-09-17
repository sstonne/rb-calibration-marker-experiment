"""Exercise rejected ChArUco grabs and camera-map preflight without hardware."""

import json
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from calibration_pipeline.config import CharucoBoardConfig
from capture_pipeline.calibrate_intrinsics import (
    _diagnose_rejected_grab,
    _intrinsics_board_config,
    _make_intrinsics_target,
    main,
)


def _target(sx=6, sy=9, start=90, legacy=False):
    cfg = CharucoBoardConfig(
        squares_x=sx, squares_y=sy, square_length_m=0.025,
        marker_length_m=0.018, dictionary_name="DICT_4X4_250",
        marker_id_start=start,
    )
    return _make_intrinsics_target(cfg, legacy_pattern=legacy)


def _frame(target):
    gray = cv2.copyMakeBorder(
        target.generate_board_image(px_per_square=100),
        40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255,
    )
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_new_board_detects_all_markers_and_corners():
    target = _target()
    _, _, count, _, marker_ids = target.detect(_frame(target))
    assert count == 40
    assert set(marker_ids.flatten()) == set(range(90, 117))


@pytest.mark.parametrize("legacy", [False, True])
def test_rejected_grab_identifies_transposed_layout_without_changing_target(legacy):
    physical = _target(9, 6, legacy=legacy)
    target = _target()
    before = _intrinsics_board_config(target)
    frame = _frame(physical)
    count = target.detect(frame)[2]
    diagnostic = _diagnose_rejected_grab(frame, target, count, 8, 0, None)
    assert diagnostic["raw_marker_ids"] == list(range(90, 117))
    candidate = next(
        item for item in diagnostic["layout_candidates"]
        if item["squares_x"] == 9 and item["squares_y"] == 6
        and item["legacy_pattern"] == legacy
    )
    assert candidate["charuco_corners"] == 40
    assert _intrinsics_board_config(target) == before
    assert target.detect(_frame(target))[2] == 40


def test_wrong_ids_are_reported_and_original_rejected_frame_is_saved(tmp_path):
    frame = _frame(_target(start=5))
    target = _target()
    assert target.detect(frame)[2] == 0
    diagnostic = _diagnose_rejected_grab(frame, target, 0, 8, 2, str(tmp_path))
    assert diagnostic["raw_marker_ids"] == list(range(5, 32))
    assert diagnostic["matching_marker_ids"] == []
    assert diagnostic["layout_candidates"] == []
    pngs = list((tmp_path / "diagnostics").glob("rejected_*.png"))
    assert len(pngs) == 1
    assert np.array_equal(cv2.imread(str(pngs[0])), frame)
    saved = json.loads(pngs[0].with_suffix(".json").read_text())
    assert saved == diagnostic


def test_blank_frame_has_no_marker_or_layout_matches():
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    diagnostic = _diagnose_rejected_grab(frame, _target(), 0, 8, 0, None)
    assert diagnostic["raw_marker_ids"] == []
    assert diagnostic["matching_marker_ids"] == []
    assert diagnostic["layout_candidates"] == []


def test_legacy_target_updates_detector_and_report_metadata():
    target = _target(9, 6, legacy=True)
    assert _intrinsics_board_config(target)["legacy_pattern"] is True
    assert target.detect(_frame(target))[2] == 40


def _fake_cameras(monkeypatch, serials):
    class Camera:
        def __init__(self, *args, **kwargs):
            pytest.fail("Preflight must fail before opening any camera")

        @staticmethod
        def list_devices():
            return dict.fromkeys(serials, "RealSense")

        @staticmethod
        def reset_all_devices():
            pytest.fail("Camera reset must remain opt-in")

    monkeypatch.setitem(
        sys.modules, "capture_pipeline.camera", SimpleNamespace(RealSenseCamera=Camera),
    )


def test_stale_map_does_not_silently_calibrate_only_matching_cameras(tmp_path, monkeypatch):
    _fake_cameras(monkeypatch, ["new0", "same1", "new2", "same3"])
    (tmp_path / "device_map.json").write_text(json.dumps({
        "serial_to_idx": {"old0": 0, "same1": 1, "old2": 2, "same3": 3},
        "gripper_cam_idx": 2,
    }))
    monkeypatch.setattr(sys, "argv", ["02", "--intr_dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="Connected cameras missing"):
        main()
    assert not (tmp_path / "charuco_intrinsics_report.json").exists()


def test_npz_serial_must_match_device_map_before_opening_camera(tmp_path, monkeypatch):
    _fake_cameras(monkeypatch, ["camera-A"])
    (tmp_path / "device_map.json").write_text(json.dumps({
        "serial_to_idx": {"camera-A": 0}, "gripper_cam_idx": None,
    }))
    np.savez(tmp_path / "cam0.npz", serial="camera-B")
    monkeypatch.setattr(sys, "argv", ["02", "--intr_dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="does not match"):
        main()
