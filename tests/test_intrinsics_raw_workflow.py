"""Raw capture must not depend on a board; offline calibration must not use a camera."""

import json
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from calibration_pipeline.config import CharucoBoardConfig
from capture_pipeline import calibrate_intrinsics as app
from capture_pipeline.intrinsics_images import IntrinsicsImages


def _intrinsics(directory, mapping, width=1280, height=720):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "device_map.json").write_text(json.dumps({
        "serial_to_idx": mapping, "gripper_cam_idx": None,
    }))
    K = np.array([[900., 0, width / 2], [0, 910., height / 2], [0, 0, 1.]])
    for serial, idx in mapping.items():
        np.savez(directory / f"cam{idx}.npz", serial=serial, color_K=K,
                 color_D=np.zeros(5), color_w=width, color_h=height, fps=15,
                 depth_K=K * 2, depth_D=np.zeros(5), depth_w=width, depth_h=height,
                 depth_scale_m_per_unit=0.001, R_depth_to_color=np.eye(3),
                 t_depth_to_color=np.array([0.01, 0, 0]))
    return K


def _gui(monkeypatch, keys):
    for name in ("namedWindow", "imshow", "destroyWindow", "destroyAllWindows"):
        monkeypatch.setattr(cv2, name, lambda *args, **kwargs: None)
    iterator = iter(keys)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: next(iterator))


def _cameras(monkeypatch, mapping):
    opened, stopped = [], []

    class Camera:
        @staticmethod
        def list_devices():
            return dict.fromkeys(mapping, "RealSense")

        @staticmethod
        def reset_all_devices():
            pytest.fail("Raw capture must not reset cameras by default")

        def __init__(self, serial, width, height, fps, use_color, use_depth):
            assert use_color and not use_depth
            self.serial = serial
            self.frame = np.zeros((height, width, 3), dtype=np.uint8)
            opened.append(serial)

        def start(self, max_attempts):
            assert max_attempts == 1

        def get_latest(self):
            return self.frame, None, 0.0

        def stop(self):
            stopped.append(self.serial)

    monkeypatch.setitem(sys.modules, "capture_pipeline.camera", SimpleNamespace(RealSenseCamera=Camera))
    return opened, stopped


def test_raw_capture_all_four_cameras_ignores_board_and_minimums(tmp_path, monkeypatch):
    mapping = {f"serial-{i}": i for i in range(4)}
    _intrinsics(tmp_path, mapping, 64, 48)
    before = {p.name: p.read_bytes() for p in tmp_path.glob("cam*.npz")}
    opened, stopped = _cameras(monkeypatch, mapping)
    _gui(monkeypatch, [ord(" "), ord("c")] * 4)
    for name in ("_make_intrinsics_target", "calibrate_intrinsics", "_sharpness"):
        monkeypatch.setattr(app, name, lambda *a, **kw: pytest.fail("Raw mode must bypass detection/calibration"))
    monkeypatch.setattr(sys, "argv", [
        "02", "--intr_dir", str(tmp_path), "--capture_only", "--min_views", "20",
        "--min_corners", "999", "--dictionary", "NOT_A_DICTIONARY", "--squares_x", "0",
    ])
    app.main()
    images = IntrinsicsImages(tmp_path / "raw_capture", mapping)
    assert opened == stopped == list(mapping)
    for i in range(4):
        assert images.count(i) == 1
        frame = cv2.imread(str(images.image_path(images.frames(i)[0])))
        assert np.array_equal(frame, np.zeros((48, 64, 3), dtype=np.uint8))
    assert before == {p.name: p.read_bytes() for p in tmp_path.glob("cam*.npz")}
    assert not (tmp_path / "charuco_intrinsics_report.json").exists()


def test_raw_quit_and_resume_preserve_images(tmp_path, monkeypatch):
    mapping = {"camera-A": 0}
    _intrinsics(tmp_path, mapping, 64, 48)
    opened, stopped = _cameras(monkeypatch, mapping)
    monkeypatch.setattr(sys, "argv", ["02", "--intr_dir", str(tmp_path), "--capture_only"])
    _gui(monkeypatch, [ord(" "), ord("q")])
    app.main()
    images = IntrinsicsImages(tmp_path / "raw_capture", mapping)
    first_path = images.image_path(images.frames(0)[0])
    first_bytes = first_path.read_bytes()
    _gui(monkeypatch, [ord(" "), ord("c")])
    app.main()
    resumed = IntrinsicsImages(tmp_path / "raw_capture", mapping)
    assert resumed.count(0) == 2
    assert first_path.read_bytes() == first_bytes
    assert len(set(f["file"] for f in resumed.frames(0))) == 2
    assert opened == stopped == ["camera-A", "camera-A"]


def test_raw_undo_preserves_originals_and_checks_camera_profile(tmp_path):
    images = IntrinsicsImages(tmp_path, {"A": 0}, create=True)
    images.check_camera(0, "A", 64, 48, 15, create=True)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    first = images.capture(0, frame)
    second = images.capture(0, frame + 1)
    assert images.undo(0)
    assert images.count(0) == 1 and first.exists() and second.exists()
    images.capture(0, frame + 2)
    resumed = IntrinsicsImages(tmp_path, {"A": 0})
    assert [f["enabled"] for f in resumed.frames(0)] == [True, False, True]
    assert resumed.count(0) == 2
    before = resumed.manifest_path.read_bytes()
    with pytest.raises(ValueError, match="mapping"):
        IntrinsicsImages(tmp_path, {"B": 0})
    with pytest.raises(ValueError, match="resolution/FPS"):
        resumed.check_camera(0, "A", 1920, 1080, 15, create=True)
    assert resumed.manifest_path.read_bytes() == before


def _projected_boards(K, count=14):
    target = app._make_intrinsics_target(CharucoBoardConfig(
        squares_x=6, squares_y=9, marker_id_start=90,
    ))
    pattern = target.generate_board_image(px_per_square=160)
    h, w = pattern.shape
    source = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    bounds = np.float32([[0, 0, 0], [0.15, 0, 0], [0.15, 0.225, 0], [0, 0.225, 0]])
    for i in range(count):
        rvec = np.array([-0.3 + 0.13 * (i % 5), 0.3 - 0.16 * (i % 4), 0.07 * (i % 3 - 1)])
        R = cv2.Rodrigues(rvec)[0]
        center = np.array([-0.07 + 0.035 * (i % 5), -0.04 + 0.035 * (i % 3), 0.5 + 0.015 * i])
        tvec = center - R @ np.array([0.075, 0.1125, 0])
        projected = cv2.projectPoints(bounds, rvec, tvec, K, np.zeros(5))[0].reshape(-1, 2)
        H = cv2.getPerspectiveTransform(source, projected.astype(np.float32))
        gray = cv2.warpPerspective(pattern, H, (1280, 720), borderValue=255)
        yield cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_raw_manifest_accepts_newly_added_cameras(tmp_path):
    images = IntrinsicsImages(tmp_path, {"A": 0}, create=True)
    images.check_camera(0, "A", 64, 48, 15, create=True)
    images.capture(0, np.zeros((48, 64, 3), dtype=np.uint8))
    extended = IntrinsicsImages(tmp_path, {"A": 0, "C": 1}, create=True)
    extended.check_camera(1, "C", 64, 48, 15, create=True)
    assert extended.count(0) == 1
    assert IntrinsicsImages(tmp_path, {"A": 0, "C": 1}).manifest["serial_to_idx"] == {"A": 0, "C": 1}
    with pytest.raises(ValueError, match="mapping"):
        IntrinsicsImages(tmp_path, {"A": 1, "C": 0})
    with pytest.raises(ValueError, match="mapping"):
        IntrinsicsImages(tmp_path, {"C": 1})
    with pytest.raises(ValueError, match="mapping"):
        IntrinsicsImages(tmp_path, {"A": 0, "C": 1, "D": 1})


def test_offline_corrects_definition_without_recapture_or_camera_access(tmp_path, monkeypatch):
    mapping = {"camera-A": 0}
    K = _intrinsics(tmp_path, mapping)
    images = IntrinsicsImages(tmp_path / "raw_capture", mapping, create=True)
    images.check_camera(0, "camera-A", 1280, 720, 15, create=True)
    for frame in _projected_boards(K):
        images.capture(0, frame)
    images.capture(0, np.full((720, 1280, 3), 255, dtype=np.uint8))
    wrong_size = images.capture(0, frame)
    cv2.imwrite(str(wrong_size), np.zeros((480, 640, 3), dtype=np.uint8))
    images.capture(0, frame)
    images.undo(0)
    image_bytes = {p: p.read_bytes() for p in images.directory.glob("cam0/*.png")}
    npz_bytes = (tmp_path / "cam0.npz").read_bytes()
    monkeypatch.setitem(sys.modules, "pyrealsense2", None)
    monkeypatch.setitem(sys.modules, "capture_pipeline.camera", None)
    for name in ("namedWindow", "imshow", "waitKey", "destroyAllWindows"):
        monkeypatch.setattr(cv2, name, lambda *a, **kw: pytest.fail("Offline mode must not use GUI"))
    command = ["02", "--intr_dir", str(tmp_path), "--from_images", "--min_views", "8", "--use_factory_guess"]
    monkeypatch.setattr(sys, "argv", command)
    app.main()
    wrong = json.loads((tmp_path / "charuco_intrinsics_report.json").read_text())
    assert wrong["cameras"]["0"]["status"] == "too_few_views"
    assert (tmp_path / "cam0.npz").read_bytes() == npz_bytes
    monkeypatch.setattr(sys, "argv", command + [
        "--squares_x", "6", "--squares_y", "9", "--marker_id_start", "90",
    ])
    app.main()
    report = json.loads((tmp_path / "charuco_intrinsics_report.json").read_text())
    camera = report["cameras"]["0"]
    assert report["mode"] == "from_images"
    assert camera["status"] == "written"
    assert camera["num_views_total"] == 14
    assert camera["rms_px"] < 1.0
    assert {d["status"] for d in camera["image_observations"]} == {
        "detected", "too_few_corners", "resolution_mismatch", "excluded_by_user",
    }
    with np.load(tmp_path / "cam0.npz") as calibrated:
        np.testing.assert_allclose(calibrated["color_K"][[0, 1], [0, 1]], K[[0, 1], [0, 1]], rtol=0.08)
        np.testing.assert_array_equal(calibrated["factory_color_K"], K)
        np.testing.assert_array_equal(calibrated["depth_K"], K * 2)
        assert calibrated["color_w"] == 1280 and calibrated["color_h"] == 720
    assert (tmp_path / "factory_backup" / "cam0.npz").read_bytes() == npz_bytes
    assert image_bytes == {p: p.read_bytes() for p in images.directory.glob("cam0/*.png")}
