"""--joint_intr_dir: 같은 카메라의 두 해상도 사진을 합쳐 한 번에 보정한다.

1280x720 스트림은 1920x1080 을 정확히 1.5배 줄인 것이라(공장 K 도 정확히 비례)
따로 보정하면 주점이 ±10px 넘게 따로 흔들렸다. 합친 보정은 두 폴더의 K 가
정확히 배율 관계이고 D 가 같아야 한다.
"""

import json
import sys

import cv2
import numpy as np
import pytest

from calibration_pipeline.board_config import resolve_charuco_config
from calibration_pipeline.charuco import CharucoTarget
from capture_pipeline import calibrate_intrinsics as app
from capture_pipeline.intrinsics_images import IntrinsicsImages

SERIAL = "camera-A"
K1280 = np.array([[910.0, 0.0, 641.0], [0.0, 912.0, 363.0], [0.0, 0.0, 1.0]])
SCALE = 1.5


def _scaled(K, scale):
    out = K.copy()
    out[:2] *= scale
    return out


def _folder(path, K, size):
    path.mkdir(parents=True)
    (path / "device_map.json").write_text(json.dumps({
        "serial_to_idx": {SERIAL: 0}, "gripper_cam_idx": 0,
    }))
    np.savez(path / "cam0.npz", serial=SERIAL, is_gripper=True,
             color_K=K, color_D=np.zeros((5, 1)), color_w=size[0], color_h=size[1], fps=15,
             depth_K=np.eye(3), depth_D=np.zeros(5), depth_w=1280, depth_h=720,
             depth_scale_m_per_unit=0.001, R_depth_to_color=np.eye(3),
             t_depth_to_color=np.zeros(3))
    images = IntrinsicsImages(path / "raw_capture", {SERIAL: 0}, create=True)
    images.check_camera(0, SERIAL, size[0], size[1], 15, create=True)
    return images


def _render(K, size, poses):
    cfg, _ = resolve_charuco_config("9x6_id90")
    pattern = CharucoTarget(cfg).generate_board_image(px_per_square=160)
    h, w = pattern.shape
    source = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    bounds = np.float32([[0, 0, 0], [0.225, 0, 0], [0.225, 0.15, 0], [0, 0.15, 0]])
    for rvec, tvec in poses:
        projected = cv2.projectPoints(bounds, rvec, tvec, K, np.zeros(5))[0].reshape(-1, 2)
        H = cv2.getPerspectiveTransform(source, projected.astype(np.float32))
        gray = cv2.warpPerspective(pattern, H, size, flags=cv2.INTER_AREA, borderValue=255)
        yield cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _poses(count, seed):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        rvec = rng.uniform([-0.5, -0.5, -0.3], [0.5, 0.5, 0.3])
        R = cv2.Rodrigues(rvec)[0]
        center = np.array([rng.uniform(-0.12, 0.12), rng.uniform(-0.06, 0.06), rng.uniform(0.45, 0.6)])
        yield rvec, center - R @ np.array([0.1125, 0.075, 0])


def _two_folders(tmp_path, joint_factory_K=None):
    base = _folder(tmp_path / "i1280", K1280, (1280, 720))
    joint = _folder(tmp_path / "i1920", _scaled(K1280, SCALE) if joint_factory_K is None
                    else joint_factory_K, (1920, 1080))
    for frame in _render(K1280, (1280, 720), _poses(10, 1)):
        base.capture(0, frame)
    for frame in _render(_scaled(K1280, SCALE), (1920, 1080), _poses(10, 2)):
        joint.capture(0, frame)
    return tmp_path / "i1280", tmp_path / "i1920"


def _run(monkeypatch, *argv):
    monkeypatch.setitem(sys.modules, "pyrealsense2", None)
    monkeypatch.setitem(sys.modules, "capture_pipeline.camera", None)
    monkeypatch.setattr(sys, "argv", ["02", *map(str, argv)])
    app.main()


def test_joint_calibration_writes_exactly_scaled_K_and_shared_D(tmp_path, monkeypatch):
    base, joint = _two_folders(tmp_path)
    _run(monkeypatch, "--intr_dir", base, "--joint_intr_dir", joint, "--from_images",
         "--use_factory_guess", "--zero_tangent", "--board", "9x6_id90", "--min_views", "8")

    with np.load(base / "cam0.npz") as a, np.load(joint / "cam0.npz") as b:
        np.testing.assert_allclose(b["color_K"], _scaled(a["color_K"], SCALE), rtol=0, atol=1e-9)
        np.testing.assert_array_equal(a["color_D"], b["color_D"])
        assert a["color_D"][2, 0] == 0.0 and a["color_D"][3, 0] == 0.0  # zero_tangent
        np.testing.assert_allclose(a["color_K"], K1280, atol=3.0)
        np.testing.assert_array_equal(b["factory_color_K"], _scaled(K1280, SCALE))
        assert float(a["charuco_joint_scale_to_other"]) == SCALE
        assert float(b["charuco_joint_scale_to_other"]) == pytest.approx(1 / SCALE)
        assert int(b["depth_w"]) == 1280  # depth 필드는 그대로

    reports = [json.loads((d / "charuco_intrinsics_report.json").read_text()) for d in (base, joint)]
    for report, other in zip(reports, (joint, base)):
        camera = report["cameras"]["0"]
        assert camera["status"] == "written"
        assert camera["rms_px"] < 1.0 and camera["joint"]["other_rms_px"] < 1.5
        assert camera["joint"]["intr_dir"] == str(other.resolve())
        assert report["joint_with"]["intr_dir"] == str(other.resolve())
        assert report["calib_flags"]["zero_tangent"] is True
        assert report["board_source"].endswith("board_9x6_id90.json")
    assert reports[0]["cameras"]["0"]["num_images"] == 10
    assert reports[1]["cameras"]["0"]["num_images"] == 10
    assert reports[0]["cameras"]["0"]["joint"]["num_views_total_pooled"] == 20


def test_joint_unavailable_falls_back_to_this_folder_only(tmp_path, monkeypatch):
    cropped = _scaled(K1280, SCALE)
    cropped[0, 2] += 40.0  # 크롭된 스트림처럼 주점이 비례하지 않음
    base, joint = _two_folders(tmp_path, joint_factory_K=cropped)
    joint_npz = (joint / "cam0.npz").read_bytes()
    _run(monkeypatch, "--intr_dir", base, "--joint_intr_dir", joint, "--from_images",
         "--board", "9x6_id90", "--min_views", "8")

    camera = json.loads((base / "charuco_intrinsics_report.json").read_text())["cameras"]["0"]
    assert camera["status"] == "written" and "joint" not in camera
    assert "공장 K" in camera["joint_unavailable"]
    with np.load(base / "cam0.npz") as a:
        assert "charuco_joint_scale_to_other" not in a.files
    assert (joint / "cam0.npz").read_bytes() == joint_npz
    other = json.loads((joint / "charuco_intrinsics_report.json").read_text())["cameras"]["0"]
    assert other["status"] == "not_calibrated" and "공장 K" in other["joint_unavailable"]


def test_joint_never_overwrites_a_folder_without_its_own_views(tmp_path, monkeypatch):
    base = _folder(tmp_path / "i1280", K1280, (1280, 720))
    _folder(tmp_path / "i1920", _scaled(K1280, SCALE), (1920, 1080))  # 아직 촬영 전
    for frame in _render(K1280, (1280, 720), _poses(10, 1)):
        base.capture(0, frame)
    joint = tmp_path / "i1920"
    joint_npz = (joint / "cam0.npz").read_bytes()
    _run(monkeypatch, "--intr_dir", tmp_path / "i1280", "--joint_intr_dir", joint,
         "--from_images", "--board", "9x6_id90", "--min_views", "8")
    camera = json.loads((tmp_path / "i1280" / "charuco_intrinsics_report.json").read_text())["cameras"]["0"]
    assert camera["status"] == "written"
    assert "보드 검출 사진이 없음" in camera["joint_unavailable"]
    assert (joint / "cam0.npz").read_bytes() == joint_npz


def test_joint_report_keeps_entries_for_cameras_it_did_not_write(tmp_path, monkeypatch):
    base, joint = _two_folders(tmp_path)
    device_map = json.loads((joint / "device_map.json").read_text())
    device_map["serial_to_idx"]["camera-B"] = 1  # 합칠 폴더에만 있는 카메라
    (joint / "device_map.json").write_text(json.dumps(device_map))
    previous = {"calibrated_at": "earlier", "cameras": {
        "1": {"serial": "camera-B", "status": "written", "rms_px": 0.5},
        "0": {"serial": "camera-A", "status": "written", "rms_px": 9.9},
    }}
    (joint / "charuco_intrinsics_report.json").write_text(json.dumps(previous))
    _run(monkeypatch, "--intr_dir", base, "--joint_intr_dir", joint, "--from_images",
         "--use_factory_guess", "--board", "9x6_id90", "--min_views", "8")

    cameras = json.loads((joint / "charuco_intrinsics_report.json").read_text())["cameras"]
    assert cameras["0"]["rms_px"] < 1.5 and "joint" in cameras["0"]
    assert cameras["1"] == {
        "serial": "camera-B", "status": "written", "rms_px": 0.5,
        "previous_calibrated_at": "earlier",
        "joint_unavailable": "기준 폴더 실행에 없는 카메라",
    }


@pytest.mark.parametrize("extra, message", [
    (["--joint_intr_dir", "x"], "requires --from_images"),
    (["--from_images", "--joint_images_dir", "x"], "requires --joint_intr_dir"),
])
def test_joint_cli_validation(tmp_path, monkeypatch, capsys, extra, message):
    monkeypatch.setattr(sys, "argv", ["02", "--intr_dir", str(tmp_path), *extra])
    with pytest.raises(SystemExit):
        app.main()
    assert message in capsys.readouterr().err


def test_joint_rejects_its_own_folder(tmp_path, monkeypatch):
    base, _ = _two_folders(tmp_path)
    with pytest.raises(SystemExit, match="--intr_dir"):
        _run(monkeypatch, "--intr_dir", base, "--joint_intr_dir", base, "--from_images",
             "--board", "9x6_id90")
