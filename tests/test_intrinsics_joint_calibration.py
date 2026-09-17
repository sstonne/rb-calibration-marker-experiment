"""--joint_intr_dir: 같은 카메라의 여러 해상도 사진을 합쳐 한 번에 보정한다.

RealSense color 스트림은 한 센서를 배율/잘림으로 만든 것이라 (1280 = 1920 ÷1.5,
848 = 1920 ÷2.25 에서 가로 양끝 잘림, 640 = 1920 ÷3) 공장 K 도 그 관계를 그대로
따른다. 해상도마다 따로 보정하면 주점이 ±10px 넘게 따로 흔들렸다. 합친 보정은
모든 폴더의 K 가 공장 K 관계를 정확히 따르고 D 가 같아야 한다.
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
K1920 = np.array([[1365.0, 0.0, 961.5], [0.0, 1368.0, 544.5], [0.0, 0.0, 1.0]])


def _stream_K(scale, offset=(0.0, 0.0)):
    """1920 픽셀 -> 스트림 픽셀: u' = u/scale - offset/scale (848 은 가로 잘림)."""
    K = K1920.copy()
    K[:2] /= scale
    K[0, 2] -= offset[0] / scale
    K[1, 2] -= offset[1] / scale
    K[2, 2] = 1.0
    return K


STREAMS = {  # name: (size, factory K)
    "i1280": ((1280, 720), _stream_K(1.5)),
    "i1920": ((1920, 1080), K1920),
    "i848": ((848, 480), _stream_K(2.25, (6.0, 0.0))),
}


def _folder(path, K, size, serial_to_idx=None):
    path.mkdir(parents=True)
    mapping = serial_to_idx or {SERIAL: 0}
    (path / "device_map.json").write_text(json.dumps({
        "serial_to_idx": mapping, "gripper_cam_idx": 0,
    }))
    np.savez(path / "cam0.npz", serial=SERIAL, is_gripper=True,
             color_K=K, color_D=np.zeros((5, 1)), color_w=size[0], color_h=size[1], fps=15,
             depth_K=np.eye(3), depth_D=np.zeros(5), depth_w=1280, depth_h=720,
             depth_scale_m_per_unit=0.001, R_depth_to_color=np.eye(3),
             t_depth_to_color=np.zeros(3))
    images = IntrinsicsImages(path / "raw_capture", mapping, create=True)
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


def _make(tmp_path, name, seed, factory_K=None, frames=10, **kwargs):
    size, true_K = STREAMS[name]
    images = _folder(tmp_path / name, true_K if factory_K is None else factory_K, size, **kwargs)
    for frame in _render(true_K, size, _poses(frames, seed)):
        images.capture(0, frame)
    return tmp_path / name


def _run(monkeypatch, *argv):
    monkeypatch.setitem(sys.modules, "pyrealsense2", None)
    monkeypatch.setitem(sys.modules, "capture_pipeline.camera", None)
    monkeypatch.setattr(sys, "argv", ["02", *map(str, argv)])
    app.main()


def _report(folder):
    return json.loads((folder / "charuco_intrinsics_report.json").read_text())


def test_joint_across_three_streams_follows_factory_relation_exactly(tmp_path, monkeypatch):
    base = _make(tmp_path, "i1920", 1)
    others = [_make(tmp_path, "i1280", 2), _make(tmp_path, "i848", 3)]
    _run(monkeypatch, "--intr_dir", base, "--joint_intr_dir", *others, "--from_images",
         "--use_factory_guess", "--zero_tangent", "--board", "9x6_id90", "--min_views", "8")

    with np.load(base / "cam0.npz") as b:
        K_base, D_base = b["color_K"], b["color_D"]
        np.testing.assert_allclose(K_base, K1920, atol=4.0)
        assert D_base[2, 0] == 0.0 and D_base[3, 0] == 0.0  # zero_tangent
    for folder in others:
        with np.load(folder / "cam0.npz") as z:
            stream_map = app._stream_map(K1920, z["factory_color_K"])
            np.testing.assert_allclose(z["color_K"], app._map_K(K_base, stream_map), atol=1e-9)
            np.testing.assert_array_equal(z["color_D"], D_base)
            scale, ox, oy = z["charuco_joint_map_from_base"]
            assert (scale, ox, oy) == pytest.approx(
                (stream_map["scale"], *stream_map["offset"]))
            assert json.loads(str(z["charuco_joint_dirs"]))[0] == str(base.resolve())
            assert int(z["depth_w"]) == 1280  # depth 필드는 그대로
    with np.load(others[1] / "cam0.npz") as z:  # 848: 가로 잘림 오프셋이 살아 있다
        assert z["charuco_joint_map_from_base"][1] == pytest.approx(-6.0 / 2.25)

    for folder in [base, *others]:
        report = _report(folder)
        camera = report["cameras"]["0"]
        assert camera["status"] == "written" and camera["rms_px"] < 1.0
        members = camera["joint"]["members"]
        assert [m["intr_dir"] for m in members] == [str(p.resolve()) for p in [base, *others]]
        assert all(m["num_views_used"] > 0 for m in members)
        assert camera["joint"]["num_views_total_pooled"] == 30
        assert {m["intr_dir"] for m in report["joint_with"]} == {
            str(p.resolve()) for p in [base, *others] if p != folder}
        assert report["calib_flags"]["zero_tangent"] is True
        assert report["board_source"].endswith("board_9x6_id90.json")


def test_outlier_threshold_is_per_folder():
    board = CharucoTarget(resolve_charuco_config("9x6_id90")[0])
    views = []
    for name, seed, noise in (("i1280", 1, 0.05), ("i848", 2, 0.6)):
        size, K = STREAMS[name]
        rng = np.random.default_rng(seed)
        for frame in _render(K, size, _poses(8, seed)):
            corners, ids = board.detect(frame)[:2]
            corners = corners + rng.normal(0, noise, corners.shape).astype(np.float32)
            views.append((app._to_base_pixels(corners, app._stream_map(STREAMS["i1280"][1], K)), ids))
    groups = [0] * 8 + [1] * 8
    result = app.calibrate_intrinsics(board.board, views, (1280, 720), 0,
                                      K0=STREAMS["i1280"][1].copy(), groups=groups)
    kept = [groups[i] for i in result["used_index"]]
    assert kept.count(1) >= 5, "잡음이 큰 폴더도 자기 기준으로만 걸러져야 한다"
    assert set(result["reject_thr_px"]) == {0, 1}


def test_folder_without_its_own_views_is_left_alone(tmp_path, monkeypatch):
    base = _make(tmp_path, "i1280", 1)
    empty = _make(tmp_path, "i1920", 2, frames=0)  # 아직 촬영 전
    empty_npz = (empty / "cam0.npz").read_bytes()
    _run(monkeypatch, "--intr_dir", base, "--joint_intr_dir", empty,
         "--from_images", "--board", "9x6_id90", "--min_views", "8")
    camera = _report(base)["cameras"]["0"]
    assert camera["status"] == "written" and "joint" not in camera
    assert "보드 검출 사진이 없음" in camera["joint_unavailable"][str(empty.resolve())]
    with np.load(base / "cam0.npz") as z:
        assert not any(k.startswith("charuco_joint_") for k in z.files)
    assert (empty / "cam0.npz").read_bytes() == empty_npz
    other = _report(empty)["cameras"]["0"]
    assert other["status"] == "not_calibrated" and "보드 검출 사진이 없음" in other["joint_unavailable"]


def test_non_uniform_scale_is_excluded_but_other_folders_still_join(tmp_path, monkeypatch):
    base = _make(tmp_path, "i1920", 1)
    squeezed_K = STREAMS["i1280"][1].copy()
    squeezed_K[1, 1] *= 1.01  # 세로 배율만 다름 = 같은 화각 아님
    squeezed = _make(tmp_path, "i1280", 2, factory_K=squeezed_K)
    good = _make(tmp_path, "i848", 3)
    squeezed_npz = (squeezed / "cam0.npz").read_bytes()
    _run(monkeypatch, "--intr_dir", base, "--joint_intr_dir", squeezed, good,
         "--from_images", "--board", "9x6_id90", "--min_views", "8")
    camera = _report(base)["cameras"]["0"]
    assert [m["intr_dir"] for m in camera["joint"]["members"]] == [
        str(base.resolve()), str(good.resolve())]
    assert "배율이 다름" in camera["joint_unavailable"][str(squeezed.resolve())]
    assert (squeezed / "cam0.npz").read_bytes() == squeezed_npz


def test_joint_report_keeps_entries_for_cameras_it_did_not_write(tmp_path, monkeypatch):
    base = _make(tmp_path, "i1280", 1)
    joint = _make(tmp_path, "i1920", 2, serial_to_idx={SERIAL: 0, "camera-B": 1})
    previous = {"calibrated_at": "earlier", "cameras": {
        "1": {"serial": "camera-B", "status": "written", "rms_px": 0.5},
        "0": {"serial": "camera-A", "status": "written", "rms_px": 9.9},
    }}
    (joint / "charuco_intrinsics_report.json").write_text(json.dumps(previous))
    _run(monkeypatch, "--intr_dir", base, "--joint_intr_dir", joint, "--from_images",
         "--use_factory_guess", "--board", "9x6_id90", "--min_views", "8")
    cameras = _report(joint)["cameras"]
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


@pytest.mark.parametrize("dirs", [["SELF"], ["OTHER", "OTHER"]])
def test_joint_rejects_duplicate_folders(tmp_path, monkeypatch, dirs):
    base = _make(tmp_path, "i1280", 1, frames=4)
    other = _make(tmp_path, "i1920", 2, frames=4)
    paths = [base if d == "SELF" else other for d in dirs]
    with pytest.raises(SystemExit, match="같습니다"):
        _run(monkeypatch, "--intr_dir", base, "--joint_intr_dir", *paths, "--from_images",
             "--board", "9x6_id90")


# --- --views_per_folder: 폴더마다 같은 장수, 최대 다양성 -----------------------------

def test_farthest_point_order_is_deterministic_and_spread():
    rng = np.random.default_rng(0)
    cluster = rng.normal([0.5, 0.5, 0.3, 0.0, 0.0], [0.01, 0.01, 0.002, 1.0, 1.0], (20, 5))
    extremes = np.array([[0.1, 0.5, 0.3, 0, 0], [0.9, 0.5, 0.3, 0, 0],
                         [0.5, 0.5, 0.3, 60, 0], [0.5, 0.5, 0.3, 0, -60]])
    desc = np.vstack([cluster, extremes])
    picked = app._farthest_point_order(desc, 5)
    assert picked == app._farthest_point_order(desc, 5)
    assert set(range(20, 24)) <= set(picked), "양끝 사진이 먼저 뽑혀야 한다"
    assert app._farthest_point_order(desc[:3], 10) == [0, 1, 2]


def test_views_per_folder_uses_exactly_n_from_every_folder(tmp_path, monkeypatch):
    base = _make(tmp_path, "i1920", 1)
    others = [_make(tmp_path, "i1280", 2), _make(tmp_path, "i848", 3, frames=8)]
    _run(monkeypatch, "--intr_dir", base, "--joint_intr_dir", *others, "--from_images",
         "--use_factory_guess", "--zero_tangent", "--board", "9x6_id90",
         "--views_per_folder", "6", "--min_views", "8")

    with np.load(base / "cam0.npz") as b:
        K_base = b["color_K"]
        assert int(b["charuco_balance_views_per_folder"]) == 6
    for folder in [base, *others]:
        report = _report(folder)
        assert report["calib_flags"]["views_per_folder"] == 6
        camera = report["cameras"]["0"]
        assert camera["status"] == "written"
        assert camera["num_views_used"] == camera["num_views_total"] == 6
        assert camera["balance"]["views_per_folder"] == 6
        statuses = [o["status"] for o in camera["image_observations"]]
        assert statuses.count("detected") == 6
        assert statuses.count("not_selected_for_balance") == len(statuses) - 6
        assert all(m["num_views_used"] == 6 for m in camera["joint"]["members"])
        with np.load(folder / "cam0.npz") as z:
            np.testing.assert_allclose(
                z["color_K"], app._map_K(K_base, app._stream_map(K1920, z["factory_color_K"])),
                atol=1e-9)


def test_balance_drops_gross_failures_before_choosing():
    size, K = STREAMS["i1280"]
    board = CharucoTarget(resolve_charuco_config("9x6_id90")[0])
    accepted = []
    for frame in _render(K, size, _poses(9, 4)):
        corners, ids = board.detect(frame)[:2]
        accepted.append((corners, ids))
    rng = np.random.default_rng(1)
    broken = accepted[3][0] + rng.normal(0, 15, accepted[3][0].shape).astype(np.float32)
    accepted[3] = (broken, accepted[3][1])
    keep, dropped = app._balance_views(board.board, accepted, [0] * 9, 6, size, 0,
                                       K.copy(), K, ["i1280"])
    assert dropped[3] == "rejected_gross_error"
    assert 3 not in keep and len(keep) == 6
    assert sorted(dropped) == sorted(set(range(9)) - set(keep))


def test_views_per_folder_skips_camera_when_a_folder_is_short(tmp_path, monkeypatch):
    base = _make(tmp_path, "i1280", 1)
    short = _make(tmp_path, "i1920", 2, frames=5)
    before = {p: p.read_bytes() for p in (base / "cam0.npz", short / "cam0.npz")}
    _run(monkeypatch, "--intr_dir", base, "--joint_intr_dir", short, "--from_images",
         "--board", "9x6_id90", "--views_per_folder", "6", "--min_views", "8")
    camera = _report(base)["cameras"]["0"]
    assert camera["status"] == "too_few_views_for_balance"
    assert "i1920: 5장" in camera["error"]
    assert before == {p: p.read_bytes() for p in before}
    assert "장수 맞춤 불가" in _report(short)["cameras"]["0"]["joint_unavailable"]


@pytest.mark.parametrize("extra, message", [
    (["--views_per_folder", "6"], "requires --from_images"),
    (["--from_images", "--views_per_folder", "3"], "at least 4"),
])
def test_views_per_folder_cli_validation(tmp_path, monkeypatch, capsys, extra, message):
    monkeypatch.setattr(sys, "argv", ["02", "--intr_dir", str(tmp_path), *extra])
    with pytest.raises(SystemExit):
        app.main()
    assert message in capsys.readouterr().err
