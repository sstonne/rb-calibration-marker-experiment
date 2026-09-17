"""
멀티카메라 캘리브레이션용 데이터셋을 캡처한다.

composite_rig_45_v2 파이프라인:
  1. 사전 검증된 pose plan을 PC에서 robot server로 보낸다.
  2. P1 15, P2 20, P3 10 planned event를 순서대로 동기 촬영한다.
  3. 모든 카메라 RGB-D와 robot/release state를 attempt 단위로 저장한다.
  4. Marker quality는 진단으로만 기록하고 transport/sync 실패만 같은 ID로 재시도한다.
"""

"""
<< 서버 >> 
python c1.py --auto pc --speed 30
[set 0 z+100]
gotoj 37.96, -9.45, -136.81, 0.25, -33.05, -117.92
p z,-100
gc
[자동화 촬영시] start
[티칭시] rs(set) / rp(pose) -A / rg(grip) -B

<< 최종 45-event 촬영 >>
python3 03_capture.py \
    --data_root zeus_gello_calibration/data \
    --session_label zeus_composite_rig \
    --intrinsics_dir intrinsics \
    --waypoints_file capture_plans/composite_rig_45.json \
    --use_robot --manual_robot \
    --robot_ip 192.168.0.23 --robot_port 12348 \
    --max_capture_span_ms 120 --show

저장 파일:
  - meta.json               : 캡처별 상세 (robot pose, set_index, set_cube_center_6dof, cube/board quality)
  - capture_waypoints.json  : frozen 45-event pose plan
  - capture_protocol_manifest.json : planned event 완료/누락/attempt 수

참고:
  - depth 저장은 기본 ON이다. 끄려면 `--no-save-depth`를 사용한다.
  - 최종 protocol은 flange/release state와 frozen rig geometry를 사용한다.
"""

import os
import sys as _sys_top
import json
import hashlib
import time
import shutil
import argparse
import select as _select
import threading as _threading
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from calibration_pipeline.apriltag_cube import AprilTagCubeTarget, depth_metrics_to_fields, rodrigues_to_Rt
from calibration_pipeline.charuco import CharucoTarget
from calibration_pipeline.config import get_default_cube_config
from calibration_pipeline.board_config import (
    charuco_config_mismatch_keys,
    charuco_config_to_dict,
    charuco_configs_equivalent,
    describe_charuco_config,
    list_charuco_boards,
    load_charuco_config_from_meta,
    resolve_charuco_config,
)
from calibration_pipeline.runtime import resolve_cube_config_for_run
from capture_pipeline.detection import detect_cube_markers_in_frame, marker_roi_quality
from capture_pipeline.gate import evaluate_capture_gate, resolve_camera_storage
from capture_pipeline.session import allocate_next_capture_session
from calibration_pipeline.cube_config import (
    cube_config_mismatch_keys,
    cube_config_to_dict,
    cube_configs_equivalent,
    load_cube_config_from_meta,
)
from capture_pipeline.robot import euler_deg_to_matrix
from capture_pipeline.waypoint_safety import (
    PHASE_P2,
    PHASE_P3,
    PROTOCOL_COMPOSITE_RIG_45,
    validate_safe_joint_config,
    validate_waypoint_semantics,
)
from capture_pipeline.paths import REPO_ROOT, require_data_path_inside

ZEUS_CALIBRATION_ROOT = REPO_ROOT / "zeus_gello_calibration"


def resolve_capture_roots(data_root, root_folder=None):
    """Make the explicit CLI data root authoritative for this capture run."""
    resolved_data = require_data_path_inside(
        data_root, ZEUS_CALIBRATION_ROOT, label="--data_root")
    resolved_root = None
    if root_folder is not None:
        resolved_root = require_data_path_inside(
            root_folder, resolved_data, label="--root_folder")
    return str(resolved_data), None if resolved_root is None else str(resolved_root)


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def canonical_json_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate_rig_geometry(path: str, expected_rig_id: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != "composite_rig_geometry_v1":
        raise ValueError("rig geometry schema_version must be composite_rig_geometry_v1")
    if payload.get("template_only") is not False:
        raise ValueError("rig geometry template_only must be false")
    if payload.get("target_rig_id") != expected_rig_id:
        raise ValueError(
            "rig geometry target_rig_id does not match the waypoint plan"
        )
    if payload.get("translation_unit") != "meter":
        raise ValueError("rig geometry translation_unit must be meter")
    for field in ("T_rig_board", "T_rig_cube"):
        matrix = np.asarray(payload.get(field), dtype=float)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"{field} must be a finite 4x4 matrix")
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
            raise ValueError(f"{field} has an invalid homogeneous bottom row")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
            raise ValueError(f"{field} rotation is not orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
            raise ValueError(f"{field} rotation determinant is not +1")
    return payload


def evaluate_transport_integrity(
    frames: Dict[int, dict],
    expected_camera_ids: List[int],
    max_capture_span_ms: float,
) -> dict:
    """Marker-independent camera transport/sync decision for protocol retries."""
    expected = sorted(int(ci) for ci in expected_camera_ids)
    received = sorted(int(ci) for ci in frames)
    missing_frames = sorted(set(expected) - set(received))
    missing_timestamps = sorted(
        ci for ci in received if frames[ci].get("ts_ms") is None
    )
    timestamps = [
        float(frames[ci]["ts_ms"])
        for ci in received
        if frames[ci].get("ts_ms") is not None
    ]
    span_ms = max(timestamps) - min(timestamps) if len(timestamps) >= 2 else 0.0
    reasons = []
    if missing_frames:
        reasons.append("missing camera frames: {}".format(missing_frames))
    if missing_timestamps:
        reasons.append("missing camera timestamps: {}".format(missing_timestamps))
    if max_capture_span_ms > 0 and span_ms > max_capture_span_ms:
        reasons.append(
            "camera timestamp span {:.1f}ms > {:.1f}ms".format(
                span_ms, max_capture_span_ms
            )
        )
    return {
        "schema_version": "capture_transport_integrity_v1",
        "pass": not reasons,
        "status": "PASS" if not reasons else "FAIL",
        "reason": "transport/sync valid" if not reasons else "; ".join(reasons),
        "reasons": reasons,
        "expected_camera_ids": expected,
        "received_camera_ids": received,
        "missing_frame_camera_ids": missing_frames,
        "missing_timestamp_camera_ids": missing_timestamps,
        "capture_span_ms": float(span_ms),
        "max_capture_span_ms": float(max_capture_span_ms),
    }


def annotate_image(bgr, cube, cam_idx, is_gripper, n_markers, ids, corners,
                    board_mkr_corners=None, board_mkr_ids=None,
                    ch_corners=None, ch_ids=None):
    """마커 오버레이 및 정보 텍스트를 이미지에 그림."""
    out = bgr.copy()

    # Board ArUco markers (DICT_4X4_250) — 모든 카메라에서 표시 (고정캠도 보드 인식)
    n_board = 0
    if board_mkr_corners is not None and board_mkr_ids is not None:
        n_board = len(board_mkr_ids)
        try:
            cv2.aruco.drawDetectedMarkers(out, board_mkr_corners, board_mkr_ids)
        except Exception:
            pass

    # ChArUco interpolated corners — 모든 카메라에서 표시
    n_charuco = 0
    if ch_corners is not None and ch_ids is not None:
        n_charuco = len(ch_ids)
        try:
            cv2.aruco.drawDetectedCornersCharuco(out, ch_corners, ch_ids)
        except Exception:
            pass

    # Draw cube markers
    if ids is not None and len(corners) > 0:
        try:
            draw_ids = ids.reshape(-1, 1) if getattr(ids, "ndim", 1) == 1 else ids
            cv2.aruco.drawDetectedMarkers(out, corners, draw_ids)
        except Exception:
            pass

    role = "GRIPPER" if is_gripper else "FIXED"
    ids_txt = ",".join(str(int(x)) for x in ids) if ids is not None and len(ids) > 0 else "-"
    board_txt = ""
    if n_board > 0 or n_charuco > 0:
        board_txt = f" board={n_board}mkr ch={n_charuco}cor"
    lines = [
        f"cam{cam_idx} [{role}]",
        f"markers={n_markers} ids={ids_txt}{board_txt}",
    ]
    y = 24
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (4, y - 18), (10 + tw, y + 4), (0, 0, 0), -1)
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        y += 22
    return out


def wait_for_start_command_capture(cams, cam_order, gripper_cam_idx,
                                     extra_lines: Optional[List[str]] = None,
                                     frame_builder=None, cube=None,
                                     preview_frac: float = 0.6) -> bool:
    """캘리브레이션 캡처 시작 전 cv2 프리뷰 + 'start' 입력 대기.

    `frame_builder`가 주어지면 각 카메라에서 인식되는 AprilTag 큐브/보드/ChArUco
    마커를 실시간으로 오버레이해서 보여준다 (마커 인식 상태까지 확인 가능).
    주어지지 않으면 단순 4-캠 raw 프리뷰만 띄운다. 터미널에서:
      start  -> 캡처 메인 루프 진입 (창은 닫지 않고 후속 모드에서 같은 창 갱신)
      quit   -> 캡처 시작 안 하고 종료 (창 닫음)
    Returns: True 시작 / False 사용자 취소.
    """
    print("")
    print("=" * 60)
    print(" Live preview — type 'start' (then ENTER) in this terminal to begin")
    print(" or type 'quit' / press q in the preview window to abort")
    print("=" * 60)
    if extra_lines:
        for ln in extra_lines:
            print(" " + ln)

    start_event = _threading.Event()
    quit_event = _threading.Event()

    def _stdin_reader():
        while not (start_event.is_set() or quit_event.is_set()):
            try:
                r, _, _ = _select.select([_sys_top.stdin], [], [], 0.2)
                if not r:
                    continue
                line = _sys_top.stdin.readline()
                if not line:
                    quit_event.set(); return
                token = line.strip().lower()
                if token == "start":
                    start_event.set(); return
                if token in ("quit", "q", "exit"):
                    quit_event.set(); return
                if token:
                    print(f"  type 'start' or 'quit' (got: {token!r})")
            except Exception:
                quit_event.set(); return

    t = _threading.Thread(target=_stdin_reader, daemon=True)
    t.start()

    win = "Capture Preview"
    while not start_event.is_set() and not quit_event.is_set():
        # frame_builder가 있으면 마커 검출 오버레이가 포함된 quad를 만든다.
        if frame_builder is not None:
            preview_frames = {}
            for ci in cam_order:
                cam = cams.get(ci)
                if cam is None:
                    continue
                color, depth, ts_ms = cam.get_latest()
                if color is None:
                    continue
                try:
                    preview_frames[ci] = frame_builder(
                        ci, color, depth, ts_ms,
                        include_marker_poses=False,
                        include_charuco_pose=False,
                        log_pose_status=False,
                    )
                except Exception:
                    continue
            if preview_frames:
                quad = make_quad_image(preview_frames, cam_order, cube, gripper_cam_idx)
            else:
                quad = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(quad, "no frames", (20, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            # make_quad_image 와 같은 이유로 빈 타일은 나중에 채운다.
            tiles = []
            missing = []
            tile_h = tile_w = None
            for ci in cam_order:
                cam = cams.get(ci)
                color = None
                if cam is not None:
                    color, _depth, _ts = cam.get_latest()
                if color is not None:
                    if tile_h is None:
                        tile_h, tile_w = color.shape[:2]
                    disp = color.copy()
                    tag = "GRIP" if (gripper_cam_idx is not None and ci == gripper_cam_idx) else "FIX"
                    col = (0, 200, 255) if tag == "GRIP" else (0, 255, 0)
                    cv2.putText(disp, f"cam{ci} [{tag}]", (10, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
                    tiles.append(disp)
                else:
                    missing.append((len(tiles), ci))
                    tiles.append(None)
            if tile_h is None:
                tile_h, tile_w = 720, 1280
            for idx, ci in missing:
                blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                cv2.putText(blank, f"cam{ci} N/A", (20, tile_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                tiles[idx] = blank
            while len(tiles) < 4:
                tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
            tiles = tiles[:4]
            top = cv2.hconcat([tiles[0], tiles[1]])
            bot = cv2.hconcat([tiles[2], tiles[3]])
            quad = cv2.vconcat([top, bot])

        # footer
        foot_h = 28 + 26 * (1 + (len(extra_lines) if extra_lines else 0))
        foot = np.zeros((foot_h, quad.shape[1], 3), dtype=np.uint8)
        wait_txt = ("[WAITING] live marker overlay — Type 'start' + ENTER in terminal to begin"
                    if frame_builder is not None
                    else "[WAITING] Type 'start' + ENTER in terminal to begin")
        cv2.putText(foot, wait_txt,
                    (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
        if extra_lines:
            y = 48
            for ln in extra_lines:
                cv2.putText(foot, ln, (12, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
                y += 24
        quad = cv2.vconcat([quad, foot])
        cv2.imshow(win, fit_to_screen(quad, preview_frac))

        key = cv2.waitKey(50) & 0xFF
        if key == 27 or key == ord('q'):
            quit_event.set()
            break

    if start_event.is_set():
        print("[start] confirmed, proceeding...")
        # 후속 모드 진입 전까지 창이 "응답 없음"으로 빠지지 않도록 한 번 펌프.
        try:
            cv2.waitKey(1)
        except Exception:
            pass
        return True
    try:
        cv2.destroyWindow(win)
    except Exception:
        pass
    print("[abort] user cancelled before start.")
    return False


_SCREEN_SIZE: Optional[Tuple[int, int]] = None


def get_screen_size() -> Tuple[int, int]:
    """(width, height) of the primary display, cached.

    720p x 4 타일이면 원본이 2560x1440 이라 그대로 띄우면 화면을 덮는다. 화면 크기를
    알아야 "모니터의 몇 %" 로 맞출 수 있는데, OpenCV 는 이를 알려주지 않는다.
    tkinter(표준 라이브러리) -> xrandr 순으로 시도하고, 둘 다 실패하면 1920x1080 으로
    가정한다. 실패해도 프리뷰는 떠야 하므로 예외를 올리지 않는다.
    """
    global _SCREEN_SIZE
    if _SCREEN_SIZE is not None:
        return _SCREEN_SIZE
    size = None
    try:
        import tkinter
        _root = tkinter.Tk()
        _root.withdraw()
        size = (_root.winfo_screenwidth(), _root.winfo_screenheight())
        _root.destroy()
    except Exception:
        try:
            import subprocess
            out = subprocess.check_output(["xrandr"], stderr=subprocess.DEVNULL).decode()
            for line in out.splitlines():
                if " connected" in line and "x" in line:
                    for tok in line.split():
                        if "x" in tok and tok.split("x")[0].isdigit():
                            w, h = tok.split("+")[0].split("x")
                            size = (int(w), int(h))
                            break
                if size:
                    break
        except Exception:
            size = None
    _SCREEN_SIZE = size or (1920, 1080)
    return _SCREEN_SIZE


def fit_to_screen(img, frac: float):
    """Scale ``img`` so it occupies at most ``frac`` of the screen in both axes.

    Aspect ratio is preserved and the image is never upscaled — a small panel
    should stay small rather than being blown up to fill the budget.
    """
    if img is None or frac <= 0:
        return img
    sw, sh = get_screen_size()
    h, w = img.shape[:2]
    scale = min(frac * sw / float(w), frac * sh / float(h), 1.0)
    if scale >= 0.999:
        return img
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA)


def make_quad_image(frames_dict, cam_order, cube, gripper_cam_idx):
    """4개 카메라로부터 마커 오버레이가 포함된 2x2 분할 이미지를 생성."""
    # 빈 타일은 실제 타일 크기를 안 뒤에 채운다. 먼저 채우면 cam_order 앞쪽
    # 카메라가 프레임을 못 준 순간 그 크기가 고정돼, 뒤따르는 실제 타일과
    # hconcat 에서 크기가 어긋난다.
    tiles: List[Optional[np.ndarray]] = []
    missing: List[Tuple[int, int]] = []
    tile_h, tile_w = None, None

    for ci in cam_order:
        fr = frames_dict.get(ci)
        if fr is not None and fr.get("color") is not None:
            img = fr["color"]
            if tile_h is None:
                tile_h, tile_w = img.shape[:2]
            annotated = annotate_image(
                img, cube, ci,
                is_gripper=(ci == gripper_cam_idx),
                n_markers=fr.get("n_markers", 0),
                ids=fr.get("ids_np"),
                corners=fr.get("corners", []),
                board_mkr_corners=fr.get("board_mkr_corners"),
                board_mkr_ids=fr.get("board_mkr_ids"),
                ch_corners=fr.get("ch_corners"),
                ch_ids=fr.get("ch_ids"),
            )
            tiles.append(annotated)
        else:
            missing.append((len(tiles), ci))
            tiles.append(None)

    # 어느 카메라도 프레임을 못 준 경우에만 기본 크기를 쓴다 (표준 촬영 해상도).
    if tile_h is None:
        tile_h, tile_w = 720, 1280
    for idx, ci in missing:
        blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        cv2.putText(blank, f"cam{ci} N/A", (20, tile_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        tiles[idx] = blank

    while len(tiles) < 4:
        tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
    tiles = tiles[:4]

    top = cv2.hconcat([tiles[0], tiles[1]])
    bottom = cv2.hconcat([tiles[2], tiles[3]])
    return cv2.vconcat([top, bottom])


def make_capture_gate_config(args) -> dict:
    common_span = float(args.max_capture_span_ms)
    return {
        "schema_version": "capture_gate_profiles_v1",
        "profiles": {
            "A_placement": {
                "expected_cube_gripped": False,
                "min_cams_with_cube": int(args.min_cams_with_cube),
                "min_fixed_cams_with_cube": int(args.min_fixed_cams_with_cube),
                "min_fixed_multimarker_cams": int(args.a_min_fixed_multimarker_cams),
                "fixed_multimarker_min_markers": int(args.fixed_multimarker_min_markers),
                "max_cube_pnp_reproj_mean_px": float(args.max_cube_pnp_reproj_mean_px),
                "min_depth_samples": int(args.min_depth_samples),
                "min_cube_pnp_ok_cams": int(args.min_cube_pnp_ok_cams),
                "min_fixed_cube_pnp_ok_cams": int(args.min_fixed_cube_pnp_ok_cams),
                "min_fixed_depth_quality_cams": 0,
                "min_gripper_markers": int(args.gripper_cube_min_markers),
                "min_gripper_charuco_corners": int(args.min_gripper_charuco_corners),
                "require_gripper_cube_pnp": bool(args.require_gripper_cube_pnp),
                "require_gripper_depth_valid": bool(args.require_gripper_depth_valid),
                "max_gripper_depth_plane_mean_mm": float(args.max_gripper_depth_plane_mean_mm),
                "max_fixed_depth_plane_mean_mm": 0.0,
                "max_capture_span_ms": common_span,
                "require_all_frame_timestamps": True,
                "max_roi_clip_frac": float(args.max_roi_clip_frac),
                "min_roi_sharpness": float(args.min_roi_sharpness),
            },
            "B_eyetohand": {
                "expected_cube_gripped": True,
                # The gripper camera is not an observation requirement in B.
                "min_cams_with_cube": 0,
                "min_fixed_cams_with_cube": int(args.b_min_fixed_cams_with_cube),
                "min_fixed_multimarker_cams": int(args.b_min_fixed_multimarker_cams),
                "fixed_multimarker_min_markers": int(args.fixed_multimarker_min_markers),
                "max_cube_pnp_reproj_mean_px": float(args.max_cube_pnp_reproj_mean_px),
                "min_depth_samples": int(args.min_depth_samples),
                "min_cube_pnp_ok_cams": 0,
                "min_fixed_cube_pnp_ok_cams": int(args.b_min_fixed_cube_pnp_ok_cams),
                "min_fixed_depth_quality_cams": int(args.b_min_fixed_depth_quality_cams),
                "min_gripper_markers": 0,
                "min_gripper_charuco_corners": 0,
                "require_gripper_cube_pnp": False,
                "require_gripper_depth_valid": False,
                "max_gripper_depth_plane_mean_mm": 0.0,
                "max_fixed_depth_plane_mean_mm": float(args.b_max_fixed_depth_plane_mean_mm),
                "max_capture_span_ms": common_span,
                "require_all_frame_timestamps": True,
                "max_roi_clip_frac": float(args.max_roi_clip_frac),
                "min_roi_sharpness": float(args.min_roi_sharpness),
            },
        },
    }


def build_capture_gate_lines(gate: dict,
                             gripper_cam_idx: Optional[int],
                             frames_dict: Dict[int, dict]) -> List[str]:
    line1 = (
        "SAVE gate [{}]: {} | visible cams {}/{} | span {:.1f}/{:.1f} ms".format(
            gate.get("capture_block", "A_placement"),
            gate.get("status", "N/A"),
            int(gate.get("cams_with_cube", 0)),
            int(gate.get("min_cams_with_cube", 0)),
            float(gate.get("capture_span_ms", 0.0)),
            float(gate.get("max_capture_span_ms", 0.0)),
        )
    )

    depth_plane = gate.get("gripper_depth_plane_mean_mm")
    depth_plane_txt = "-" if depth_plane is None else "{:.1f}mm".format(float(depth_plane))
    grip_txt = (
        "Gripper: markers={} cube_pnp={} depth={} plane={} charuco={}/{}".format(
            int(gate.get("gripper_markers", 0)),
            "Y" if gate.get("gripper_cube_pnp_ok", False) else "N",
            "Y" if gate.get("gripper_depth_valid", False) else "N",
            depth_plane_txt,
            int(gate.get("gripper_charuco_corners", 0)),
            int(gate.get("min_gripper_charuco_corners", 0)),
        )
    )

    line2 = (
        "{} | fixed visible={}/{}".format(
            grip_txt,
            int(gate.get("fixed_visible_cams", 0)),
            int(gate.get("min_fixed_cams_with_cube", 0)),
        )
    )
    line3 = (
        "PnP/depth: total {}/{} | fixed {}/{} | fixed multi {}/{} | fixed depth-quality {}/{}".format(
            int(gate.get("cube_pnp_ok_cams", 0)),
            int(gate.get("min_cube_pnp_ok_cams", 0)),
            int(gate.get("fixed_cube_pnp_ok_cams", 0)),
            int(gate.get("min_fixed_cube_pnp_ok_cams", 0)),
            int(gate.get("fixed_multimarker_cams", 0)),
            int(gate.get("min_fixed_multimarker_cams", 0)),
            int(gate.get("fixed_depth_quality_cams", 0)),
            int(gate.get("min_fixed_depth_quality_cams", 0)),
        )
    )

    lines = [line1, line2, line3]
    if not gate.get("pass", False):
        lines.append("FAIL reason: {}".format(gate.get("reason", "unknown")))
    return lines


def append_status_footer(image: np.ndarray,
                         lines: List[str],
                         colors: Optional[List[Tuple[int, int, int]]] = None,
                         bg_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    if not lines:
        return image
    footer_h = 28 * len(lines) + 12
    footer = np.zeros((footer_h, image.shape[1], 3), dtype=np.uint8)
    footer[:, :] = np.array(bg_color, dtype=np.uint8)
    if colors is None:
        colors = [(255, 255, 255)] * len(lines)
    for idx, line in enumerate(lines):
        color = colors[min(idx, len(colors) - 1)]
        y = 28 + idx * 28
        cv2.putText(footer, line, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
    return cv2.vconcat([image, footer])


def load_device_map(intr_dir: str):
    map_path = os.path.join(intr_dir, "device_map.json")
    if not os.path.exists(map_path):
        return None, None, None
    with open(map_path, "r") as f:
        m = json.load(f)
    serial_to_idx = m.get("serial_to_idx", {})
    gripper_cam_idx = m.get("gripper_cam_idx", None)
    return serial_to_idx, gripper_cam_idx, map_path


def load_intrinsics(intr_dir: str, cam_idx: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """카메라 내부 파라미터 행렬 K, 왜곡 계수 D, depth scale을 로드."""
    p = os.path.join(intr_dir, f"cam{cam_idx}.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Intrinsics not found: {p}")
    d = np.load(p, allow_pickle=True)
    K = d["color_K"].astype(np.float64)
    D = d["color_D"].astype(np.float64)
    depth_scale = float(d["depth_scale_m_per_unit"]) if "depth_scale_m_per_unit" in d else 0.001
    if not np.isfinite(depth_scale):
        depth_scale = 0.001
    return K, D, float(depth_scale)


def marker_aspect_ratio(img_pts: np.ndarray) -> float:
    pts = np.asarray(img_pts, dtype=np.float64).reshape(4, 2)
    edge_w = np.linalg.norm(pts[1] - pts[0])
    edge_h = np.linalg.norm(pts[3] - pts[0])
    return float(min(edge_w, edge_h) / (max(edge_w, edge_h) + 1e-6))


def estimate_per_marker_poses(
    cube: AprilTagCubeTarget,
    corners_list: list,
    ids: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    depth_u16: Optional[np.ndarray] = None,
    depth_scale: Optional[float] = None,
) -> List[dict]:
    """
    알려진 큐브 형상을 이용하여 개별 마커의 포즈를 추정.
    마커 1개만으로도 카메라-큐브 변환을 추정할 수 있음.

    마커별 결과 리스트 (rvec, tvec, 재투영 오차 포함)를 반환.
    """
    results = []
    if ids is None or len(ids) == 0:
        return results

    for c, mid in zip(corners_list, ids):
        mid = int(mid)
        if not cube.model.has_marker(mid):
            continue

        img_pts = cube.model.reorder_image_corners(mid, c.reshape(4, 2).astype(np.float64))
        aspect = marker_aspect_ratio(img_pts)
        ippe_candidates = cube.single_marker_ippe_candidates(
            mid,
            c.reshape(4, 2).astype(np.float64),
            K,
            D,
            corners_list=corners_list,
            ids=ids,
            depth_u16=depth_u16,
            depth_scale=depth_scale,
        )
        if not ippe_candidates:
            continue

        pose_candidates = []
        best_idx = None
        best_rank = None
        best_rvec, best_tvec = None, None
        best_T_cam_cube = None
        best_err = None
        best_err_px = None
        best_depth_metrics = None

        for cand in ippe_candidates:
            sol_idx = int(cand["solution_index"])
            rvec = cand["rvec"]
            tvec = cand["tvec"]
            err_px = cand["err"]
            err_mean = float(cand["err_mean"])
            T_cam_cube = cand["T_C_O"]
            depth_metrics = cand["depth_metrics"]
            pose_candidates.append({
                "solution_index": int(sol_idx),
                "rvec": rvec.flatten().tolist(),
                "tvec": tvec.flatten().tolist(),
                "reproj_error_mean_px": err_mean,
                "reproj_error_max_px": float(np.max(err_px)),
                "T_cam_cube_4x4": T_cam_cube.tolist(),
                "z_ok": bool(cand["z_ok"]),
                "vis_ok": bool(cand["vis_ok"]),
                "vis_score": float(cand["vis_score"]),
                "visibility_tier": int(cand["visibility_tier"]),
                **depth_metrics_to_fields(depth_metrics),
            })
            rank = cand["rank"]
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_idx = int(sol_idx)
                best_rvec = rvec
                best_tvec = tvec
                best_T_cam_cube = T_cam_cube
                best_err = err_mean
                best_err_px = err_px
                best_depth_metrics = depth_metrics

        if best_idx is None:
            continue

        results.append({
            "marker_id": mid,
            "face": cube.cfg.id_to_face[mid],
            "corners_2d": img_pts.tolist(),
            "aspect_ratio": aspect,
            "rvec": best_rvec.flatten().tolist(),
            "tvec": best_tvec.flatten().tolist(),
            "reproj_error_mean_px": float(best_err),
            "reproj_error_max_px": float(np.max(best_err_px)),
            "T_cam_cube_4x4": best_T_cam_cube.tolist(),
            "selected_solution_index": int(best_idx),
            "pose_candidates": pose_candidates,
            **depth_metrics_to_fields(best_depth_metrics),
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Zeus composite_rig_45_v2 capture: synchronized RGB-D and robot state"
        )
    )
    parser.add_argument(
        "--root_folder",
        default=None,
        help=(
            "Explicit capture folder for a deliberate resume. "
            "It must be inside the explicit --data_root. Omit this option "
            "for automatic sessionNN/calib_train allocation."
        ),
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help=("Parent for automatic session allocation when --root_folder is omitted; "
              f"must be inside {ZEUS_CALIBRATION_ROOT}"),
    )
    parser.add_argument(
        "--session_label",
        default=None,
        help=("Short description folded into the new session folder name, e.g. "
              "\"zeus wrist motion\" -> "
              "<--data_root>/session11_zeus_wrist_motion_<MMDD>. "
              "Required for a newly allocated session; omit only when "
              "--root_folder deliberately resumes an existing session."),
    )
    parser.add_argument(
        "--waypoints_file",
        required=True,
        help=(
            "Validated waypoint JSON to copy into a newly allocated session as "
            "capture_waypoints.json before robot connection"
        ),
    )
    parser.add_argument("--intrinsics_dir", required=True)
    parser.add_argument("--cube_config_json", type=str, default=None,
                        help="Optional cube config JSON override. Leave unset to use the project's canonical cube definition.")
    parser.add_argument("--board", type=str, default=None,
                        help="ChArUco board definition: a name from targets/charuco_boards/ "
                             f"({', '.join(list_charuco_boards()) or 'none'}) or a JSON path. "
                             "Leave unset for config.py's default board. A resumed session "
                             "must use the board frozen in its meta.json.")

    # 스트림 설정
    # 해상도 기본값 1280x720 — 프로젝트 표준 촬영 해상도(color/depth 동일).
    # 640x480 에서는 18mm ChArUco 마커가 한 변 중앙값 21px 로 검출 하한(~18px)에 걸쳐
    # 고정 카메라 3대의 코너 수율이 60개 중 4~8개까지 떨어졌다. 720p 는 4대 동시
    # color+depth 로 드롭 0, 카메라 간 span p50 46ms 로 통과 확인됨
    # (tools/smoke_test_resolution.py). 인트린식도 반드시 이 해상도로 캘리브되어야 하며,
    # 아래 정합성 검사가 --intrinsics_dir 의 color_w/color_h 와 불일치를 잡아 중단시킨다.
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--allow_intrinsics_res_mismatch", action="store_true",
                        help="캡처 해상도가 인트린식(color_w/h)과 달라도 강행. "
                             "PnP/ChArUco pose 가 부정확해지므로 디버깅용에만 사용.")

    # 측광 고정 — 캘리브레이션 프레임은 세션 내내 동일 노출이어야 한다. auto-exposure 는
    # 로봇이 자세를 바꿀 때마다 재수렴해 코너 서브픽셀 정확도를 프레임마다 바꾸고,
    # 어두운 뷰에서 노출을 늘려 모션 블러를 만든다. 기본은 warmup 수렴값을 읽어 잠근다.
    parser.add_argument("--no-lock-exposure", dest="no_lock_exposure", action="store_true",
                        help="측광 고정을 끄고 auto-exposure 로 촬영(비권장, 진단용).")
    parser.add_argument("--color_exposure_us", type=float, default=None,
                        help="색상 노출을 이 값(us)으로 명시 고정. 생략 시 warmup 수렴값 사용.")
    parser.add_argument("--color_gain", type=float, default=None,
                        help="색상 gain 명시 고정. 생략 시 warmup 수렴값 사용.")
    parser.add_argument("--color_white_balance", type=float, default=None,
                        help="화이트밸런스(K) 명시 고정. 생략 시 warmup 수렴값 사용.")

    # 마커 ROI 품질 게이트. clip 은 스케일 무관이라 기본 활성, sharpness 는 카메라마다
    # 절대값이 달라 0(비활성)으로 두고 pilot 첫 set 의 측정값으로 카메라별 확정한다.
    parser.add_argument("--max_roi_clip_frac", type=float, default=0.05,
                        help="마커 ROI 에서 흑/백 포화 픽셀 비율 상한. 0 이면 비활성.")
    parser.add_argument("--min_roi_sharpness", type=float, default=0.0,
                        help="마커 ROI Laplacian 분산 하한. 0 이면 비활성(기본). "
                             "pilot 측정 후 카메라별로 설정.")

    # 검출 설정
    parser.add_argument("--min_markers", type=int, default=1,
                        help="Min markers per camera to count as 'cube visible'")
    parser.add_argument("--min_cams_with_cube", type=int, default=2,
                        help="Min cameras that must see cube to accept capture")
    parser.add_argument("--min_fixed_cams_with_cube", type=int, default=1,
                        help="Min fixed cameras that must see cube markers to accept capture")
    parser.add_argument("--gripper_cube_min_markers", type=int, default=1,
                        help="Min cube markers required for gripper-camera cube pose")
    parser.add_argument("--gripper_cube_min_aspect", type=float, default=0.35,
                        help="Reject gripper-camera cube markers below this aspect ratio")
    parser.add_argument("--board_mask_pad_px", type=float, default=6.0,
                        help="Extra padding in pixels when masking ChArUco board markers in gripper images")
    parser.add_argument("--min_cube_pnp_ok_cams", type=int, default=2,
                        help="Min cameras with successful cube pose solve to accept capture")
    parser.add_argument("--min_fixed_cube_pnp_ok_cams", type=int, default=1,
                        help="Min fixed cameras with successful cube pose solve to accept capture")
    parser.add_argument("--fixed_multimarker_min_markers", type=int, default=2,
                        help="Number of cube markers that makes one fixed-camera observation multi-marker")
    parser.add_argument("--a_min_fixed_multimarker_cams", type=int, default=1,
                        help="A_placement: fixed cameras that must see at least --fixed_multimarker_min_markers")
    parser.add_argument("--b_min_fixed_cams_with_cube", type=int, default=2,
                        help="B_eyetohand: minimum fixed cameras that must see the gripped cube")
    parser.add_argument("--b_min_fixed_cube_pnp_ok_cams", type=int, default=2,
                        help="B_eyetohand: minimum fixed cameras with a successful cube pose")
    parser.add_argument("--b_min_fixed_multimarker_cams", type=int, default=2,
                        help="B_eyetohand: fixed cameras that must have a multi-marker observation")
    parser.add_argument("--b_min_fixed_depth_quality_cams", type=int, default=1,
                        help="B_eyetohand: fixed cameras with valid depth support (0 disables)")
    parser.add_argument("--b_max_fixed_depth_plane_mean_mm", type=float, default=20.0,
                        help="B_eyetohand: max depth-plane mean error for a depth-quality fixed camera")
    parser.add_argument("--max_cube_pnp_reproj_mean_px", type=float, default=2.0,
                        help="Count a cube PnP as gate-quality only at or below this mean reprojection error")
    parser.add_argument("--min_depth_samples", type=int, default=20,
                        help="Minimum valid depth samples for a depth-supported cube pose")
    parser.add_argument("--min_gripper_charuco_corners", type=int, default=8,
                        help="Min ChArUco corners required in gripper camera to accept capture")
    parser.add_argument("--a_fixed_cam_views_per_set", type=int, default=1,
                        help="A_placement: how many A captures per set store the fixed-camera "
                             "images and records. The cube is static within a set, so the "
                             "remaining views are the same scene again (default: 1). "
                             "0 stores the fixed cameras on every A view.")
    parser.add_argument("--b_save_gripper_cam", action="store_true",
                        help="B_eyetohand: also store the gripper camera. Off by default — the "
                             "wrist camera cannot see the cube it is holding and the B gate "
                             "requires nothing from it.")
    parser.add_argument(
        "--require_gripper_cube_pnp",
        dest="require_gripper_cube_pnp",
        action="store_true",
        default=True,
        help="Require successful cube pose solve in gripper camera (default: on)",
    )
    parser.add_argument(
        "--allow_gripper_cube_pnp_fail",
        dest="require_gripper_cube_pnp",
        action="store_false",
        help="Do not require successful cube pose solve in gripper camera",
    )
    parser.add_argument(
        "--require_gripper_depth_valid",
        dest="require_gripper_depth_valid",
        action="store_true",
        default=True,
        help="Require depth-supported gripper cube pose when depth capture is enabled (default: on)",
    )
    parser.add_argument(
        "--allow_gripper_depth_invalid",
        dest="require_gripper_depth_valid",
        action="store_false",
        help="Allow gripper cube pose even when depth support is invalid",
    )
    parser.add_argument("--max_gripper_depth_plane_mean_mm", type=float, default=40.0,
                        help="Reject a capture when gripper cube depth plane error exceeds this mm (<=0 disables). "
                             "Conservative 40mm margin; with correct per-marker plane geometry the measured "
                             "plane residual is ~5mm (fixed-cam single-marker), so 15-20mm is also defensible if "
                             "you want the gate to actually catch bad depth.")
    parser.add_argument("--max_capture_span_ms", type=float, default=120.0,
                        help="Skip a capture when camera timestamps span more than this many ms (<=0 disables)")
    parser.add_argument("--allow_force_save", action="store_true",
                        help="Diagnostic only: allow robot force_save to retain a gate-failed frame. "
                             "Never use for Table 1 data.")

    # 뎁스 저장
    parser.add_argument(
        "--save_depth",
        dest="save_depth",
        action="store_true",
        default=True,
        help="Save aligned depth frames for every accepted capture (default: on)",
    )
    parser.add_argument(
        "--no-save-depth",
        dest="save_depth",
        action="store_false",
        help="Disable aligned depth capture and depth PNG saving",
    )

    # 화면 표시
    parser.add_argument("--show", action="store_true")

    # 로봇 모드
    parser.add_argument("--use_robot", action="store_true")
    parser.add_argument("--robot_ip", type=str, default="192.168.0.23")
    parser.add_argument("--robot_port", type=int, default=12348)
    parser.add_argument("--manual_robot", action="store_true",
                        help="Required robot-server mode: server/c1.py sends final protocol commands")
    parser.add_argument("--preview_frac", type=float, default=0.6,
                        help="프리뷰 창이 차지할 화면 비율(0~1). 종횡비는 유지하고 "
                             "원본보다 키우지는 않는다. 기본 0.6 = 모니터의 60%%.")
    parser.add_argument("--frame_sync_timeout_s", type=float, default=3.0,
                        help="촬영 직전 네 카메라의 최신 프레임이 max_capture_span_ms 안으로 "
                             "모일 때까지 기다리는 최대 시간. 초과하면 어느 카메라가 "
                             "뒤처졌는지 출력하고 그대로 진행한다.")
    parser.add_argument("--settle_time", type=float, default=1.5,
                        help="Wait time (s) after robot signals capture before taking images")
    # start gate — 기본은 대기 없이 즉시 시작. --start_gate 를 줘야 프리뷰 + 'start' 대기.
    parser.add_argument("--start_gate", action="store_true",
                        help="시작 전 cv2 프리뷰 + 'start' 입력 대기 (기본: 대기 없이 즉시 시작)")
    # (구) --no_start_gate: 이제 기본 동작이라 무시됨. 기존 명령 호환용으로만 수용.
    parser.add_argument("--no_start_gate", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()
    try:
        args.data_root, args.root_folder = resolve_capture_roots(
            args.data_root, args.root_folder)
    except ValueError as exc:
        parser.error(str(exc))

    waypoint_source = None
    waypoint_payload = None
    if args.waypoints_file:
        waypoint_source = os.path.abspath(os.path.expanduser(args.waypoints_file))
        try:
            with open(waypoint_source, "r", encoding="utf-8") as waypoint_handle:
                waypoint_payload = json.load(waypoint_handle)
            validate_safe_joint_config(waypoint_payload)
            validate_waypoint_semantics(waypoint_payload)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            parser.error(f"invalid --waypoints_file: {exc}")

    if waypoint_payload is None:
        parser.error(
            "03_capture.py requires --waypoints_file with a validated "
            f"{PROTOCOL_COMPOSITE_RIG_45} plan"
        )
    protocol = waypoint_payload.get("capture_protocol")
    if protocol != PROTOCOL_COMPOSITE_RIG_45:
        parser.error(
            "03_capture.py accepts only capture_protocol="
            f"{PROTOCOL_COMPOSITE_RIG_45!r}; received {protocol!r}"
        )
    if not (args.use_robot and args.manual_robot):
        parser.error(
            "03_capture.py requires --use_robot --manual_robot for the final protocol"
        )
    if args.root_folder is None and not str(args.session_label or "").strip():
        parser.error(
            "new capture requires --session_label so the generated "
            "<--data_root>/sessionNN_<label>_<MMDD> folder "
            "identifies the dataset"
        )
    if args.root_folder is not None and args.session_label is not None:
        parser.error(
            "--session_label cannot be combined with --root_folder; "
            "the existing session folder already fixes the dataset name"
        )
    if float(args.max_capture_span_ms) <= 0:
        parser.error("03_capture.py requires a positive --max_capture_span_ms")

    try:
        from capture_pipeline.camera import RealSenseCamera
    except ModuleNotFoundError as error:
        if error.name == "pyrealsense2":
            raise SystemExit(
                "[ERROR] pyrealsense2가 없습니다. RealSense Python 환경에서 "
                "03번을 실행하세요.") from error
        raise
    intr_dir = args.intrinsics_dir
    print(f"[INFO] Depth capture/save: {'ON' if args.save_depth else 'OFF'}")

    # ─── 디바이스 맵 로드 ───
    serial_to_idx, gripper_cam_idx, _ = load_device_map(intr_dir)
    devs = RealSenseCamera.list_devices()
    if len(devs) == 0:
        raise RuntimeError("No RealSense devices found.")

    if serial_to_idx is None:
        print("[WARN] No device_map.json. Run 01_export_intrinsics.py first.")
        serials = sorted(devs.keys())
        idx_serial_pairs = [(i, s) for i, s in enumerate(serials)]
        gripper_cam_idx = None
    else:
        idx_serial_pairs = []
        for serial in devs.keys():
            if serial in serial_to_idx:
                idx_serial_pairs.append((int(serial_to_idx[serial]), serial))
        idx_serial_pairs.sort(key=lambda x: x[0])

    if len(idx_serial_pairs) == 0:
        raise RuntimeError("No usable cameras found.")

    n_fixed = 0
    n_gripper = 0
    print("[INFO] Cameras:")
    for idx, s in idx_serial_pairs:
        if idx == gripper_cam_idx:
            tag = "GRIPPER"
            n_gripper += 1
        else:
            tag = "FIXED"
            n_fixed += 1
        print(f"  cam{idx}: {s} ({tag})")

    if gripper_cam_idx is None:
        print("[WARN] No gripper camera configured in device_map.json.")
        print("[WARN] Gripper camera views will not be available.")
    else:
        print(f"[INFO] Gripper camera: cam{gripper_cam_idx}")

    print(f"[INFO] Fixed cameras: {n_fixed}, Gripper cameras: {n_gripper}")

    # ─── PnP용 내부 파라미터 로드 ───
    cam_intrinsics: Dict[int, Tuple[np.ndarray, np.ndarray, float]] = {}
    for ci, _ in idx_serial_pairs:
        try:
            K, D, depth_scale = load_intrinsics(intr_dir, ci)
            cam_intrinsics[ci] = (K, D, depth_scale)
            print(f"[INFO] Loaded intrinsics for cam{ci}")
        except FileNotFoundError:
            print(f"[WARN] No intrinsics for cam{ci}. Per-marker PnP will be skipped.")

    # ─── 인트린식 해상도 정합성 검사 ───
    # 인트린식(K,D)은 캘리브된 해상도에서만 유효하다. 캡처 해상도가 다르면 cx,cy,fx,fy
    # 가 어긋나 PnP/ChArUco pose 가 조용히 부정확해진다(종횡비까지 다르면 화각도 다름).
    # color_w/color_h 를 읽어 (args.width,args.height)와 비교, 불일치 시 중단.
    res_mismatch = []
    intrinsics_sha256_by_camera = {}
    for ci, _ in idx_serial_pairs:
        p = os.path.join(intr_dir, f"cam{ci}.npz")
        if not os.path.exists(p):
            continue
        intrinsics_sha256_by_camera[str(ci)] = file_sha256(p)
        d = np.load(p, allow_pickle=True)
        if "color_w" in d and "color_h" in d:
            iw, ih = int(d["color_w"]), int(d["color_h"])
            if (iw, ih) != (int(args.width), int(args.height)):
                res_mismatch.append((ci, iw, ih))
    if res_mismatch:
        print("[ERROR] 캡처 해상도 {}x{} 가 인트린식 캘리브 해상도와 다릅니다:"
              .format(args.width, args.height))
        for ci, iw, ih in res_mismatch:
            print(f"          cam{ci}: intrinsics {iw}x{ih}")
        print("        -> {}x{} 해상도로 인트린식을 다시 캘리브하세요:".format(args.width, args.height))
        print("           1) python3 01_export_intrinsics.py --out_dir {} "
              "--color_w {} --color_h {} --fps {}".format(intr_dir, args.width, args.height, args.fps))
        print("           2) python3 02_calibrate_intrinsics.py --intr_dir {}  "
              "(npz 의 {}x{} 를 읽어 리파인)".format(intr_dir, args.width, args.height))
        if not args.allow_intrinsics_res_mismatch:
            raise RuntimeError(
                "인트린식 해상도 불일치로 중단. 재캘리브하거나 "
                "--allow_intrinsics_res_mismatch 로 강행(부정확).")
        print("[WARN] --allow_intrinsics_res_mismatch: 불일치 상태로 강행합니다(pose 부정확).")

    # Allocate only after device, intrinsic-resolution and waypoint validation.
    # This avoids consuming a session number for an invalid command/config.
    allocated_session = None
    if args.root_folder is None:
        allocated_session = allocate_next_capture_session(
            args.data_root, label=args.session_label)
        root = allocated_session.capture_root
        print(f"[SESSION] Allocated {allocated_session.session_id}: {allocated_session.session_root}")
        print(f"[SESSION] Calibration capture root: {root}")
        print(f"[SESSION] Manifest: {allocated_session.manifest_path}")
    else:
        root = ensure_dir(args.root_folder)
        print(f"[SESSION] Explicit capture root: {os.path.abspath(root)}")
        print("[SESSION] Automatic numbering bypassed because --root_folder was supplied.")
    # The network teaching/waypoint paths below use args.root_folder directly.
    args.root_folder = root
    # 캡처 프레임은 <session>/calib_train 에, 티칭 풀과 최종 웨이포인트는 그 위
    # <session>/ 에 둔다. 티칭은 여러 번 이어붙이고 웨이포인트는 그것들을 조합한
    # 산출물이라, 프레임과 섞이면 어느 티칭이 어느 촬영을 만들었는지가 흐려진다.
    session_root = (os.path.dirname(os.path.abspath(root))
                    if os.path.basename(os.path.normpath(root)) == "calib_train"
                    else os.path.abspath(root))
    teach_dir = ensure_dir(os.path.join(session_root, "teaching"))
    waypoints_path = os.path.join(session_root, "capture_waypoints.json")
    print(f"[SESSION] frames    -> {root}")
    print(f"[SESSION] teaching  -> {teach_dir}")
    print(f"[SESSION] waypoints -> {waypoints_path}")
    if waypoint_source is not None and waypoint_payload is not None:
        waypoint_destination = waypoints_path
        if os.path.realpath(waypoint_source) != os.path.realpath(waypoint_destination):
            shutil.copyfile(waypoint_source, waypoint_destination)
        print(f"[SESSION] Validated waypoints: {waypoint_destination}")
    elif os.path.exists(waypoints_path):
        with open(waypoints_path, "r", encoding="utf-8") as waypoint_handle:
            waypoint_payload = json.load(waypoint_handle)
        validate_safe_joint_config(waypoint_payload)
        validate_waypoint_semantics(waypoint_payload)
        waypoint_source = os.path.abspath(waypoints_path)
        print(f"[SESSION] Loaded existing waypoints: {waypoints_path}")

    active_capture_protocol = (
        None if waypoint_payload is None else waypoint_payload.get("capture_protocol")
    )
    final_protocol_mode = active_capture_protocol == PROTOCOL_COMPOSITE_RIG_45
    waypoint_plan_sha256 = (
        None if waypoint_payload is None else canonical_json_sha256(waypoint_payload)
    )
    rig_geometry_path = None
    rig_geometry_sha256 = None
    rig_geometry_payload = None
    if final_protocol_mode:
        if not (args.use_robot and args.manual_robot):
            raise RuntimeError(
                "composite_rig_45_v2 requires --use_robot --manual_robot"
            )
        if float(args.max_capture_span_ms) <= 0:
            raise RuntimeError(
                "composite_rig_45_v2 requires a positive --max_capture_span_ms"
            )
        rig_geometry_path = os.path.expanduser(waypoint_payload["rig_geometry_file"])
        if not os.path.isabs(rig_geometry_path):
            source_dir = os.path.dirname(waypoint_source or waypoints_path)
            rig_geometry_path = os.path.join(source_dir, rig_geometry_path)
        rig_geometry_path = os.path.abspath(rig_geometry_path)
        if not os.path.isfile(rig_geometry_path):
            raise RuntimeError(f"rig geometry file not found: {rig_geometry_path}")
        rig_geometry_payload = load_and_validate_rig_geometry(
            rig_geometry_path,
            expected_rig_id=str(waypoint_payload["target_rig_id"]),
        )
        rig_geometry_sha256 = file_sha256(rig_geometry_path)
        declared_rig_hash = str(waypoint_payload["rig_geometry_sha256"]).lower()
        if rig_geometry_sha256.lower() != declared_rig_hash:
            raise RuntimeError(
                "rig geometry SHA-256 mismatch: "
                f"declared={declared_rig_hash} actual={rig_geometry_sha256}"
            )
        frozen_geometry_path = os.path.join(
            session_root, "composite_rig_geometry.json"
        )
        if os.path.exists(frozen_geometry_path):
            frozen_hash = file_sha256(frozen_geometry_path)
            if frozen_hash.lower() != rig_geometry_sha256.lower():
                raise RuntimeError(
                    "session already contains a different composite_rig_geometry.json"
                )
        elif os.path.realpath(rig_geometry_path) != os.path.realpath(frozen_geometry_path):
            shutil.copyfile(rig_geometry_path, frozen_geometry_path)
        rig_geometry_path = frozen_geometry_path
        print(f"[PROTOCOL] {active_capture_protocol}: validated 45-event pose plan")
        print(f"[PROTOCOL] pose plan sha256: {waypoint_plan_sha256}")
        print(f"[PROTOCOL] rig geometry: {rig_geometry_path}")

    # ─── 카메라 시작 ───
    # 이전 실행이 비정상 종료(세그폴트 등)된 경우 디바이스가 비정상 상태로
    # 남을 수 있어 D435가 첫 pipeline.start()에서 "Frame didn't arrive"로
    # 타임아웃하는 일이 잦다. 모든 디바이스를 한 번 hardware_reset해서 깨끗한
    # 상태에서 시작한다. 그 뒤 USB 협상 안정화를 위해 카메라 간 0.8초 간격으로 순차 시작.
    RealSenseCamera.reset_all_devices()
    cams: Dict[int, RealSenseCamera] = {}
    for i, (ci, serial) in enumerate(idx_serial_pairs):
        if i > 0:
            time.sleep(0.8)
        cam = RealSenseCamera(
            serial=serial,
            width=args.width,
            height=args.height,
            fps=args.fps,
            use_color=True,
            use_depth=args.save_depth,
            align_depth_to_color=True,
            warmup_frames=10,
            lock_color_exposure=not args.no_lock_exposure,
            color_exposure_us=args.color_exposure_us,
            color_gain=args.color_gain,
            color_white_balance=args.color_white_balance,
        )
        cam.start()
        cams[ci] = cam
        ensure_dir(os.path.join(root, f"cam{ci}"))

    cfg, cube_cfg_source = resolve_cube_config_for_run(
        root_folder=root,
        cube_config_json=args.cube_config_json,
        default_cfg=get_default_cube_config(),
    )
    cube = AprilTagCubeTarget(cfg)
    _cube_ids = set(cfg.marker_ids)  # {0,1,2,3,4} — filter out board markers
    print(f"[INFO] Cube config source: {cube_cfg_source}")
    print(f"[INFO] Cube id_to_face: {cfg.id_to_face}")

    # ChArUco board target — 그리퍼캠 + 고정캠 모두 검출 (보드-전용 비교실험)
    charuco_cfg, charuco_cfg_source = resolve_charuco_config(args.board)
    charuco = CharucoTarget(charuco_cfg)
    print(f"[INFO] ChArUco board source: {charuco_cfg_source}")
    print(f"[INFO] ChArUco board: {describe_charuco_config(charuco_cfg)}")

    if not args.save_depth and args.require_gripper_depth_valid:
        print("[WARN] Depth capture is disabled; gripper depth-valid gate will be ignored.")
        args.require_gripper_depth_valid = False
    if not args.save_depth and args.b_min_fixed_depth_quality_cams > 0:
        print("[WARN] Depth capture is disabled; B fixed depth-quality gate will be ignored.")
        args.b_min_fixed_depth_quality_cams = 0
    capture_gate_cfg = make_capture_gate_config(args)
    print("[INFO] Block-aware capture gates:")
    for block_name in ("A_placement", "B_eyetohand"):
        p = capture_gate_cfg["profiles"][block_name]
        print("  {}: fixed visible >= {} | fixed multi-marker >= {} | fixed PnP >= {} | fixed depth-quality >= {}".format(
            block_name,
            p["min_fixed_cams_with_cube"],
            p["min_fixed_multimarker_cams"],
            p["min_fixed_cube_pnp_ok_cams"],
            p["min_fixed_depth_quality_cams"],
        ))
        print("    gripper PnP required={} | gripper markers >= {} | charuco >= {} | gripper depth required={} | span <= {:.1f}ms".format(
            "yes" if p["require_gripper_cube_pnp"] else "no",
            p["min_gripper_markers"],
            p["min_gripper_charuco_corners"],
            "yes" if p["require_gripper_depth_valid"] else "no",
            p["max_capture_span_ms"],
        ))
        print("    PnP reproj mean <= {:.2f}px | depth samples >= {}".format(
            p["max_cube_pnp_reproj_mean_px"], p["min_depth_samples"]))
    if args.allow_force_save:
        print("[WARN] --allow_force_save is ON. Gate-failed frames are diagnostic-only and must be excluded from Table 1.")
    print(f"  gripper board mask pad: {float(args.board_mask_pad_px):.1f}px")

    # (Board marker detection uses charuco.detect() directly — no separate detector needed)

    # ─── 로봇 클라이언트 ───
    # start 게이트 전에 연결을 끝내서 cv2 창이 응답 없음 상태가 되지 않도록 한다.
    manual_sock = None
    if args.use_robot and args.manual_robot:
        import socket as _sock
        manual_sock = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
        manual_sock.settimeout(None)
        print(f"[ManualRobot] Connecting to {args.robot_ip}:{args.robot_port} ...")
        manual_sock.connect((args.robot_ip, args.robot_port))
        print(f"[ManualRobot] Connected to {args.robot_ip}:{args.robot_port}")

    # ─── 메타 데이터 (기존 meta.json이 있으면 이어서 저장) ───
    meta_path = os.path.join(root, "meta.json")
    capture_config = {
        "schema_version": (
            "capture_config_v2" if final_protocol_mode else "capture_config_v1"
        ),
        "capture_protocol": active_capture_protocol,
        "waypoint_plan_sha256": waypoint_plan_sha256,
        "rig_geometry_path": rig_geometry_path,
        "rig_geometry_sha256": rig_geometry_sha256,
        "rig_geometry": rig_geometry_payload,
        "marker_gate_role": (
            "diagnostic_only" if final_protocol_mode else "capture_acceptance"
        ),
        "camera_storage_policy": (
            "all_connected_cameras_every_attempt"
            if final_protocol_mode
            else "legacy_block_dependent"
        ),
        "charuco_board_config": charuco_config_to_dict(charuco_cfg),
        "intrinsics_dir": os.path.abspath(intr_dir),
        "intrinsics_sha256_by_camera": intrinsics_sha256_by_camera,
        "width": int(args.width),
        "height": int(args.height),
        "fps": int(args.fps),
        "save_depth": bool(args.save_depth),
        "settle_time_s": float(args.settle_time),
        "cross_camera_timestamp_basis": "host_monotonic_receipt_v1",
        "min_markers_for_visibility": int(args.min_markers),
        "gripper_cube_min_aspect": float(args.gripper_cube_min_aspect),
        "board_mask_pad_px": float(args.board_mask_pad_px),
        "capture_gate": capture_gate_cfg,
        "allow_force_save": bool(args.allow_force_save),
        "allow_intrinsics_res_mismatch": bool(args.allow_intrinsics_res_mismatch),
        # Which cameras each block persists.  Part of capture_config so a session
        # cannot silently mix two storage policies across a resume — a set that
        # stored one A view of the fixed cameras is not comparable to one that
        # stored six.
        "a_fixed_cam_views_per_set": int(args.a_fixed_cam_views_per_set),
        "b_save_gripper_cam": bool(args.b_save_gripper_cam),
        # Per-camera exposure/gain/white balance actually in force. Recorded so a
        # later session can reproduce the same photometry, and so a session shot
        # with auto-exposure is identifiable after the fact rather than silently
        # mixed in with locked ones.
        "color_photometry": {
            str(ci): cams[ci].color_photometry for ci, _ in idx_serial_pairs
        },
    }
    capture_config_sha256 = canonical_json_sha256(capture_config)
    _unlocked = [ci for ci, _ in idx_serial_pairs
                 if not cams[ci].color_photometry.get("locked")]
    if _unlocked:
        print(f"[WARN] cam {_unlocked}: color photometry NOT locked — exposure will "
              f"drift between captures. Corner accuracy varies frame to frame.")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        meta_cfg, meta_cfg_source = load_cube_config_from_meta(root, default_cfg=cfg)
        if meta.get("captures") and not cube_configs_equivalent(meta_cfg, cfg):
            mismatch_keys = cube_config_mismatch_keys(cfg, meta_cfg)
            raise RuntimeError(
                "Existing meta.json uses a different cube definition.\n"
                f"Resolved cube config: {cube_cfg_source}\n"
                f"Session cube config: {meta.get('cube_config_source', meta_cfg_source)}\n"
                f"Differing fields: {', '.join(mismatch_keys) if mismatch_keys else 'unknown'}\n"
                "Use a new session folder, or run recompute_session_cube_pnp.py with the intended cube config before resuming."
            )
        if meta.get("captures"):
            try:
                meta_board_cfg, _ = load_charuco_config_from_meta(
                    root, require_frozen=True, default_cfg=charuco_cfg)
            except ValueError as error:
                raise RuntimeError(str(error)) from error
            if not charuco_configs_equivalent(meta_board_cfg, charuco_cfg):
                mismatch_keys = charuco_config_mismatch_keys(
                    charuco_cfg, meta_board_cfg)
                raise RuntimeError(
                    "Existing meta.json uses a different ChArUco definition.\n"
                    f"Resolved board: {charuco_cfg_source}\n"
                    f"Session board: {meta.get('charuco_board_config_source', 'unknown')}\n"
                    f"Differing fields: {', '.join(mismatch_keys)}\n"
                    "Use a new session folder; existing images must never be "
                    "reinterpreted with a different board geometry."
                )
        if meta.get("captures") and meta.get("capture_config") != capture_config:
            raise RuntimeError(
                "Existing meta.json has missing or different capture_config; refusing to mix "
                "resolutions/gates in one session. Use a new session folder."
            )
        event_id = max((int(c.get("event_id", -1)) for c in meta.get("captures", [])), default=-1) + 1
        print(f"[INFO] Resuming from existing meta.json ({len(meta['captures'])} captures, next event_id={event_id})")
    else:
        meta = {
            "root_folder": os.path.abspath(root),
            "session_allocation": (
                None if allocated_session is None else {
                    "session_id": allocated_session.session_id,
                    "session_index": int(allocated_session.index),
                    "session_root": allocated_session.session_root,
                    "manifest_path": allocated_session.manifest_path,
                    "policy": "max_existing_index_plus_one_no_reuse",
                }
            ),
            "gripper_cam_idx": gripper_cam_idx,
            "n_fixed_cams": n_fixed,
            "n_gripper_cams": n_gripper,
            "cam_indices": [ci for ci, _ in idx_serial_pairs],
            "cube_config_source": cube_cfg_source,
            "cube_config": cube_config_to_dict(cfg),
            "charuco_board_config_source": charuco_cfg_source,
            "charuco_board_config": charuco_config_to_dict(charuco_cfg),
            "capture_config": capture_config,
            "capture_config_sha256": capture_config_sha256,
            "capture_protocol": active_capture_protocol,
            "waypoint_plan_sha256": waypoint_plan_sha256,
            "waypoint_plan": waypoint_payload,
            "captures": [],
        }
        event_id = 0
        print("[INFO] New session (meta.json created)")

    # set_index -> A captures whose fixed-camera frames are already on disk.
    # Rebuilt from meta so resuming a session continues where it stopped instead
    # of storing a second redundant copy of every covered set.
    fixed_cam_stored: Dict[int, int] = {}
    for _cap in meta.get("captures", []):
        if _cap.get("cube_gripped") or _cap.get("set_index") is None:
            continue
        if any(rec.get("saved") and not rec.get("is_gripper")
               for rec in (_cap.get("cams") or {}).values()):
            _s = int(_cap["set_index"])
            fixed_cam_stored[_s] = fixed_cam_stored.get(_s, 0) + 1
    meta["cube_config_source"] = cube_cfg_source
    meta["charuco_board_config_source"] = charuco_cfg_source
    meta["charuco_board_config"] = charuco_config_to_dict(charuco_cfg)
    meta["capture_config"] = capture_config
    meta["capture_config_sha256"] = capture_config_sha256
    meta["capture_protocol"] = active_capture_protocol
    meta["waypoint_plan_sha256"] = waypoint_plan_sha256
    if rig_geometry_payload is not None:
        meta["rig_geometry"] = rig_geometry_payload
    if waypoint_payload is not None:
        meta["waypoint_plan"] = waypoint_payload
    if "cube_config" not in meta:
        meta["cube_config"] = cube_config_to_dict(cfg)
    else:
        meta["cube_config"] = cube_config_to_dict(cfg)
    quad_dir = ensure_dir(os.path.join(root, "marker_quads"))
    cam_order = sorted(ci for ci, _ in idx_serial_pairs)
    protocol_waypoint_by_id = {
        str(wp["planned_event_id"]): wp
        for wp in ((waypoint_payload or {}).get("waypoints") or [])
        if isinstance(wp, dict) and wp.get("planned_event_id") is not None
    }

    def build_frame_record(
        ci: int,
        color: np.ndarray,
        depth: Optional[np.ndarray],
        ts_ms: Optional[float],
        device_ts_ms: Optional[float] = None,
        device_timestamp_domain: Optional[str] = None,
        include_marker_poses: bool = True,
        include_charuco_pose: bool = True,
        log_pose_status: bool = False,
    ) -> dict:
        detect_info = detect_cube_markers_in_frame(
            color,
            cube,
            cube_ids=cfg.marker_ids,
            charuco=charuco,  # 전 카메라에서 ChArUco 검출 (고정캠 보드-전용 비교실험)
            is_gripper=(ci == gripper_cam_idx),
            board_mask_pad_px=float(args.board_mask_pad_px),
        )
        corners = detect_info["corners"]
        ids = detect_info["ids"]
        cube_img = detect_info["cube_image"]
        n_markers = 0 if ids is None else len(ids)
        fr = {
            "color": color,
            "depth": depth,
            "ts_ms": ts_ms,
            "host_monotonic_ts_ms": ts_ms,
            "device_ts_ms": device_ts_ms,
            "device_timestamp_domain": device_timestamp_domain,
            "ok": bool(n_markers >= args.min_markers),
            "n_markers": n_markers,
            "ids": ([] if ids is None else [int(x) for x in ids]),
            "corners": corners,
            "ids_np": ids,
            "marker_poses": [],
            "cube_pnp": None,
            "cube_detect_raw_ids": detect_info["raw_ids"],
            "cube_detect_filtered_ids": detect_info["filtered_ids"],
            "board_mask_applied": bool(detect_info["board_mask_applied"]),
        }
        # ChArUco 검출 결과는 모든 카메라에 저장 (고정캠도 보드 인식 -> 보드-전용 비교실험).
        fr["board_mkr_corners"] = detect_info["board_mkr_corners"]
        fr["board_mkr_ids"] = detect_info["board_mkr_ids"]
        fr["ch_corners"] = detect_info["ch_corners"]
        fr["ch_ids"] = detect_info["ch_ids"]
        fr["charuco_detect_n"] = int(detect_info["charuco_detect_n"])

        # Photometric quality of the regions the solver will actually use. Both
        # cube and board quads count: a board-only row (A0/B3) is gated on the
        # same evidence as a cube row.
        _quads = list(corners or [])
        _board_quads = detect_info.get("board_mkr_corners") or []
        _quads.extend(_board_quads)
        fr["roi_quality"] = marker_roi_quality(color, _quads)

        intr = cam_intrinsics.get(ci)
        if intr is not None and ids is not None and len(ids) > 0:
            K, D, depth_scale = intr
            if include_marker_poses:
                fr["marker_poses"] = estimate_per_marker_poses(
                    cube, corners, ids, K, D,
                    depth_u16=depth, depth_scale=depth_scale)

            min_cube_markers = args.gripper_cube_min_markers if ci == gripper_cam_idx else 1
            min_cube_aspect = args.gripper_cube_min_aspect if ci == gripper_cam_idx else 0.0
            pnp_ok, rvec, tvec, used_ids, reproj = cube.solve_pnp_cube(
                cube_img, K, D,
                use_ransac=True,
                min_markers=max(int(min_cube_markers), 1),
                return_reproj=True,
                min_aspect=float(min_cube_aspect),
                depth_u16=depth,
                depth_scale=depth_scale,
            )
            tag = "G" if ci == gripper_cam_idx else "F"
            if log_pose_status:
                if pnp_ok:
                    print(f"  [PnP] cam{ci}({tag}): OK ids={used_ids} reproj={reproj['err_mean']:.2f}px")
                else:
                    det_ids = [int(x) for x in ids] if ids is not None else []
                    print(
                        f"  [PnP] cam{ci}({tag}): FAILED "
                        f"(cube_ids={det_ids}, raw_ids={detect_info['raw_ids']}, mask={fr['board_mask_applied']})"
                    )
            if pnp_ok and rvec is not None:
                T_cam_cube = rodrigues_to_Rt(rvec, tvec)
                fr["cube_pnp"] = {
                    "ok": True,
                    "rvec": rvec.flatten().tolist(),
                    "tvec": tvec.flatten().tolist(),
                    "used_ids": [int(x) for x in used_ids],
                    "reproj_mean_px": reproj["err_mean"] if reproj else None,
                    "T_cam_cube_4x4": T_cam_cube.tolist(),
                    "min_markers_required": int(min_cube_markers),
                    "min_aspect_required": float(min_cube_aspect),
                    **depth_metrics_to_fields((reproj or {}).get("depth_metrics")),
                }

        # ChArUco 보드 pose 추정도 모든 카메라에서 (고정캠 보드 기반 외부파라미터 비교용).
        if include_charuco_pose and intr is not None:
            K, D, _ = intr
            try:
                ch_ok, ch_rvec, ch_tvec, ch_n, ch_reproj = charuco.estimate_pose(color, K, D)
            except Exception as e:
                ch_ok = False
                ch_n = int(fr.get("charuco_detect_n", 0))
                ch_reproj = None
                if log_pose_status:
                    print(f"  [ChArUco] pose ERROR: {e}")
            if ch_ok and ch_rvec is not None:
                T_cam_board = rodrigues_to_Rt(ch_rvec, ch_tvec)
                fr["charuco"] = {
                    "ok": True,
                    "n_corners": int(ch_n),
                    "reproj_error_px": float(ch_reproj) if ch_reproj is not None else None,
                    "rvec": ch_rvec.flatten().tolist(),
                    "tvec": ch_tvec.flatten().tolist(),
                    "T_cam_board_4x4": T_cam_board.tolist(),
                }
                if log_pose_status:
                    print(f"  [ChArUco] OK: {ch_n} corners, reproj={ch_reproj:.3f}px")
            elif log_pose_status:
                print(f"  [ChArUco] FAILED (corners={int(fr.get('charuco_detect_n', 0))})")

        return fr

    # ── start 게이트(옵션): --start_gate 시에만 첫 cv2 프리뷰 + 'start' 입력 대기 ──
    if args.start_gate:
        extra = []
        if args.use_robot:
            extra.append(f"robot {args.robot_ip}:{args.robot_port}"
                         + (" (manual)" if args.manual_robot else ""))
        if not wait_for_start_command_capture(cams, cam_order, gripper_cam_idx, extra,
                                               frame_builder=build_frame_record, cube=cube,
                                               preview_frac=float(args.preview_frac)):
            for cam in cams.values():
                cam.stop()
            cv2.destroyAllWindows()
            return

    print("\n[MODE] Final robot-server capture")
    print("  server/c1.py controls all 45 planned events")
    print("  ESC/q : abort\n")

    def do_capture(
        capture_gripper_pose_6dof: Optional[List[float]] = None,
        capture_pose_reference: Optional[str] = None,
        capture_tool_offset_6dof: Optional[List[float]] = None,
        place_pose_6dof: Optional[List[float]] = None,
        capture_index: Optional[int] = None,
        capture_robot_joints_6dof: Optional[List[float]] = None,
        capture_cube_center_6dof: Optional[List[float]] = None,
        set_cube_center_6dof: Optional[List[float]] = None,
        set_index: Optional[int] = None,
        cube_gripped: Optional[bool] = None,
        capture_block: Optional[str] = None,
        grasp_id: Optional[int] = None,
        force_save: bool = False,
        motion_safety: Optional[dict] = None,
        protocol_version: Optional[str] = None,
        planned_event_id: Optional[str] = None,
        phase: Optional[str] = None,
        target_state: Optional[str] = None,
        placement_id: Optional[str] = None,
        view_index: Optional[int] = None,
        attempt_index: int = 0,
        release_state: Optional[dict] = None,
        planned_waypoint: Optional[dict] = None,
        robot_state: Optional[dict] = None,
    ) -> Tuple[bool, dict]:
        """모든 카메라에서 마커별 포즈 추정과 함께 촬영."""
        nonlocal event_id
        pc_command_receive_epoch_s = float(time.time())
        pc_command_receive_monotonic_s = float(time.monotonic())

        is_final_protocol = protocol_version == PROTOCOL_COMPOSITE_RIG_45
        if is_final_protocol:
            expected_waypoint = protocol_waypoint_by_id.get(str(planned_event_id))
            if expected_waypoint is None:
                raise RuntimeError(
                    f"unknown planned_event_id from robot: {planned_event_id!r}"
                )
            expected_fields = {
                "capture_index": capture_index,
                "phase": phase,
                "target_state": target_state,
                "placement_id": placement_id,
                "view_index": view_index,
            }
            for field_name, received_value in expected_fields.items():
                if expected_waypoint.get(field_name) != received_value:
                    raise RuntimeError(
                        f"robot command mismatch for {planned_event_id}: "
                        f"{field_name}={received_value!r}, "
                        f"expected {expected_waypoint.get(field_name)!r}"
                    )
            if planned_waypoint != expected_waypoint:
                raise RuntimeError(
                    f"robot planned_waypoint differs from frozen plan for {planned_event_id}"
                )
            already_selected = next(
                (
                    cap
                    for cap in meta.get("captures", [])
                    if cap.get("planned_event_id") == planned_event_id
                    and cap.get("selected_for_analysis") is True
                ),
                None,
            )
            if already_selected is not None:
                print(
                    f"[PROTOCOL] {planned_event_id} already captured as "
                    f"event {already_selected.get('event_id')}"
                )
                return True, {
                    "reason": "planned event already has a transport-valid attempt",
                    "already_captured": True,
                    "event_id": already_selected.get("event_id"),
                }

        # 안정화 대기
        if args.settle_time > 0 and args.use_robot:
            time.sleep(args.settle_time)

        frames: Dict[int, dict] = {}

        # Software-sync: 공통 host-monotonic clock의 latest ts 중 가장 오래된 것(=가장 느린 카메라)을
        # 기준으로 잡고, 다른 카메라들은 버퍼에서 그 시각에 가장 가까운 프레임을 고른다.
        # 독립 RealSense device timestamp는 epoch가 다를 수 있으므로 동기화에 직접 쓰지 않는다.
        # 먼저 네 카메라의 최신 프레임이 서로 가까워질 때까지 기다린다. 한 대라도
        # 뒤처져 있으면 target_ts 가 그만큼 과거로 잡히고, 나머지 카메라 버퍼
        # (buffer_size 프레임 = 1초 미만)는 그 시각을 담고 있지 않아 수 초짜리 span 이
        # 만들어진다. 그러면 게이트가 전 프레임을 버린다 — 실제로 그렇게 되었다.
        _sync_deadline = time.time() + float(args.frame_sync_timeout_s)
        _lag = None
        while True:
            _ts = {ci: cam.get_latest()[2] for ci, cam in cams.items()}
            _have = {ci: t for ci, t in _ts.items() if t is not None}
            if len(_have) == len(cams):
                _lag = max(_have.values()) - min(_have.values())
                if _lag <= float(args.max_capture_span_ms):
                    break
            if time.time() >= _sync_deadline:
                if len(_have) < len(cams):
                    _missing = sorted(set(cams) - set(_have))
                    print(f"[SYNC] cam {_missing}: 프레임 없음 — 스트림이 끊겼는지 확인")
                else:
                    _newest = max(_have.values())
                    _behind = {ci: round(_newest - t, 1) for ci, t in _have.items()}
                    _worst = max(_behind, key=lambda k: _behind[k])
                    print(f"[SYNC] {float(args.frame_sync_timeout_s):.1f}s 안에 동기화 실패 "
                          f"(span {_lag:.0f}ms). 카메라별 지연(ms): {_behind}  "
                          f"-> cam{_worst} 가 가장 뒤처짐")
                break
            time.sleep(0.02)

        latest_ts_list = [t for t in
                          (cam.get_latest()[2] for cam in cams.values()) if t is not None]

        if latest_ts_list:
            target_ts = min(latest_ts_list)
            for ci, cam in cams.items():
                color, depth, ts_ms, device_ts_ms, ts_domain = cam.get_at_with_timestamps(target_ts)
                if color is None:
                    continue
                frames[ci] = build_frame_record(
                    ci, color, depth, ts_ms,
                    device_ts_ms=device_ts_ms,
                    device_timestamp_domain=ts_domain,
                    include_marker_poses=True,
                    include_charuco_pose=True,
                    log_pose_status=True,
                )
        else:
            for ci, cam in cams.items():
                color, depth, ts_ms, device_ts_ms, ts_domain = cam.get_latest_with_timestamps()
                if color is None:
                    continue
                frames[ci] = build_frame_record(
                    ci, color, depth, ts_ms,
                    device_ts_ms=device_ts_ms,
                    device_timestamp_domain=ts_domain,
                    include_marker_poses=True,
                    include_charuco_pose=True,
                    log_pose_status=True,
                )

        gate = evaluate_capture_gate(
            frames,
            capture_gate_cfg,
            gripper_cam_idx=gripper_cam_idx,
            capture_block=capture_block,
            cube_gripped=cube_gripped,
        )
        capture_span_ms = float(gate["capture_span_ms"])
        transport = evaluate_transport_integrity(
            frames,
            cam_order,
            max_capture_span_ms=float(args.max_capture_span_ms),
        )
        if not gate["pass"] and not is_final_protocol:
            if force_save and args.allow_force_save:
                # c+Enter 확인 시: 마커/게이트 실패여도 프레임을 무조건 저장한다.
                # (gate 결과는 meta에 남고, 04는 저장 이미지에서 다시 검출한다.)
                print(f"[FORCE-SAVE] gate 실패({gate['reason']}) 이지만 강제 저장")
            else:
                if force_save:
                    print("[WARN] robot requested force_save, but it is disabled without --allow_force_save")
                print(f"[SKIP] {gate['reason']}")
                return False, gate
        if is_final_protocol and not gate["pass"]:
            print(
                f"[DIAGNOSTIC] marker quality failed for {planned_event_id}: "
                f"{gate['reason']} (attempt is still stored)"
            )

        # ─── 저장 ───
        fid = int(event_id)
        cap_rec: dict = {
            "event_id": fid,
            "capture_index": capture_index,
            "capture_span_ms": float(capture_span_ms),
            "capture_gate": gate,
            "marker_quality_pass": bool(gate["pass"]),
            "transport_integrity": transport,
            "force_saved": bool(not gate["pass"] and force_save and args.allow_force_save),
            "pc_command_receive_epoch_s": pc_command_receive_epoch_s,
            "pc_command_receive_monotonic_s": pc_command_receive_monotonic_s,
            "pc_capture_record_epoch_s": float(time.time()),
            "cams": {},
        }

        if is_final_protocol:
            analysis_group_id = (
                f"P2:{placement_id}" if phase == PHASE_P2
                else "P3:STATIONARY_RIG" if phase == PHASE_P3
                else f"P1:{planned_event_id}"
            )
            cap_rec.update({
                "protocol_version": protocol_version,
                "planned_event_id": str(planned_event_id),
                "attempt_index": int(attempt_index),
                "phase": str(phase),
                "target_state": str(target_state),
                "placement_id": placement_id,
                "view_index": None if view_index is None else int(view_index),
                "planned_waypoint": planned_waypoint,
                "robot_state": robot_state,
                "release_state": release_state,
                "analysis_group_id": analysis_group_id,
                "split_unit_id": (
                    str(placement_id) if phase == PHASE_P2 else None
                ),
            })

        # 로봇 포즈 데이터
        # capture_pose = 이미지 촬영 시 현재 로봇 TCP
        # 05 calibration에서 참조: robot_pose_6dof / robot_pose_matrix_4x4
        robot_tcp = capture_gripper_pose_6dof or place_pose_6dof
        if robot_tcp is not None:
            tcp_f = [float(x) for x in robot_tcp]
            cap_rec["robot_pose_6dof"] = tcp_f        # calibration input contract
            cap_rec["capture_gripper_pose_6dof"] = tcp_f
            try:
                T44 = euler_deg_to_matrix(*tcp_f).tolist()
                cap_rec["robot_pose_matrix_4x4"] = T44  # calibration input contract
                cap_rec["capture_gripper_pose_matrix_4x4"] = T44
            except Exception:
                pass
        if capture_pose_reference is not None:
            cap_rec["capture_pose_reference"] = str(capture_pose_reference)
        if capture_tool_offset_6dof is not None:
            cap_rec["capture_tool_offset_6dof"] = [
                float(x) for x in capture_tool_offset_6dof
            ]

        if capture_robot_joints_6dof is not None:
            cap_rec["capture_robot_joints_6dof"] = [float(x) for x in capture_robot_joints_6dof]

        # 촬영 순간의 실제(live) 큐브 중점 = tool4 포즈(position). B(그립 스윕)에서는
        # set마다 큐브가 그리퍼에 붙어 움직이므로 pose별 이 값이 핵심 데이터가 된다.
        if capture_cube_center_6dof is not None:
            cap_rec["capture_cube_center_6dof"] = [float(x) for x in capture_cube_center_6dof]

        if set_cube_center_6dof is not None:
            cap_rec["set_cube_center_6dof"] = [float(x) for x in set_cube_center_6dof]

        if set_index is not None:
            cap_rec["set_index"] = set_index

        # capture-block tags let 05 separate placement and eye-to-hand frames.
        if cube_gripped is not None:
            cap_rec["cube_gripped"] = bool(cube_gripped)
        if capture_block is not None:
            cap_rec["capture_block"] = str(capture_block)
        if grasp_id is not None:
            cap_rec["grasp_id"] = int(grasp_id)
        if motion_safety is not None:
            cap_rec["motion_safety"] = motion_safety

        if place_pose_6dof is not None and place_pose_6dof != robot_tcp:
            cap_rec["place_pose_6dof"] = [float(x) for x in place_pose_6dof]
            try:
                cap_rec["place_pose_matrix_4x4"] = euler_deg_to_matrix(
                    *place_pose_6dof
                ).tolist()
            except Exception:
                pass

        # ─── 카메라별 저장 범위 (근거는 resolve_camera_storage 참조) ───
        is_placement = not bool(cube_gripped)
        sidx = None if set_index is None else int(set_index)
        storage = None
        if not is_final_protocol:
            storage = resolve_camera_storage(
                is_placement=is_placement,
                set_index=sidx,
                fixed_views_already_stored=fixed_cam_stored.get(sidx, 0),
                a_fixed_cam_views_per_set=int(args.a_fixed_cam_views_per_set),
                b_save_gripper_cam=bool(args.b_save_gripper_cam),
            )

        for ci in cam_order:
            if ci not in frames:
                cap_rec["cams"][str(ci)] = {
                    "saved": False,
                    "is_gripper": ci == gripper_cam_idx,
                    "skip_reason": "camera_frame_missing",
                    "transport_success": False,
                }
                continue
            fr = frames[ci]
            is_gripper_cam = (ci == gripper_cam_idx)

            # 저장하지 않는 카메라도 기록은 남긴다. 아래 소비자들이 모두
            # cams[ci]["saved"] 로 거르므로 (calibration_pipeline common/
            # observations, 05 calibration), 조용히 빠지지 않고 왜 없는지가 남는다.
            if storage is not None and (
                (is_gripper_cam and not storage.store_gripper)
                or (not is_gripper_cam and not storage.store_fixed)
            ):
                cap_rec["cams"][str(ci)] = {
                    "saved": False,
                    "is_gripper": is_gripper_cam,
                    "skip_reason": (storage.gripper_skip_reason if is_gripper_cam
                                    else storage.fixed_skip_reason),
                    "n_markers_detected": fr["n_markers"],
                    "cube_visible": fr["ok"],
                    "charuco_detect_n": int(fr.get("charuco_detect_n", 0)),
                }
                continue

            rgb_rel = f"cam{ci}/rgb_{fid:05d}.jpg"
            rgb_write_ok = bool(cv2.imwrite(os.path.join(root, rgb_rel), fr["color"]))

            depth_rel = None
            depth_write_ok = True
            if args.save_depth and fr["depth"] is not None:
                depth_rel = f"cam{ci}/depth_{fid:05d}.png"
                depth_write_ok = bool(
                    cv2.imwrite(os.path.join(root, depth_rel), fr["depth"])
                )
            elif args.save_depth:
                depth_write_ok = False

            cam_rec = {
                "saved": bool(rgb_write_ok and depth_write_ok),
                "transport_success": bool(rgb_write_ok and depth_write_ok),
                "rgb_write_ok": rgb_write_ok,
                "depth_write_ok": depth_write_ok,
                "is_gripper": (ci == gripper_cam_idx),
                "rgb_path": rgb_rel,
                "depth_path": depth_rel,
                "ts_ms": fr["ts_ms"],
                "host_monotonic_ts_ms": fr.get("host_monotonic_ts_ms"),
                "device_ts_ms": fr.get("device_ts_ms"),
                "device_timestamp_domain": fr.get("device_timestamp_domain"),
                "n_markers_detected": fr["n_markers"],
                "marker_ids": fr["ids"],
                "cube_visible": fr["ok"],
                "markers": fr["marker_poses"],  # per-marker PnP results
                "cube_detect_raw_ids": fr.get("cube_detect_raw_ids", []),
                "cube_detect_filtered_ids": fr.get("cube_detect_filtered_ids", []),
                "board_mask_applied": bool(fr.get("board_mask_applied", False)),
                # Saved on every frame, including ones the sharpness gate is not
                # yet enforcing — this is the pilot evidence the per-camera
                # threshold gets fixed from.
                "roi_quality": fr.get("roi_quality"),
            }
            # ChArUco 검출/pose 는 모든 카메라에 기록 (고정캠 보드-전용 비교실험).
            cam_rec["charuco_detect_n"] = int(fr.get("charuco_detect_n", 0))

            if fr["cube_pnp"] is not None:
                cam_rec["cube_pnp"] = fr["cube_pnp"]

            if fr.get("charuco") is not None:
                cam_rec["charuco"] = fr["charuco"]

            cap_rec["cams"][str(ci)] = cam_rec

        if is_final_protocol:
            write_failed_cams = sorted(
                int(ci)
                for ci, rec in cap_rec["cams"].items()
                if not rec.get("transport_success", False)
            )
            if write_failed_cams:
                transport["pass"] = False
                transport["status"] = "FAIL"
                transport["write_failed_camera_ids"] = write_failed_cams
                transport["reasons"].append(
                    "camera file write failed: {}".format(write_failed_cams)
                )
                transport["reason"] = "; ".join(transport["reasons"])
            else:
                transport["write_failed_camera_ids"] = []
            cap_rec["attempt_status"] = (
                "captured" if transport["pass"] else "transport_failed"
            )
            cap_rec["selected_for_analysis"] = bool(transport["pass"])
            cap_rec["retry_allowed"] = not bool(transport["pass"])

        # 게이트를 통과해 실제로 디스크에 쓴 뒤에만 센다. 게이트 실패는 위에서
        # 이미 return 했으므로 여기 도달한 캡처만 그 set 의 몫을 채운다.
        if (not is_final_protocol and is_placement and storage is not None
                and storage.store_fixed and sidx is not None):
            fixed_cam_stored[sidx] = fixed_cam_stored.get(sidx, 0) + 1

        meta["captures"].append(cap_rec)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # 마커 오버레이가 포함된 2x2 분할 이미지 저장
        quad = make_quad_image(frames, cam_order, cube, gripper_cam_idx)
        quad_path = os.path.join(quad_dir, f"frame_{fid:05d}.jpg")
        cv2.imwrite(quad_path, quad)

        # (프리뷰 스레드와 충돌 방지: imshow 제거, 파일로만 저장)

        # 요약 출력
        cam_summary = []
        for ci in sorted(frames.keys()):
            fr = frames[ci]
            tag = "G" if ci == gripper_cam_idx else "F"
            n = fr["n_markers"]
            cam_summary.append(f"cam{ci}({tag}):{n}mkr")
        # ChArUco 검출/pose 요약을 카메라별로 출력 (고정캠 보드 인식 확인용).
        ch_summary = []
        for ci in sorted(frames.keys()):
            rec = cap_rec["cams"].get(str(ci), {})
            ndet = int(rec.get("charuco_detect_n", 0))
            if ndet > 0 or "charuco" in rec:
                tag = "G" if ci == gripper_cam_idx else "F"
                npose = rec.get("charuco", {}).get("n_corners") if "charuco" in rec else None
                ch_summary.append(f"cam{ci}({tag}):{ndet}cor" + (f"/pose{npose}" if npose else ""))
        charuco_txt = (" charuco[" + " ".join(ch_summary) + "]") if ch_summary else ""
        if is_final_protocol:
            print(
                f"[SAVE] event={fid} planned={planned_event_id} "
                f"attempt={attempt_index} transport={transport['status']} "
                f"marker={gate['status']} | {' '.join(cam_summary)} "
                f"span={capture_span_ms:.1f}ms{charuco_txt}"
            )
        else:
            print(f"[SAVE] event={fid} | {' '.join(cam_summary)} span={capture_span_ms:.1f}ms{charuco_txt}")
        event_id += 1
        if is_final_protocol:
            result = dict(gate)
            result["transport_integrity"] = transport
            result["event_id"] = fid
            return bool(transport["pass"]), result
        return True, gate

    try:
        if args.use_robot and args.manual_robot:
            # ─── final robot-server mode ───
            # cv2는 main thread 전용. 소켓 recv는 백그라운드 스레드.
            # main thread가 recv에 블로킹되면 cv2 윈도우가 응답 없음 상태가 되므로
            # 분리한다.
            print("[MODE] Waiting for final protocol commands from server/c1.py")

            import threading

            # Waypoint accumulator (mirror of robot's capture_waypoints.json)
            wp_list: list = []
            wp_set_joints = None
            wp_set_tcp = None
            wp_set_cube_center = None
            wp_tool_pose_reference = None
            wp_tool_offset_6dof = None

            # PC-side teach recording (grip/pose/set). 서버가 recgrip/recpose/recset 시
            # 전체 리스트를 보내면 여기서 세션 번호 붙은 파일로 PC 에만 저장한다.
            teach = {"session": None}

            network_done = threading.Event()
            user_quit = threading.Event()

            # 짧은 timeout으로 recv가 주기적으로 깨어나 종료 플래그를 확인하게 함.
            manual_sock.settimeout(0.5)

            def network_loop():
                nonlocal wp_set_joints, wp_set_tcp, wp_set_cube_center
                nonlocal wp_tool_pose_reference, wp_tool_offset_6dof
                try:
                    recv_buf = b""
                    while not network_done.is_set() and not user_quit.is_set():
                        # Newline-delimited JSON framing: teach_save(전체 리스트) 처럼 큰
                        # 메시지나 분할/합쳐 도착한 메시지도 버퍼에 모아 완전한 한 줄씩 처리.
                        if b"\n" not in recv_buf:
                            try:
                                chunk = manual_sock.recv(65536)
                            except _sock.timeout:
                                continue
                            except OSError as e:
                                # 정상 종료 시 main thread가 socket을 닫아서 EBADF가 뜸 → 무시
                                if not (user_quit.is_set() or network_done.is_set()):
                                    print(f"[ManualRobot] socket error: {e}")
                                break
                            if not chunk:
                                print("[ManualRobot] Server disconnected.")
                                break
                            recv_buf += chunk
                            if b"\n" not in recv_buf:
                                continue

                        line, _, recv_buf = recv_buf.partition(b"\n")
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            msg = json.loads(line.decode("utf-8"))
                        except Exception as e:
                            print(f"[ManualRobot] JSON parse error: {e}")
                            continue

                        cmd = msg.get("command", "")
                        if cmd == "quit":
                            print("[ManualRobot] Server sent quit.")
                            break

                        if cmd == "teach_save":
                            # 로봇의 recgrip/recpose/recset 기록을 PC 에만 저장.
                            # 서버 세션(재연결)마다 새 번호 파일 (grip_poses_NNN.json 등).
                            kind = msg.get("kind")
                            entries = msg.get("data")
                            tool_pose_reference = msg.get("tool_pose_reference")
                            tool_offset_6dof = msg.get("tool_offset_6dof")
                            name = {"grip": "grip_poses", "pose": "capture_poses",
                                    "set": "capture_sets"}.get(kind)
                            if name is None or not isinstance(entries, list):
                                print(f"[Teach] invalid teach_save (kind={kind})")
                                continue
                            if teach["session"] is None:
                                # 세션은 그대로 두고 티칭 회차만 올린다. 로봇 서버를
                                # 다시 붙일 때마다 새 번호가 붙으므로 이전 회차 파일은
                                # 덮이지 않고 남는다.
                                n = 1
                                while any(os.path.exists(os.path.join(
                                        teach_dir, f"{b}_{n:03d}.json"))
                                        for b in ("grip_poses", "capture_poses", "capture_sets")):
                                    n += 1
                                teach["session"] = n
                                print(f"[Teach] recording round #{n:03d} "
                                      f"-> {teach_dir}/*_{n:03d}.json")
                            path = os.path.join(
                                teach_dir, f"{name}_{teach['session']:03d}.json")
                            try:
                                with open(path, "w") as tf:
                                    teach_data = {name: entries}
                                    if tool_pose_reference is not None:
                                        teach_data["tool_pose_reference"] = tool_pose_reference
                                    if tool_offset_6dof is not None:
                                        teach_data["tool_offset_6dof"] = tool_offset_6dof
                                    json.dump(teach_data, tf, indent=2)
                                print(f"[Teach] {kind}: {len(entries)} entries -> {path}")
                            except Exception as e:
                                print(f"[Teach] save error: {e}")
                            continue

                        if cmd == "request_waypoints":
                            wp_path = waypoints_path
                            print(f"[ManualRobot] Robot requested waypoints. Sending {wp_path}")
                            try:
                                with open(wp_path, "r") as wf:
                                    wp_data = json.load(wf)
                                validate_safe_joint_config(wp_data)
                                validate_waypoint_semantics(wp_data)
                                resp_msg = json.dumps({
                                    "action": "waypoints",
                                    "status": "ok",
                                    "waypoints_data": wp_data,
                                })
                                manual_sock.sendall((resp_msg + "\n").encode("utf-8"))
                                print(f"[ManualRobot]   sent {len(wp_data.get('waypoints', []))} waypoints")
                            except FileNotFoundError:
                                err = json.dumps({
                                    "action": "waypoints",
                                    "status": "error",
                                    "reason": f"file_not_found: {wp_path}",
                                })
                                manual_sock.sendall((err + "\n").encode("utf-8"))
                                print(f"[ManualRobot]   ERROR: file not found")
                            except Exception as e:
                                err = json.dumps({
                                    "action": "waypoints",
                                    "status": "error",
                                    "reason": str(e),
                                })
                                manual_sock.sendall((err + "\n").encode("utf-8"))
                                print(f"[ManualRobot]   ERROR: {e}")
                            continue

                        if cmd == "save_waypoints":
                            # teach_extend.py가 머지된 전체 waypoint 데이터를 통째로 보내며
                            # PC에 영구 저장을 요청. 기존 파일은 .bak으로 백업한 뒤 덮어씀.
                            wp_path = waypoints_path
                            wp_data = msg.get("waypoints_data")
                            if not isinstance(wp_data, dict):
                                err = json.dumps({
                                    "action": "save_waypoints",
                                    "status": "error",
                                    "reason": "missing_or_invalid_waypoints_data",
                                })
                                try:
                                    manual_sock.sendall((err + "\n").encode("utf-8"))
                                except OSError:
                                    break
                                continue
                            try:
                                if os.path.exists(wp_path):
                                    bak_path = wp_path + ".bak"
                                    shutil.copyfile(wp_path, bak_path)
                                    print(f"[ManualRobot]   backup: {bak_path}")
                                with open(wp_path, "w") as wf:
                                    json.dump(wp_data, wf, indent=2)
                                n_wp = len(wp_data.get("waypoints", []))
                                print(f"[ManualRobot] Waypoints saved by robot: {wp_path} ({n_wp} poses)")
                                resp_msg = json.dumps({
                                    "action": "save_waypoints",
                                    "status": "ok",
                                    "n_waypoints": n_wp,
                                })
                                # robot 측에 저장이 끝났음을 알려주면 robot 측 wp_list 재기록을
                                # 막을 수 있도록 표시. 메인 thread에서는 wp_list가 비어있을 때만
                                # 저장하므로, 이 메시지 처리 시 wp_list를 비워두면 두 번 안 씀.
                                wp_list.clear()
                            except Exception as e:
                                resp_msg = json.dumps({
                                    "action": "save_waypoints",
                                    "status": "error",
                                    "reason": str(e),
                                })
                                print(f"[ManualRobot]   save error: {e}")
                            try:
                                manual_sock.sendall((resp_msg + "\n").encode("utf-8"))
                            except OSError:
                                break
                            continue

                        if cmd == "protocol_complete":
                            protocol_name = msg.get("protocol_version")
                            if protocol_name != PROTOCOL_COMPOSITE_RIG_45:
                                response = {
                                    "action": "protocol_complete",
                                    "status": "error",
                                    "reason": f"unknown protocol {protocol_name!r}",
                                }
                            else:
                                expected_ids = list(protocol_waypoint_by_id)
                                selected_ids = {
                                    str(cap.get("planned_event_id"))
                                    for cap in meta.get("captures", [])
                                    if cap.get("selected_for_analysis") is True
                                }
                                missing_ids = [
                                    event_name
                                    for event_name in expected_ids
                                    if event_name not in selected_ids
                                ]
                                attempt_counts = {
                                    event_name: sum(
                                        1
                                        for cap in meta.get("captures", [])
                                        if cap.get("planned_event_id") == event_name
                                    )
                                    for event_name in expected_ids
                                }
                                completion = {
                                    "schema_version": "capture_protocol_manifest_v1",
                                    "protocol_version": PROTOCOL_COMPOSITE_RIG_45,
                                    "waypoint_plan_sha256": waypoint_plan_sha256,
                                    "rig_geometry_sha256": rig_geometry_sha256,
                                    "capture_config_sha256": capture_config_sha256,
                                    "expected_event_count": len(expected_ids),
                                    "selected_event_count": len(selected_ids),
                                    "missing_planned_event_ids": missing_ids,
                                    "attempt_counts": attempt_counts,
                                    "server_report": {
                                        "planned_events": msg.get("planned_events"),
                                        "successful_events": msg.get("successful_events"),
                                        "failed_events": msg.get("failed_events"),
                                    },
                                    "completed": not missing_ids and len(expected_ids) == 45,
                                    "locked_at_epoch_s": float(time.time()),
                                }
                                completion_path = os.path.join(
                                    root, "capture_protocol_manifest.json"
                                )
                                with open(completion_path, "w", encoding="utf-8") as handle:
                                    json.dump(completion, handle, indent=2)
                                    handle.write("\n")
                                meta["protocol_completion"] = completion
                                with open(meta_path, "w", encoding="utf-8") as handle:
                                    json.dump(meta, handle, indent=2)
                                    handle.write("\n")
                                response = {
                                    "action": "protocol_complete",
                                    "status": "ok" if completion["completed"] else "incomplete",
                                    "missing_planned_event_ids": missing_ids,
                                    "manifest_path": "capture_protocol_manifest.json",
                                }
                            try:
                                manual_sock.sendall(
                                    (json.dumps(response) + "\n").encode("utf-8")
                                )
                            except OSError:
                                break
                            continue

                        if cmd == "capture":
                            capture_tcp = msg.get("capture_gripper_pose_6dof")
                            pose_idx = msg.get("capture_index", event_id)
                            r_joints = msg.get("capture_robot_joints_6dof")
                            live_cube = msg.get("capture_cube_center_6dof")
                            s_cube = msg.get("set_cube_center_6dof")
                            s_idx = msg.get("set_index")
                            m_set_joints = msg.get("set_joints")
                            m_set_tcp = msg.get("set_tcp")
                            m_place_joints = msg.get("place_joints")
                            m_gripped = msg.get("cube_gripped")
                            m_block = msg.get("capture_block")
                            m_grasp = msg.get("grasp_id")
                            m_force = msg.get("force_save")
                            m_motion_safety = msg.get("motion_safety")
                            m_pose_reference = msg.get("capture_pose_reference")
                            m_tool_offset = msg.get("capture_tool_offset_6dof")
                            m_protocol = msg.get("protocol_version")
                            m_planned_event_id = msg.get("planned_event_id")
                            m_phase = msg.get("phase")
                            m_target_state = msg.get("target_state")
                            m_placement_id = msg.get("placement_id")
                            m_view_index = msg.get("view_index")
                            m_attempt_index = msg.get("attempt_index", 0)
                            m_release_state = msg.get("release_state")
                            m_planned_waypoint = msg.get("planned_waypoint")
                            m_robot_state = msg.get("robot_state")
                            if m_pose_reference is not None:
                                wp_tool_pose_reference = m_pose_reference
                            if m_tool_offset is not None:
                                wp_tool_offset_6dof = m_tool_offset

                            print(f"\n[ManualRobot] Capture signal received (capture_index={pose_idx}, set_index={s_idx})")
                            if capture_tcp:
                                print(f"  TCP: {capture_tcp}")
                            if r_joints:
                                print(f"  Joints: {r_joints}")

                            saved, gate = do_capture(
                                capture_gripper_pose_6dof=capture_tcp,
                                capture_pose_reference=m_pose_reference,
                                capture_tool_offset_6dof=m_tool_offset,
                                capture_index=pose_idx,
                                capture_robot_joints_6dof=r_joints,
                                capture_cube_center_6dof=live_cube,
                                set_cube_center_6dof=s_cube,
                                set_index=s_idx,
                                cube_gripped=m_gripped,
                                capture_block=m_block,
                                grasp_id=m_grasp,
                                force_save=bool(m_force),
                                motion_safety=m_motion_safety,
                                protocol_version=m_protocol,
                                planned_event_id=m_planned_event_id,
                                phase=m_phase,
                                target_state=m_target_state,
                                placement_id=m_placement_id,
                                view_index=m_view_index,
                                attempt_index=m_attempt_index,
                                release_state=m_release_state,
                                planned_waypoint=m_planned_waypoint,
                                robot_state=m_robot_state,
                            )

                            if gate.get("already_captured"):
                                status = "already_captured"
                            elif saved:
                                status = "success"
                            elif m_protocol == PROTOCOL_COMPOSITE_RIG_45:
                                status = "retryable"
                            else:
                                status = "skipped"
                            response_reason = gate.get("reason")
                            if gate.get("transport_integrity") is not None:
                                response_reason = gate["transport_integrity"].get("reason")
                            resp = json.dumps({
                                "action": "captured",
                                "status": status,
                                "reason": response_reason,
                                "event_id": gate.get("event_id"),
                                "marker_quality_pass": gate.get("pass"),
                            })
                            try:
                                manual_sock.sendall((resp + "\n").encode("utf-8"))
                            except OSError:
                                break

                            if m_set_joints is not None:
                                wp_set_joints = m_set_joints
                            if m_set_tcp is not None:
                                wp_set_tcp = m_set_tcp
                            if s_cube is not None:
                                wp_set_cube_center = s_cube

                            wp_entry = {
                                "capture_index": pose_idx,
                                "capture_joints": r_joints,
                                "capture_tcp": capture_tcp,
                                "cube_center_6dof": msg.get("capture_cube_center_6dof"),
                                "set_index": s_idx,
                            }
                            if m_protocol is not None:
                                wp_entry.update({
                                    "protocol_version": m_protocol,
                                    "planned_event_id": m_planned_event_id,
                                    "attempt_index": m_attempt_index,
                                    "phase": m_phase,
                                    "target_state": m_target_state,
                                    "placement_id": m_placement_id,
                                    "view_index": m_view_index,
                                    "robot_state": m_robot_state,
                                    "release_state": m_release_state,
                                    "planned_waypoint": m_planned_waypoint,
                                    "capture_status": status,
                                })
                            if m_pose_reference is not None:
                                wp_entry["capture_pose_reference"] = m_pose_reference
                            if m_tool_offset is not None:
                                wp_entry["capture_tool_offset_6dof"] = m_tool_offset
                            if m_place_joints is not None:
                                wp_entry["place_joints"] = m_place_joints
                            wp_list.append(wp_entry)

                            if saved:
                                print(f"[OK] Capture {pose_idx} saved")
                            else:
                                print(f"[SKIP] Capture {pose_idx} skipped")

                        else:
                            print(f"[ManualRobot] Unknown command: {cmd}")
                except Exception as e:
                    import traceback
                    print(f"[ManualRobot] network thread crashed: {e}")
                    traceback.print_exc()
                finally:
                    network_done.set()

            net_thread = threading.Thread(target=network_loop, daemon=True)
            net_thread.start()
            if args.show:
                print("[INFO] Live preview started (4-camera quad view)")

            try:
                # Main thread: cv2 preview만 담당. recv는 net_thread.
                while not network_done.is_set():
                    if args.show:
                        live_frames = {}
                        for ci, cam in cams.items():
                            color, depth, ts_ms = cam.get_latest()
                            if color is None:
                                continue
                            live_frames[ci] = build_frame_record(
                                ci, color, depth, ts_ms,
                                include_marker_poses=False,
                                include_charuco_pose=False,
                                log_pose_status=False,
                            )

                        if live_frames:
                            quad = make_quad_image(live_frames, cam_order, cube, gripper_cam_idx)
                            gate = evaluate_capture_gate(
                                live_frames,
                                capture_gate_cfg,
                                gripper_cam_idx=gripper_cam_idx,
                            )
                            gate_lines = build_capture_gate_lines(gate, gripper_cam_idx, live_frames)
                            gate_colors = [(0, 255, 0)] if gate["pass"] else [(0, 0, 255)]
                            gate_colors = gate_colors + [(255, 255, 255)] * (len(gate_lines) - 1)
                            quad = append_status_footer(quad, gate_lines, gate_colors)
                            ph = int(quad.shape[0] * 0.6)
                            pw = int(quad.shape[1] * 0.6)
                            preview = cv2.resize(quad, (pw, ph))
                            cv2.imshow("Capture Preview",
                                       fit_to_screen(preview, float(args.preview_frac)))

                        key = cv2.waitKey(50) & 0xFF
                        if key == 27 or key == ord('q'):
                            print("[ManualRobot] User quit preview.")
                            user_quit.set()
                            break
                    else:
                        time.sleep(0.1)

            finally:
                network_done.set()
                user_quit.set()
                try:
                    manual_sock.shutdown(_sock.SHUT_RDWR)
                except Exception:
                    pass
                try:
                    manual_sock.close()
                except Exception:
                    pass
                net_thread.join(timeout=2.0)

                # 이번 세션에 실제 촬영된 waypoint 기록(미러)을 저장한다.
                # 주의: 입력 파일 capture_waypoints.json (생성기 산출물, request_waypoints 가
                # 읽는 파일)을 덮어쓰면 안 되므로 별도 파일 capture_waypoints_recorded.json 에 쓴다.
                if wp_list:
                    wp_save = {
                        "set_joints": wp_set_joints,
                        "set_tcp": wp_set_tcp,
                        "set_cube_center": wp_set_cube_center,
                        "capture_protocol": active_capture_protocol,
                        "waypoint_plan_sha256": waypoint_plan_sha256,
                        "waypoints": wp_list,
                    }
                    if wp_tool_pose_reference is not None:
                        wp_save["tool_pose_reference"] = wp_tool_pose_reference
                    if wp_tool_offset_6dof is not None:
                        wp_save["tool_offset_6dof"] = wp_tool_offset_6dof
                    wp_path = os.path.join(root, "capture_waypoints_recorded.json")
                    with open(wp_path, "w") as f:
                        json.dump(wp_save, f, indent=2)
                    print(f"[INFO] Recorded waypoints saved: {wp_path} ({len(wp_list)} poses)")

            print(f"\n[DONE] Final robot capture complete. {event_id} captures saved.")

    finally:
        for cam in cams.values():
            cam.stop()
        cv2.destroyAllWindows()

    print(f"\n[DONE] Total captures: {event_id}")
    print(f"  Meta saved: {meta_path}")
