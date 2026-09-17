"""
ChArUco 보드로 카메라 color intrinsics (K, D)를 정밀 보정한다.

01은 RealSense 공장(factory) intrinsics를 그대로 덤프한다. 그런데 D415/D435
color 스트림의 공장 왜곡계수(D)는 전부 0으로 보고되어 실제 렌즈 왜곡이 보정되지
않는다. 이 스크립트는 ChArUco 보드를 라이브 대화형으로 촬영해 각 카메라의 color
K, D를 직접 추정하고, 기존 intrinsics/cam{idx}.npz의 color_K/color_D만 교체한다.
depth 관련 필드(depth_K, depth_scale, R_depth_to_color 등)와 해상도/시리얼은 그대로
보존하므로 다운스트림 03~05 코드는 수정할 필요가 없다.

전제조건:
  - 먼저 01_export_intrinsics.py 를 실행해 device_map.json 과 cam{idx}.npz
    (depth 필드 포함)를 만들어 두어야 한다.
  - --board 로 고른 보드 정의(targets/charuco_boards/*.json)가 실제 인쇄된 보드와
    일치해야 한다. 생략하면 config.py 의 CharucoBoardConfig 기본 보드를 쓴다.

동작:
  1) device_map.json 로드 (serial -> cam_idx, gripper_cam_idx)
  2) 연결된 각 카메라를 하나씩 열고, ChArUco 보드를 흔들며 다양한 각도/위치에서
     프레임을 수집 (SPACE 로 수동 그랩). 화면에 검출/커버리지/선명도 피드백.
  3) board.matchImagePoints + cv2.calibrateCamera 로 2-pass(이상치 제거) 보정
  4) intrinsics/cam{idx}.npz 의 color_K/color_D 교체 (factory 값은 백업 보존)
  5) intrinsics/charuco_intrinsics_report.json 리포트 저장

--capture_only: 보드 검출 없이 SPACE마다 raw_capture/에 원본 PNG와 카메라 정보를 저장.
--from_images: 카메라 없이 raw_capture/를 읽고 현재 지정한 보드 정의로 보정.
--images_dir: 다른 이미지 수집 폴더를 선택 (기본: intr_dir/raw_capture).
--joint_intr_dir: (--from_images) 같은 카메라를 다른 해상도로 찍은 폴더의 사진까지
  기준 해상도 좌표로 환산해 한 번에 보정한다. 공장 K 가 해상도 비율로 정확히
  비례할 때만 합치고, 두 폴더에 비율로 환산한 K 와 같은 D 를 쓴다.
--zero_tangent: 접선 왜곡 p1, p2 를 0 으로 고정.

키 조작 (카메라별 수집 중):
  SPACE : 현재 프레임 그랩
  u     : 마지막 그랩 취소(undo)
  c/Enter: 이 카메라 수집 종료 -> 다음 카메라 촬영
  s     : 이 카메라 건너뛰기 (factory 값 유지)
  q     : 전체 중단 (이미 저장된 이미지와 이전 카메라 보정은 유지)

명령어 예시:
  python3 02_calibrate_intrinsics.py --list_boards
  python3 02_calibrate_intrinsics.py --intr_dir ./intrinsics --board 9x6_id90
  python3 02_calibrate_intrinsics.py --intr_dir intrinsics_1280x720 \
      --joint_intr_dir intrinsics_1920x1080_rgbd720 \
      --from_images --use_factory_guess --zero_tangent --board 9x6_id90
"""

import os
import json
import time
import argparse
from dataclasses import replace

import numpy as np
import cv2

from calibration_pipeline.charuco import CharucoTarget
from calibration_pipeline.board_config import (
    charuco_board_dir, charuco_config_to_dict, charuco_topology,
    describe_charuco_config, list_charuco_boards, resolve_charuco_config,
)
from capture_pipeline.intrinsics_images import (
    IntrinsicsImages, collect_raw_for_camera, load_image_views,
)


def _make_intrinsics_target(cfg, legacy_pattern=None):
    """Build the detector for ``cfg``.

    ``legacy_pattern`` is an optional override kept for callers that probe
    alternate layouts; normally the flag travels on the board definition
    itself (targets/charuco_boards/*.json -> CharucoBoardConfig.legacy_pattern).
    """
    if legacy_pattern is not None:
        cfg = replace(cfg, legacy_pattern=bool(legacy_pattern))
    return CharucoTarget(cfg)


def _print_board_catalog():
    names = list_charuco_boards()
    if not names:
        print(f"[INFO] 등록된 보드 정의 없음: {charuco_board_dir()}")
        return
    print(f"[INFO] 보드 정의 디렉터리: {charuco_board_dir()}")
    for name in names:
        cfg, source = resolve_charuco_config(name)
        print(f"  --board {name}")
        print(f"      {describe_charuco_config(cfg)}")
        print(f"      {source}")


def _intrinsics_board_config(target):
    """Board definition as stored in the report, read back off the built board
    so a report can never disagree with what actually did the detecting."""
    config = charuco_config_to_dict(target.cfg)
    config["dictionary"] = config.pop("dictionary_name")
    config["legacy_pattern"] = (
        bool(target.board.getLegacyPattern())
        if hasattr(target.board, "getLegacyPattern")
        else bool(getattr(target.cfg, "legacy_pattern", False)))
    return config


def _diagnose_rejected_grab(color, target, n_corners, min_corners, cam_idx, save_dir):
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    if target.detector is not None:
        _, raw_ids, rejected = target.detector.detectMarkers(gray)
    else:
        _, raw_ids, rejected = cv2.aruco.detectMarkers(
            gray, target.dictionary, parameters=target.det_params)
    raw_ids = [] if raw_ids is None else sorted(int(i) for i in raw_ids.reshape(-1))
    matched_ids = sorted(set(raw_ids) & target.board_id_set)
    board_config = _intrinsics_board_config(target)
    diagnostic = {
        "opencv": cv2.__version__, "cam_idx": int(cam_idx),
        "board": board_config, "charuco_corners": int(n_corners),
        "raw_marker_ids": raw_ids, "matching_marker_ids": matched_ids,
        "rejected_marker_candidates": len(rejected), "layout_candidates": [],
    }
    print(f"[DIAG] raw ArUco IDs={raw_ids}; matching board IDs={matched_ids}; "
          f"expected IDs={int(target.board_ids[0])}..{int(target.board_ids[-1])}")
    if not raw_ids:
        print("[DIAG] 이 dictionary로 디코딩된 마커가 없습니다. 원본 진단 이미지에서 "
              "마커 종류, 크기, 선명도, 반사를 확인하세요.")
    elif not matched_ids:
        print("[DIAG] 검출 ID가 설정된 보드 범위 밖입니다. dictionary와 시작 ID를 확인하세요.")

    # Probe alternate layouts only on a rejected keypress; never change the capture target.
    if len(matched_ids) >= 2:
        cfg = target.cfg
        sizes = dict.fromkeys([(cfg.squares_x, cfg.squares_y), (cfg.squares_y, cfg.squares_x)])
        for sx, sy in sizes:
            legacy_options = [False, True] if sy % 2 == 0 else [False]
            for legacy in legacy_options:
                candidate = {"squares_x": sx, "squares_y": sy, "legacy_pattern": legacy}
                try:
                    probe = _make_intrinsics_target(
                        replace(cfg, squares_x=sx, squares_y=sy), legacy_pattern=legacy)
                    candidate["charuco_corners"] = int(probe.detect(color)[2])
                except (cv2.error, RuntimeError, ValueError) as error:
                    candidate.update(charuco_corners=0, error=str(error))
                diagnostic["layout_candidates"].append(candidate)
                print(f"[DIAG] layout {sx}x{sy}, legacy={legacy}: "
                      f"charuco corners={candidate['charuco_corners']}")
                if candidate["charuco_corners"] >= min_corners and (
                    sx != cfg.squares_x or sy != cfg.squares_y
                    or legacy != board_config["legacy_pattern"]
                ):
                    flags = f"--squares_x {sx} --squares_y {sy}"
                    flags += " --legacy_pattern" if legacy else " --no-legacy_pattern"
                    print(f"[DIAG] 검출 가능한 배치 후보: {flags}. "
                          "원본 인쇄물과 일치하면 targets/charuco_boards/ 의 보드 정의 JSON 을 "
                          "고치거나 새로 만든 뒤 --board 로 재실행하세요.")

    if save_dir:
        diagnostic_dir = os.path.join(save_dir, "diagnostics")
        os.makedirs(diagnostic_dir, exist_ok=True)
        stem = os.path.join(diagnostic_dir, f"rejected_{time.time_ns()}")
        if not cv2.imwrite(stem + ".png", color):
            raise OSError(f"Could not write {stem}.png")
        with open(stem + ".json", "w", encoding="utf-8") as stream:
            json.dump(diagnostic, stream, indent=2)
        print(f"[DIAG] 원본 이미지: {os.path.abspath(stem + '.png')}")
        print(f"[DIAG] 검출 정보: {os.path.abspath(stem + '.json')}")
    return diagnostic


# ---------------------------------------------------------------------------
# 보정 헬퍼
# ---------------------------------------------------------------------------
def _sharpness(gray: np.ndarray) -> float:
    """Laplacian variance = 선명도 지표 (높을수록 선명)."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _obj_img_from_charuco(board, ch_corners, ch_ids):
    """ChArUco 코너/ID -> (objectPoints Nx1x3 f32, imagePoints Nx1x2 f32).

    OpenCV 4.7+ 의 board.matchImagePoints 를 우선 사용 (이 환경 4.13 지원 확인됨).
    구버전 대비 chessboardCorners 인덱싱 폴백도 둔다.
    """
    if ch_corners is None or ch_ids is None or len(ch_ids) < 4:
        return None, None

    if hasattr(board, "matchImagePoints"):
        try:
            obj, img = board.matchImagePoints(ch_corners, ch_ids)
            if obj is not None and img is not None and len(obj) >= 4:
                return obj.reshape(-1, 1, 3).astype(np.float32), img.reshape(-1, 1, 2).astype(np.float32)
        except Exception:
            pass

    # 폴백: chessboardCorners 를 ID로 인덱싱
    if hasattr(board, "getChessboardCorners"):
        chess = np.asarray(board.getChessboardCorners())
    elif hasattr(board, "chessboardCorners"):
        chess = np.asarray(board.chessboardCorners)
    else:
        return None, None
    chess = chess.reshape(-1, 3)
    ids = np.asarray(ch_ids).reshape(-1)
    if ids.max() >= len(chess):
        return None, None
    obj = chess[ids].reshape(-1, 1, 3).astype(np.float32)
    img = np.asarray(ch_corners).reshape(-1, 1, 2).astype(np.float32)
    return obj, img


def _run_calib(views, image_size, flags, K0=None, D0=None):
    """views: [(obj, img), ...] -> (rms, K, D, per_view_rms)."""
    obj_list = [v[0] for v in views]
    img_list = [v[1] for v in views]

    if K0 is not None:
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
            obj_list, img_list, image_size,
            K0.copy(), (None if D0 is None else D0.copy()),
            flags=flags | cv2.CALIB_USE_INTRINSIC_GUESS,
        )
    else:
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
            obj_list, img_list, image_size, None, None, flags=flags
        )

    per_view = []
    for i, (o, im) in enumerate(views):
        proj, _ = cv2.projectPoints(o, rvecs[i], tvecs[i], K, D)
        diff = proj.reshape(-1, 2) - im.reshape(-1, 2)
        per_view.append(float(np.sqrt(np.mean(np.sum(diff * diff, axis=1)))))
    return float(rms), K, D, per_view


def calibrate_intrinsics(board, accepted, image_size, flags, K0=None, D0=None):
    """2-pass 보정: 1차 보정 -> per-view 이상치 제거 -> 2차 보정.

    accepted: [(ch_corners, ch_ids), ...]
    반환: dict 또는 None (뷰 부족). used_index 는 최종 보정에 쓰인 accepted 의
    인덱스이고 per_view / used_points 와 같은 순서다.
    """
    views, index = [], []
    for i, (ch_c, ch_id) in enumerate(accepted):
        obj, img = _obj_img_from_charuco(board, ch_c, ch_id)
        if obj is not None:
            views.append((obj, img))
            index.append(i)

    if len(views) < 4:
        return None

    rms, K, D, per = _run_calib(views, image_size, flags, K0, D0)
    per_arr = np.asarray(per)
    thr = float(per_arr.mean() + per_arr.std())

    kept = [k for k, e in enumerate(per) if e <= thr]
    dropped = len(views) - len(kept)

    if len(kept) >= 4 and dropped > 0:
        keep = [views[k] for k in kept]
        rms2, K2, D2, per2 = _run_calib(keep, image_size, flags, K0, D0)
        return {
            "rms": rms2, "K": K2, "D": D2,
            "n_used": len(keep), "n_total": len(views),
            "n_dropped": dropped, "reject_thr_px": thr,
            "per_view": per2,
            "used_index": [index[k] for k in kept],
            "used_points": [len(views[k][0]) for k in kept],
        }

    return {
        "rms": rms, "K": K, "D": D,
        "n_used": len(views), "n_total": len(views),
        "n_dropped": 0, "reject_thr_px": thr,
        "per_view": per,
        "used_index": index,
        "used_points": [len(v[0]) for v in views],
    }


# ---------------------------------------------------------------------------
# 라이브 수집 (카메라 1대)
# ---------------------------------------------------------------------------
def _draw_overlay(vis, coverage, n_accepted, sharp, blur_thr,
                  n_corners, cov_cols, cov_rows, n_markers=0):
    h, w = vis.shape[:2]

    # 커버리지 그리드
    for r in range(cov_rows):
        for c in range(cov_cols):
            x0 = int(c * w / cov_cols)
            y0 = int(r * h / cov_rows)
            x1 = int((c + 1) * w / cov_cols)
            y1 = int((r + 1) * h / cov_rows)
            hit = coverage[r, c] > 0
            col = (0, 180, 0) if hit else (60, 60, 60)
            cv2.rectangle(vis, (x0, y0), (x1, y1), col, 1)

    covered = int((coverage > 0).sum())
    total_cells = cov_cols * cov_rows

    sharp_ok = sharp >= blur_thr
    lines = [
        f"cam views: {n_accepted}   coverage: {covered}/{total_cells} cells",
        f"board markers: {n_markers}   charuco corners: {n_corners}   "
        f"sharp: {sharp:.0f} ({'OK' if sharp_ok else 'BLUR'})",
        f"[SPACE]grab [u]undo [c]done [s]skip [q]quit",
    ]
    y = 22
    for i, t in enumerate(lines):
        color = (255, 255, 255)
        if i == 1 and not sharp_ok:
            color = (0, 165, 255)
        cv2.putText(vis, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y += 22
    return vis


def collect_for_camera(cam, target, cam_idx, is_gripper, args, save_dir):
    """한 카메라에서 프레임 수집. 반환: (status, accepted, image_size)
    status: 'done' | 'skip' | 'abort'
    """
    win = f"cam{cam_idx} charuco intrinsics ({'GRIPPER' if is_gripper else 'FIXED'})"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    accepted = []          # [(ch_corners, ch_ids), ...]
    coverage = np.zeros((args.cov_rows, args.cov_cols), dtype=int)
    image_size = None

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print(f"\n[cam{cam_idx}] 보드를 카메라 앞에서 다양한 각도/거리/위치로 움직이세요. "
          f"원하는 만큼 그랩 후 c/Enter로 보정. (SPACE=그랩, c=보정, s=건너뛰기)")

    while True:
        color, _, _ = cam.get_latest()
        if color is None:
            if (cv2.waitKey(30) & 0xFF) == ord('q'):
                cv2.destroyWindow(win)
                return "abort", accepted, image_size
            continue

        h, w = color.shape[:2]
        image_size = (w, h)
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        sharp = _sharpness(gray)

        ch_c, ch_id, n_corners, m_c, m_id = target.detect(color)

        vis = color.copy()
        if m_c is not None and len(m_c) > 0:
            cv2.aruco.drawDetectedMarkers(vis, m_c, m_id)
        if ch_c is not None and ch_id is not None and n_corners > 0:
            cv2.aruco.drawDetectedCornersCharuco(vis, ch_c, ch_id, (0, 255, 255))

        # grab 은 아래 SPACE 키에서만 True 로 바뀐다 (자동 그랩 없음 — 전부 수동).
        do_grab = False

        _draw_overlay(vis, coverage, len(accepted), sharp,
                      args.blur_thresh, n_corners, args.cov_cols, args.cov_rows,
                      n_markers=0 if m_id is None else len(m_id))
        cv2.imshow(win, vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyWindow(win)
            return "abort", accepted, image_size
        elif key == ord('s'):
            cv2.destroyWindow(win)
            return "skip", accepted, image_size
        elif key in (ord('c'), 13, 10):
            cv2.destroyWindow(win)
            return "done", accepted, image_size
        elif key == ord('u'):
            if accepted:
                accepted.pop()
                # 커버리지는 근사치 유지 (정확 복원 대신 재계산)
                coverage[:] = 0
                for cc, _ in accepted:
                    for px, py in cc.reshape(-1, 2):
                        cc_col = min(args.cov_cols - 1, int(px * args.cov_cols / w))
                        cc_row = min(args.cov_rows - 1, int(py * args.cov_rows / h))
                        coverage[cc_row, cc_col] += 1
                print(f"[cam{cam_idx}] undo -> {len(accepted)} views")
        elif key == ord(' '):
            if ch_c is not None and n_corners >= args.min_corners:
                do_grab = True
            else:
                print(f"[cam{cam_idx}] 그랩 불가: charuco 코너 {n_corners} < "
                      f"{args.min_corners} (보드를 더 잘 보이게)")
                try:
                    _diagnose_rejected_grab(
                        color, target, n_corners, args.min_corners, cam_idx, save_dir)
                except (cv2.error, OSError, RuntimeError, ValueError) as error:
                    print(f"[WARN] 그랩 실패 진단 저장 중 오류: {error}")

        if do_grab:
            accepted.append((ch_c, ch_id))
            for px, py in ch_c.reshape(-1, 2):
                cc_col = min(args.cov_cols - 1, int(px * args.cov_cols / w))
                cc_row = min(args.cov_rows - 1, int(py * args.cov_rows / h))
                coverage[cc_row, cc_col] += 1
            if save_dir:
                fn = os.path.join(save_dir, f"view_{len(accepted):03d}.png")
                cv2.imwrite(fn, color)
            print(f"[cam{cam_idx}] grab #{len(accepted)}  corners={n_corners}  sharp={sharp:.0f}")


# ---------------------------------------------------------------------------
# npz 갱신 / 리포트
# ---------------------------------------------------------------------------
# 두 해상도 스트림이 같은 화각인지 판단하는 공장 K 허용오차 (합칠 쪽 픽셀 단위).
# RealSense 공장 K 는 1280 <-> 1920 이 부동소수 오차 수준으로 정확히 비례한다.
JOINT_FACTORY_TOL_PX = 0.5


def _joint_scale(size, joint_size):
    """이 스트림 픽셀 -> 합칠 스트림 픽셀 배율. 가로/세로 배율이 같아야 한다."""
    sx = joint_size[0] / size[0]
    sy = joint_size[1] / size[1]
    if abs(sx - sy) > 1e-9:
        raise ValueError(
            f"가로세로 비율이 다름: {size[0]}x{size[1]} vs {joint_size[0]}x{joint_size[1]}")
    return sx


def _scale_K(K, scale):
    """공장 K 와 같은 규약(fx, fy, cx, cy 에 배율을 그대로 곱함)으로 K 환산."""
    scaled = np.asarray(K, dtype=np.float64).copy()
    scaled[:2] *= scale
    return scaled


def _rms_by_source(result, n_first):
    """최종 보정의 뷰별 오차를 accepted[:n_first] 와 나머지로 나눠 코너 수 가중 RMS 로.

    반환: ([rms_first, rms_rest], [뷰 수_first, 뷰 수_rest]). 해당 뷰가 없으면 rms 는 None.
    """
    sums = [[0.0, 0], [0.0, 0]]
    views = [0, 0]
    for i, err, n in zip(result["used_index"], result["per_view"], result["used_points"]):
        side = 0 if i < n_first else 1
        sums[side][0] += err * err * n
        sums[side][1] += n
        views[side] += 1
    return [float(np.sqrt(sq / n)) if n else None for sq, n in sums], views


def _open_joint_source(joint_intr_dir, joint_images_dir, intr_dir, base_images):
    """--joint_intr_dir 폴더의 장치 맵과 원본 이미지를 연다. 문제가 있으면 ValueError."""
    if os.path.abspath(joint_intr_dir) == os.path.abspath(intr_dir):
        raise ValueError("--joint_intr_dir 가 --intr_dir 와 같습니다")
    map_path = os.path.join(joint_intr_dir, "device_map.json")
    if not os.path.exists(map_path):
        raise ValueError(f"{map_path} 없음")
    with open(map_path, "r") as f:
        joint_map = json.load(f)
    serial_to_idx = {str(k): int(v) for k, v in joint_map.get("serial_to_idx", {}).items()}
    images = IntrinsicsImages(
        joint_images_dir or os.path.join(joint_intr_dir, "raw_capture"), serial_to_idx)
    if images.directory == base_images.directory:
        raise ValueError("합칠 원본 이미지 폴더가 기준 폴더와 같습니다")
    return {
        "intr_dir": joint_intr_dir, "serial_to_idx": serial_to_idx,
        "gripper_cam_idx": joint_map.get("gripper_cam_idx"), "images": images,
    }


def _load_joint_camera(joint, serial, size, factory_K, target, min_corners):
    """합칠 폴더에서 같은 시리얼 카메라의 검출 결과를 기준 해상도 좌표로 환산해 읽는다."""
    idx = joint["serial_to_idx"].get(serial)
    if idx is None:
        raise ValueError(f"serial {serial} 이 {joint['intr_dir']} 장치 맵에 없음")
    npz_path = os.path.join(joint["intr_dir"], f"cam{idx}.npz")
    if not os.path.exists(npz_path):
        raise ValueError(f"{npz_path} 없음")
    with np.load(npz_path, allow_pickle=True) as archive:
        d = dict(archive)
    if str(d["serial"]) != serial:
        raise ValueError(f"{npz_path} serial={d['serial']} 가 장치 맵 serial={serial} 와 다름")
    joint_size = (int(d["color_w"]), int(d["color_h"]))
    scale = _joint_scale(size, joint_size)
    joint_factory_K = np.asarray(d.get("factory_color_K", d["color_K"]), dtype=np.float64)
    gap = float(np.max(np.abs(joint_factory_K - _scale_K(factory_K, scale))))
    if gap > JOINT_FACTORY_TOL_PX:
        raise ValueError(
            f"공장 K 가 {scale:g}배 관계가 아님 (최대 차이 {gap:.2f}px) - "
            "같은 화각의 스트림이 아니면 합칠 수 없음")
    joint["images"].check_camera(idx, serial, *joint_size, int(d["fps"]))
    accepted, details = load_image_views(joint["images"], idx, target, min_corners)
    if not accepted:
        # 자기 사진 없이 다른 해상도 결과로 덮어쓰지 않는다 (촬영 전 폴더 보호).
        raise ValueError(
            f"합칠 폴더에 이 카메라의 보드 검출 사진이 없음 "
            f"({joint['images'].count(idx)}장 중 0장)")
    return {
        "idx": idx, "npz_path": npz_path, "scale": scale, "size": joint_size,
        "factory_K": joint_factory_K,
        "factory_D": np.asarray(d.get("factory_color_D", d["color_D"]), dtype=np.float64),
        "is_gripper": idx == joint["gripper_cam_idx"],
        "accepted": [((corners / scale).astype(np.float32), ids) for corners, ids in accepted],
        "num_images": joint["images"].count(idx), "image_observations": details,
    }


def _write_joint_camera(report, joint_report, joint, joint_cam, cam_idx, serial, is_gripper,
                        npz_path, backup_dir, K, D, result, n_base, factory_K, factory_D,
                        source_info):
    """합친 보정 결과를 두 폴더에 쓴다. K 는 해상도 배율로 환산하고 D 는 그대로."""
    scale = joint_cam["scale"]
    (rms_base, rms_joint), (used_base, used_joint) = _rms_by_source(result, n_base)
    if rms_joint is not None:
        rms_joint *= scale  # 기준 해상도 픽셀 -> 합칠 폴더 픽셀
    n_joint = len(joint_cam["accepted"])
    K_joint = _scale_K(K, scale)
    for rms, label in ((rms_base, "기준"), (rms_joint, "합칠")):
        if rms is None:
            print(f"[WARN] cam{cam_idx}: {label} 폴더 사진이 최종 보정에 한 장도 남지 않았습니다. "
                  "K 는 다른 해상도 사진만으로 정해졌습니다.")
    def fmt(rms):
        return "n/a" if rms is None else f"{rms:.4f}px"
    print(f"[cam{cam_idx}] 합친 RMS {result['rms']:.4f}px (기준 해상도) | "
          f"기준 {fmt(rms_base)} ({used_base}/{n_base}) | "
          f"합칠 {fmt(rms_joint)} ({used_joint}/{n_joint}, "
          f"{joint_cam['size'][0]}x{joint_cam['size'][1]} 픽셀)")

    sides = (
        dict(report=report, idx=cam_idx, npz=npz_path, backup=backup_dir, K=K,
             factory_K=factory_K, factory_D=factory_D, rms=rms_base, used=used_base,
             total=n_base, is_gripper=is_gripper, other_dir=joint["intr_dir"],
             other_idx=joint_cam["idx"], other_rms=rms_joint, to_other=scale,
             pooled=result["rms"], info=source_info),
        dict(report=joint_report, idx=joint_cam["idx"], npz=joint_cam["npz_path"],
             backup=os.path.join(joint["intr_dir"], "factory_backup"), K=K_joint,
             factory_K=joint_cam["factory_K"], factory_D=joint_cam["factory_D"],
             rms=rms_joint, used=used_joint, total=n_joint,
             is_gripper=joint_cam["is_gripper"],
             other_dir=joint_report["joint_with"]["intr_dir"], other_idx=cam_idx,
             other_rms=rms_base, to_other=1.0 / scale, pooled=result["rms"] * scale,
             info={"num_images": joint_cam["num_images"],
                   "image_observations": joint_cam["image_observations"]}),
    )
    for side in sides:
        extra = {
            "charuco_joint_intr_dir": os.path.abspath(side["other_dir"]),
            "charuco_joint_cam_idx": int(side["other_idx"]),
            "charuco_joint_scale_to_other": float(side["to_other"]),
            "charuco_joint_rms_px_pooled": float(side["pooled"]),
        }
        rms = float("nan") if side["rms"] is None else side["rms"]
        backup_path = overwrite_color_intrinsics(
            side["npz"], side["backup"], side["K"], D,
            {"rms": rms, "n_used": side["used"]}, serial, extra=extra)
        print(f"[SAVE] {side['npz']} (color_K/color_D 교체)  factory backup -> {backup_path}")
        side["report"]["cameras"][str(side["idx"])] = {
            "serial": serial, "is_gripper": bool(side["is_gripper"]), "status": "written",
            "rms_px": side["rms"], "num_views_used": side["used"],
            "num_views_total": side["total"], "num_dropped": side["total"] - side["used"],
            "K": np.asarray(side["K"]).tolist(), "D": np.asarray(D).flatten().tolist(),
            "factory_K": np.asarray(side["factory_K"]).tolist(),
            "factory_D": np.asarray(side["factory_D"]).flatten().tolist(),
            "joint": {
                "intr_dir": os.path.abspath(side["other_dir"]),
                "cam_idx": int(side["other_idx"]),
                "scale_to_other": float(side["to_other"]),
                "other_rms_px": side["other_rms"],
                "rms_px_pooled": float(side["pooled"]),
                "num_views_used_pooled": int(result["n_used"]),
                "num_views_total_pooled": int(result["n_total"]),
            },
            **side["info"],
        }


def _carry_over_joint_entries(joint, joint_report, joint_notes, base_serials):
    """합칠 폴더 리포트를 새로 쓸 때, 이번에 쓰지 않은 카메라는 npz 가 그대로이므로
    이전 리포트 항목을 유지하고 합치지 못한 이유만 덧붙인다."""
    previous = {}
    report_path = os.path.join(joint["intr_dir"], "charuco_intrinsics_report.json")
    if os.path.exists(report_path):
        try:
            with open(report_path, "r") as f:
                previous = json.load(f)
        except (OSError, json.JSONDecodeError):
            previous = {}
    for serial, idx in joint["serial_to_idx"].items():
        key = str(idx)
        if key in joint_report["cameras"]:
            continue
        old = (previous.get("cameras") or {}).get(key)
        if isinstance(old, dict) and old.get("serial") == serial:
            entry = {**old, "previous_calibrated_at": previous.get("calibrated_at")}
        else:
            entry = {"serial": serial, "status": "not_calibrated"}
        if serial in joint_notes:
            entry["joint_unavailable"] = joint_notes[serial]
        elif serial not in base_serials:
            entry["joint_unavailable"] = "기준 폴더 실행에 없는 카메라"
        joint_report["cameras"][key] = entry


def overwrite_color_intrinsics(npz_path, backup_dir, K, D, result, serial, extra=None):
    """cam{idx}.npz 의 color_K/color_D 만 교체하고 나머지 필드는 보존.
    최초 1회에 한해 원본(factory) 전체를 backup_dir 에 복사하고,
    npz 안에도 factory_color_K/D 를 남긴다. extra 는 charuco_joint_* 같은
    부가 기록이며, 이전 실행의 charuco_joint_* 는 항상 지운다.
    """
    d = dict(np.load(npz_path, allow_pickle=True))

    # 파일 단위 factory 백업 (최초 1회만 — 재실행해도 진짜 factory 보존)
    os.makedirs(backup_dir, exist_ok=True)
    base = os.path.basename(npz_path)
    backup_path = os.path.join(backup_dir, base)
    if not os.path.exists(backup_path):
        np.savez(backup_path, **d)

    # npz 내부에도 최초 factory 값 보존
    if "factory_color_K" not in d:
        d["factory_color_K"] = np.asarray(d["color_K"], dtype=np.float64)
        d["factory_color_D"] = np.asarray(d["color_D"], dtype=np.float64)

    d["color_K"] = np.asarray(K, dtype=np.float64)
    d["color_D"] = np.asarray(D, dtype=np.float64).reshape(-1, 1)
    d["intrinsics_source"] = "charuco"
    d["charuco_reproj_error_px"] = float(result["rms"])
    d["charuco_num_views"] = int(result["n_used"])
    d["charuco_calibrated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    for key in [k for k in d if k.startswith("charuco_joint_")]:
        del d[key]
    d.update(extra or {})

    np.savez(npz_path, **d)
    return backup_path


def main():
    parser = argparse.ArgumentParser(
        description="Intrinsic용 원본 RGB 촬영 및 저장 이미지/실시간 ChArUco 보정")
    parser.add_argument("--intr_dir", type=str, default="intrinsics")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--capture_only", action="store_true",
                      help="보드 검출/보정 없이 SPACE마다 원본 RGB 저장")
    mode.add_argument("--from_images", action="store_true",
                      help="카메라 없이 저장된 원본 이미지로 보정")
    parser.add_argument("--images_dir", type=str, default=None,
                        help="원본 이미지 폴더 (기본: intr_dir/raw_capture)")
    parser.add_argument("--min_views", type=int, default=12,
                        help="보정에 필요한 최소 검출 장수 (capture_only에서는 적용 안 함)")
    parser.add_argument("--min_corners", type=int, default=8,
                        help="한 프레임을 그랩하기 위한 최소 charuco 코너 수")
    parser.add_argument("--blur_thresh", type=float, default=60.0,
                        help="Laplacian variance 기준값 (화면 sharp OK/BLUR 표시용)")
    parser.add_argument("--cov_cols", type=int, default=4)
    parser.add_argument("--cov_rows", type=int, default=3)
    parser.add_argument("--save_images", action=argparse.BooleanOptionalAction, default=True,
                        help="실시간 보정 프레임을 charuco_capture/에 저장 (capture_only는 항상 저장)")
    parser.add_argument("--reset_devices", action=argparse.BooleanOptionalAction, default=False,
                        help="시작 시 모든 RealSense 하드웨어 리셋 (기본: 리셋 안 함)")
    parser.add_argument("--rational", action="store_true",
                        help="8-계수 CALIB_RATIONAL_MODEL 사용 (기본: 5-계수 Brown-Conrady)")
    parser.add_argument("--fix_aspect", action="store_true",
                        help="fx/fy 비율 고정 (CALIB_FIX_ASPECT_RATIO)")
    parser.add_argument("--use_factory_guess", action="store_true",
                        help="factory K를 초기 추정값으로 사용 (수렴 안정화)")
    parser.add_argument("--zero_tangent", action="store_true",
                        help="접선 왜곡 p1, p2를 0으로 고정 (CALIB_ZERO_TANGENT_DIST). "
                             "주점과 p1/p2가 서로 보상하며 흔들리는 것을 막는다")
    parser.add_argument("--joint_intr_dir", type=str, default=None,
                        help="--from_images 전용. 같은 카메라를 다른 해상도로 찍은 intrinsics 폴더. "
                             "두 폴더 사진을 합쳐 한 번에 보정하고, K는 해상도 비율로 환산해 "
                             "두 폴더의 cam*.npz와 리포트에 함께 쓴다 (D는 동일)")
    parser.add_argument("--joint_images_dir", type=str, default=None,
                        help="--joint_intr_dir의 원본 이미지 폴더 (기본: joint_intr_dir/raw_capture)")
    # 어떤 보드를 썼는지는 targets/charuco_boards/*.json 이 단일 소스다.
    parser.add_argument("--board", type=str, default=None,
                        help="사용할 ChArUco 보드 정의. targets/charuco_boards/ 의 이름"
                             f" ({', '.join(list_charuco_boards()) or '없음'}) 또는 JSON 경로."
                             " 생략하면 config.py 기본 보드")
    parser.add_argument("--list_boards", action="store_true",
                        help="등록된 보드 정의를 출력하고 종료")
    # 일회성 실험용 개별 override. 실제 보드가 바뀌었다면 JSON 을 새로 만들 것.
    parser.add_argument("--squares_x", type=int, default=None)
    parser.add_argument("--squares_y", type=int, default=None)
    parser.add_argument("--square_len_m", type=float, default=None)
    parser.add_argument("--marker_len_m", type=float, default=None)
    parser.add_argument("--dictionary", type=str, default=None)
    parser.add_argument("--marker_id_start", type=int, default=None)
    parser.add_argument("--legacy_pattern", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="OpenCV 4.6 이전의 짝수 행 ChArUco 인쇄 배치 (기본: 보드 정의값)")
    args = parser.parse_args()
    if args.list_boards:
        _print_board_catalog()
        return
    if args.capture_only and not args.save_images:
        parser.error("--capture_only requires image saving; remove --no-save_images")
    if args.images_dir and not (args.capture_only or args.from_images):
        parser.error("--images_dir requires --capture_only or --from_images")
    if args.joint_intr_dir and not args.from_images:
        parser.error("--joint_intr_dir requires --from_images")
    if args.joint_images_dir and not args.joint_intr_dir:
        parser.error("--joint_images_dir requires --joint_intr_dir")

    intr_dir = args.intr_dir
    map_path = os.path.join(intr_dir, "device_map.json")
    print(f"[INFO] Intrinsics directory: {os.path.abspath(intr_dir)}")
    print(f"[INFO] Device map: {os.path.abspath(map_path)}")
    if not os.path.exists(map_path):
        print(f"[ERROR] {map_path} 없음. 먼저 01_export_intrinsics.py 를 실행하세요.")
        return
    with open(map_path, "r") as f:
        dev_map = json.load(f)
    serial_to_idx = dev_map.get("serial_to_idx", {})
    gripper_cam_idx = dev_map.get("gripper_cam_idx", None)
    if not serial_to_idx:
        print("[ERROR] device_map.json 에 serial_to_idx 가 비어있음.")
        return

    # 보드 설정 구성: 정의 파일 -> 개별 override. RAW 촬영은 보드와 무관해야 하므로
    # 보드 인자를 아예 해석하지 않는다 (잘못된 보드 정의로 촬영이 막히면 안 됨).
    cfg = board_source = target = None
    if args.capture_only:
        print("[INFO] RAW capture only: 보드 검출과 최소 코너/장수 조건 없이 원본만 저장합니다.")
    else:
        try:
            cfg, board_source = resolve_charuco_config(args.board, {
                "squares_x": args.squares_x,
                "squares_y": args.squares_y,
                "square_length_m": args.square_len_m,
                "marker_length_m": args.marker_len_m,
                "dictionary_name": args.dictionary,
                "marker_id_start": args.marker_id_start,
                "legacy_pattern": args.legacy_pattern,
            })
        except (FileNotFoundError, ValueError) as error:
            print(f"[ERROR] {error}")
            return
        print(f"[INFO] ChArUco board source: {board_source}")
        print(f"[INFO] ChArUco board: {describe_charuco_config(cfg)}")
        if charuco_topology(cfg)["legacy_pattern_matters"]:
            print("[INFO] squares_y 가 짝수라 legacy 여부로 결과가 갈립니다. 인쇄물 좌상단 칸이 "
                  "검정이면 legacy_pattern=true 입니다.")
        target = _make_intrinsics_target(cfg)

    # 보정 flags
    flags = 0
    if args.rational:
        flags |= cv2.CALIB_RATIONAL_MODEL
    if args.fix_aspect:
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
    if args.zero_tangent:
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
    if not args.capture_only:
        print(f"[INFO] dist model: {'RATIONAL(8)' if args.rational else 'BROWN-CONRADY(5)'}"
              f"{' zero-tangent' if args.zero_tangent else ''}  OpenCV {cv2.__version__}")

    if args.from_images:
        connected = serial_to_idx
    else:
        try:
            from capture_pipeline.camera import RealSenseCamera
        except ModuleNotFoundError as error:
            if error.name == "pyrealsense2":
                raise SystemExit(
                    "[ERROR] pyrealsense2가 없습니다. 촬영은 RealSense Python 환경에서, "
                    "저장 이미지 보정은 --from_images로 실행하세요.") from error
            raise
        if args.reset_devices:
            RealSenseCamera.reset_all_devices()
        connected = RealSenseCamera.list_devices()  # {serial: name}
        unknown_serials = sorted(set(connected) - set(serial_to_idx))
        if unknown_serials:
            raise SystemExit(
                f"[ERROR] Connected cameras missing from {map_path}: {unknown_serials}. "
                "Use the same --intr_dir as the 01 --out_dir; the selected device map is stale."
            )
        missing_serials = sorted(set(serial_to_idx) - set(connected))
        if missing_serials:
            print(f"[WARN] 맵에 있으나 연결되지 않은 카메라: {missing_serials}")
    idx_pairs = sorted(
        [(int(serial_to_idx[s]), s) for s in connected if s in serial_to_idx],
        key=lambda x: x[0],
    )
    if not idx_pairs:
        print("[ERROR] device_map 에 매핑된 연결 카메라가 없음.")
        return
    print(f"[INFO] {'촬영' if args.capture_only else '재보정'} 대상 {len(idx_pairs)}대: "
          + ", ".join(f"cam{i}({'GRIP' if i == gripper_cam_idx else 'FIX'})" for i, _ in idx_pairs))
    if gripper_cam_idx is None:
        print("[WARN] gripper_cam_idx가 비어 있어 모두 FIXED로 표시됩니다. "
              "intrinsic 보정은 가능하지만 03 실행 전 그리퍼 카메라 지정이 필요합니다.")

    images = None
    if args.capture_only or args.from_images:
        images_dir = args.images_dir or os.path.join(intr_dir, "raw_capture")
        try:
            images = IntrinsicsImages(images_dir, serial_to_idx, create=args.capture_only)
        except ValueError as error:
            raise SystemExit(f"[ERROR] {error}") from error
        print(f"[INFO] Raw images: {images.directory}")

    joint = None
    if args.joint_intr_dir:
        try:
            joint = _open_joint_source(
                args.joint_intr_dir, args.joint_images_dir, intr_dir, images)
        except ValueError as error:
            raise SystemExit(f"[ERROR] --joint_intr_dir: {error}") from error
        print(f"[INFO] Joint images: {joint['images'].directory} "
              "(두 폴더 사진을 합쳐 보정하고 두 폴더에 함께 기록)")

    backup_dir = os.path.join(intr_dir, "factory_backup")
    report = {
        ("captured_at" if args.capture_only else "calibrated_at"): time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "capture_only" if args.capture_only else "from_images" if args.from_images else "live",
        "opencv": cv2.__version__,
        "dist_model": None if args.capture_only else "rational8" if args.rational else "brown_conrady5",
        "calib_flags": None if args.capture_only else {
            "use_factory_guess": args.use_factory_guess, "fix_aspect": args.fix_aspect,
            "zero_tangent": args.zero_tangent,
        },
        "board": None if args.capture_only else _intrinsics_board_config(target),
        "board_source": board_source,
        "board_topology": None if cfg is None else charuco_topology(cfg),
        "cameras": {},
    }
    if images is not None:
        report["images_dir"] = str(images.directory)
    joint_report = None
    joint_notes = {}  # serial -> 합칠 폴더에 쓰지 못한 이유
    if joint is not None:
        joint_report = {**report, "images_dir": str(joint["images"].directory),
                        "joint_with": {"intr_dir": os.path.abspath(intr_dir),
                                       "images_dir": str(images.directory)},
                        "cameras": {}}
        report["joint_with"] = {"intr_dir": os.path.abspath(joint["intr_dir"]),
                                "images_dir": str(joint["images"].directory)}

    aborted = False
    for cam_idx, serial in idx_pairs:
        npz_path = os.path.join(intr_dir, f"cam{cam_idx}.npz")
        if not os.path.exists(npz_path):
            print(f"[WARN] {npz_path} 없음 -> cam{cam_idx} 건너뜀 (01 먼저 실행).")
            report["cameras"][str(cam_idx)] = {"serial": serial, "status": "no_npz"}
            continue

        with np.load(npz_path, allow_pickle=True) as archive:
            d0 = dict(archive)
        if str(d0["serial"]) != serial:
            raise SystemExit(
                f"[ERROR] {npz_path} serial={d0['serial']} does not match "
                f"device_map serial={serial}. Use the matching 01 output directory."
            )
        w = int(d0["color_w"]); h = int(d0["color_h"]); fps = int(d0["fps"])
        factory_K = np.asarray(d0.get("factory_color_K", d0["color_K"]), dtype=np.float64)
        factory_D = np.asarray(d0.get("factory_color_D", d0["color_D"]), dtype=np.float64)
        is_gripper = (cam_idx == gripper_cam_idx)

        print(f"\n{'='*64}\n[cam{cam_idx}] serial={serial}  {w}x{h}@{fps}  "
              f"{'GRIPPER' if is_gripper else 'FIXED'}\n{'='*64}")

        if images is not None:
            try:
                images.check_camera(cam_idx, serial, w, h, fps, create=args.capture_only)
            except ValueError as error:
                raise SystemExit(f"[ERROR] {error}") from error

        source_info = {}
        joint_cam = None
        if args.from_images:
            accepted, details = load_image_views(images, cam_idx, target, args.min_corners)
            image_size, status = (w, h), "done"
            source_info = {"num_images": images.count(cam_idx), "image_observations": details}
            print(f"[cam{cam_idx}] 저장 이미지 {images.count(cam_idx)}장 중 "
                  f"보드 검출 {len(accepted)}장")
            if joint is not None:
                try:
                    joint_cam = _load_joint_camera(
                        joint, serial, (w, h), factory_K, target, args.min_corners)
                    if not accepted:
                        raise ValueError("기준 폴더에 이 카메라의 보드 검출 사진이 없음")
                except ValueError as error:
                    joint_cam = None
                    print(f"[WARN] cam{cam_idx} 합친 보정 불가: {error} "
                          "-> 이 폴더 사진만으로 따로 보정 (합칠 폴더는 그대로)")
                    source_info["joint_unavailable"] = str(error)
                    joint_notes[serial] = str(error)
            if joint_cam is not None:
                n_base = len(accepted)
                accepted = accepted + joint_cam["accepted"]
                print(f"[cam{cam_idx}] + 합칠 폴더 cam{joint_cam['idx']} "
                      f"{joint_cam['size'][0]}x{joint_cam['size'][1]} "
                      f"{joint_cam['num_images']}장 중 보드 검출 {len(joint_cam['accepted'])}장 "
                      f"(좌표 ÷{joint_cam['scale']:g})")
        else:
            cam = RealSenseCamera(serial, width=w, height=h, fps=fps,
                                  use_color=True, use_depth=False)
            try:
                cam.start(max_attempts=1)
            except Exception as e:
                print(f"[WARN] cam{cam_idx} 시작 실패: {e} -> 건너뜀")
                report["cameras"][str(cam_idx)] = {"serial": serial, "status": "start_failed"}
                continue

            save_dir = (os.path.join(intr_dir, "charuco_capture", f"cam{cam_idx}")
                        if args.save_images else None)
            try:
                if args.capture_only:
                    status = collect_raw_for_camera(cam, images, cam_idx)
                else:
                    status, accepted, image_size = collect_for_camera(
                        cam, target, cam_idx, is_gripper, args, save_dir)
            finally:
                cam.stop()

        if args.capture_only:
            report["cameras"][str(cam_idx)] = {
                "serial": serial, "status": status, "num_images": images.count(cam_idx),
            }
            if status == "abort":
                aborted = True
                break
            continue

        if status == "abort":
            print("[INFO] 사용자 중단(q). 지금까지 쓴 것 외에는 변경 없음.")
            aborted = True
            break
        if status == "skip":
            print(f"[cam{cam_idx}] 건너뜀 -> factory 값 유지.")
            report["cameras"][str(cam_idx)] = {
                "serial": serial, "status": "skipped", "num_views": len(accepted)}
            continue

        if len(accepted) < args.min_views:
            print(f"[cam{cam_idx}] 수집 {len(accepted)} < min_views {args.min_views} "
                  f"-> 보정 생략, factory 유지.")
            report["cameras"][str(cam_idx)] = {
                "serial": serial, "status": "too_few_views", "num_views": len(accepted), **source_info}
            if joint_cam is not None:
                joint_notes[serial] = f"합친 보정 실패: too_few_views ({len(accepted)}장)"
            continue

        if image_size is None:
            image_size = (w, h)

        K0 = factory_K if args.use_factory_guess else None
        try:
            result = calibrate_intrinsics(target.board, accepted, image_size, flags, K0=K0)
        except cv2.error as error:
            print(f"[WARN] cam{cam_idx} 보정 오류: {error}")
            source_info["error"] = str(error)
            result = None
        if result is None:
            print(f"[cam{cam_idx}] 보정 실패 (유효 뷰 부족).")
            report["cameras"][str(cam_idx)] = {
                "serial": serial, "status": "calib_failed", "num_views": len(accepted), **source_info}
            if joint_cam is not None:
                joint_notes[serial] = f"합친 보정 실패: calib_failed ({len(accepted)}장)"
            continue

        K, D = result["K"], result["D"]
        print(f"[cam{cam_idx}] RMS reproj = {result['rms']:.4f} px  "
              f"(used {result['n_used']}/{result['n_total']}, dropped {result['n_dropped']})")
        print(f"           factory fx,fy,cx,cy = "
              f"{factory_K[0,0]:.2f},{factory_K[1,1]:.2f},{factory_K[0,2]:.2f},{factory_K[1,2]:.2f}")
        print(f"           charuco fx,fy,cx,cy = "
              f"{K[0,0]:.2f},{K[1,1]:.2f},{K[0,2]:.2f},{K[1,2]:.2f}")
        print(f"           charuco D = {np.asarray(D).flatten()}")

        if joint_cam is not None:
            _write_joint_camera(
                report, joint_report, joint, joint_cam, cam_idx, serial, is_gripper,
                npz_path, backup_dir, K, D, result, n_base, factory_K, factory_D, source_info)
            continue

        backup_path = overwrite_color_intrinsics(
            npz_path, backup_dir, K, D, result, serial)
        print(f"[SAVE] {npz_path} (color_K/color_D 교체)  factory backup -> {backup_path}")

        report["cameras"][str(cam_idx)] = {
            "serial": serial, "is_gripper": is_gripper, "status": "written",
            "rms_px": result["rms"], "num_views_used": result["n_used"],
            "num_views_total": result["n_total"], "num_dropped": result["n_dropped"],
            "K": np.asarray(K).tolist(), "D": np.asarray(D).flatten().tolist(),
            "factory_K": factory_K.tolist(),
            "factory_D": factory_D.flatten().tolist(),
            **source_info,
        }

    if not args.from_images:
        cv2.destroyAllWindows()

    if args.capture_only:
        report_path = images.directory / "capture_report.json"
        if report_path.exists():
            # 나눠 찍은 경우 이전 촬영의 카메라 기록을 유지한다.
            previous = json.loads(report_path.read_text(encoding="utf-8"))
            report["cameras"] = {**previous.get("cameras", {}), **report["cameras"]}
        with report_path.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
        for cam_idx, _ in idx_pairs:
            print(f"[RAW] cam{cam_idx}: {images.count(cam_idx)}장")
        print(f"[SAVE] {images.manifest_path}")
        print(f"[DONE] RAW 촬영 {'중단' if aborted else '완료'}. 저장 이미지 유지. "
              "나중에 --from_images와 보드 옵션으로 intrinsic을 계산하세요.")
        return

    if not aborted:
        outputs = [(intr_dir, report)]
        if joint is not None:
            _carry_over_joint_entries(joint, joint_report, joint_notes,
                                      {serial for _, serial in idx_pairs})
            outputs.append((joint["intr_dir"], joint_report))
        for out_dir, out_report in outputs:
            report_path = os.path.join(out_dir, "charuco_intrinsics_report.json")
            with open(report_path, "w") as f:
                json.dump(out_report, f, indent=2)
            print(f"\n[SAVE] {report_path}")

        for out_dir, out_report in outputs:
            print(f"\n=== 요약: {out_dir} ===")
            for k, v in sorted(out_report["cameras"].items(), key=lambda item: int(item[0])):
                st = v.get("status")
                tail = (f"  (이전 실행 {v['previous_calibrated_at']} 기록 유지)"
                        if "previous_calibrated_at" in v else "")
                if v.get("joint_unavailable"):
                    tail += f"  [합치지 못함: {v['joint_unavailable']}]"
                if st == "written":
                    rms = v.get("rms_px")
                    rms = "n/a" if rms is None else f"{rms:.3f}px"
                    print(f"  cam{k}: RMS {rms}  views {v.get('num_views_used', '?')}  -> written{tail}")
                else:
                    print(f"  cam{k}: {st}{tail}")
        print("[DONE] 02_calibrate_intrinsics.py complete. "
              "이제 03~05는 갱신된 color_K/color_D 를 그대로 사용합니다.")
    else:
        print("[DONE] 중단됨.")


if __name__ == "__main__":
    main()
