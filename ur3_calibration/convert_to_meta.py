#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ur3_calibration/convert_to_meta.py -- UR3 raw capture -> calibration_pipeline meta.json

Converts the raw captures under ur3_calibration/data/session{1,2,3}_*_<MMDD>/capture/
into the calibration_pipeline's expected data/<session>/calib_train/{meta.json,
camN/*.png} layout, so that 04_filter_observations.py / 05_calibrate.py
(calibration_pipeline.table1) can run against them unmodified.

Three raw sessions, two output meta.json files:

  data/session10_ur3_handheld_floor_meta_0909/calib_train/meta.json
      session1 (cube gripped throughout, fixed cams watch it move) +
      session2 (cube placed ungripped at 12 floor spots, fixed cams + the
      parked gripper cam watch each spot) combined into one dataset so
      table1.py's unified (U) rows can jointly fit T_base_Ci, T_gripper_cam,
      and T_gripper_cube[grasp 0] from session1 while session2 supplies the
      cube-placement (T_base_cube_by_set) observations.

      event_id: 0..14 = session1 captures 000..014 (in original order),
                15..26 = session2 captures 000..011 (in original order,
                which is already the archived robot execution order: capture
                indices follow delta-yaw ascending order, not poses.json's
                original per-pose index).
      set_index: session1 -> 0 (single set, all 15 events share it);
                 session2 -> 1..12, one per floor placement (= capture
                 index + 1, so capture 000 -> set 1 ... capture 011 -> set 12).
      grasp_id: 0 for every session1 (gripped) event, null otherwise.
      capture_block: "B_eyetohand" for session1 (matches Zeus's convention
                 for gripped/hand-eye captures), "A_placement" for session2.

  data/session09_ur3_wrist_meta_0909/calib_train/meta.json
      session3 (gripper empty, only the wrist moves, watching the stationary
      floor ChArUco board) -- board-only, cube_gripped=False throughout,
      no cube observations at all. See the module docstring notes in the
      report about why this cannot go through table1.py's normal per-set
      cube-eligibility gate.

Pose conversion (UR3, NOT Zeus's mm+ZYX-Euler-degree convention): robot.json's
tcp_pose is [x, y, z (meters), rx, ry, rz (axis-angle rotation vector,
radians)] (ur_rtde convention). T_base_gripper = 4x4 with R =
Rotation.from_rotvec([rx,ry,rz]).as_matrix(), translation = [x,y,z] directly
(already meters, unlike Zeus's mm robot_pose_6dof).

Camera mapping (see fix_gripper_camera_metadata.py for the derivation):
  cam0 = fixed1 (serial 039422061216, D415)
  cam1 = gripper (serial 136622073980, D435)
  cam2 = fixed2 (serial 243622070663, D435I)
  cam3 = fixed3 (serial 314522062542, D415)
Raw capture files are already named by label (cam_fixed1.png, cam_gripper.png,
cam_fixed2.png, cam_fixed3.png), independent of this script.

set_cube_center_6dof (needed for A3/A4 FK rows) is populated only in "pass 2"
via --set-cube-center-json, a {set_index(str): [x_mm,y_mm,z_mm,rz_deg,ry_deg,
rx_deg]} mapping produced by fit_grasp_offset.py from Pass 1's fitted
T_gripper_cube. See that script's docstring for why the stored value is
T_base_cube_taught[s] @ RAW_FK_CUBE_CENTER_TO_OBJECT (not T_base_cube_taught[s]
directly): calibration_pipeline/table1.py unconditionally reapplies that same
frozen mechanical_frame_map when it decodes set_cube_center_6dof back into
T_base_fk_raw for A3/A4, and that map is its own inverse, so pre-composing it
once here is what makes the round trip land back on the correct T_base_cube.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration_pipeline.board_config import (  # noqa: E402
    charuco_config_to_dict, resolve_charuco_config,
)
from calibration_pipeline.charuco import CharucoTarget  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
UR3_DATA_ROOT = Path(__file__).resolve().parent / "data"

# cam index -> raw capture file label (see fix_gripper_camera_metadata.py)
CAM_LABELS = {0: "fixed1", 1: "gripper", 2: "fixed2", 3: "fixed3"}
GRIPPER_CAM_IDX = 1

# 이 세션들이 촬영한 물리 보드. 정의는 targets/charuco_boards/ 의 JSON 한 곳에만
# 있고, 여기서는 이름으로 불러온다 (보드가 바뀌면 JSON 을 추가하고 이름만 교체).
# session09/10 의 meta.json 이 이 값으로 frozen 되어 있으므로 이름 변경 시 주의.
CHARUCO_BOARD_NAME = "11x7_id5"
CHARUCO_BOARD_CFG, CHARUCO_BOARD_SOURCE = resolve_charuco_config(CHARUCO_BOARD_NAME)
CHARUCO_BOARD_CONFIG = charuco_config_to_dict(CHARUCO_BOARD_CFG)

# Descriptive only (calibration_pipeline.runtime.resolve_cube_config_for_run
# always uses config.py's get_default_cube_config() unless an explicit JSON
# override is passed -- this dict is never actually read back).
CUBE_CONFIG_SOURCE = "config_py:CubeConfig"

# calibration_pipeline.observations.load_board_pixel_observations (the board
# loader table1.py actually calls) hard-gates on the PRECOMPUTED
# cams.<idx>.charuco_detect_n field (>= 4) before it will even try to detect
# a board in the image -- unlike the cube path, this one precomputed field is
# NOT optional. It is computed here for real with the same CharucoTarget the
# pipeline itself uses, not guessed or left at a placeholder.
_CHARUCO_TARGET = CharucoTarget(CHARUCO_BOARD_CFG)


def compute_charuco_detect_n(image_path: Path) -> int:
    image = cv2.imread(str(image_path))
    if image is None:
        return 0
    _corners, _ids, n_corners, _marker_corners, _marker_ids = _CHARUCO_TARGET.detect(image)
    return int(n_corners or 0)


def tcp_pose_to_matrix(tcp_pose: List[float]) -> np.ndarray:
    """UR3 tcp_pose [x,y,z metres, rx,ry,rz rotvec radians] -> 4x4 T_base_gripper."""
    x, y, z, rx, ry, rz = [float(v) for v in tcp_pose]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_rotvec([rx, ry, rz]).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def matrix_to_pose6_zyx_deg(T: np.ndarray) -> List[float]:
    """Inverse of capture_pipeline.robot.euler_deg_to_matrix.

    Verified by round-trip test (see conversation record): for R built as
    Rz(rz) @ Ry(ry) @ Rx(rx), scipy's Rotation.from_matrix(R).as_euler(
    'ZYX', degrees=True) returns exactly (rz, ry, rx) in euler_deg_to_matrix's
    own (x_mm,y_mm,z_mm,rz_deg,ry_deg,rx_deg) argument order.
    """
    T = np.asarray(T, dtype=np.float64)
    x_mm, y_mm, z_mm = (T[:3, 3] * 1000.0).tolist()
    rz, ry, rx = Rotation.from_matrix(T[:3, :3]).as_euler("ZYX", degrees=True)
    return [float(x_mm), float(y_mm), float(z_mm), float(rz), float(ry), float(rx)]


def load_robot_pose(capture_dir: Path) -> List[float]:
    robot = json.loads((capture_dir / "robot.json").read_text())
    return list(robot["tcp_pose"])


def copy_camera_images(capture_dir: Path, calib_train_dir: Path, event_index: int) -> Dict[str, dict]:
    """Copy this capture's 4 color images into camN/rgb_<event_index:05d>.png.

    Depth PNGs are not copied: calibration_pipeline's cube/board corner
    detection and evaluation never read depth_path (confirmed by grep -- only
    runtime.py references the key, purely descriptively), so copying them
    would only cost disk space for this task.
    """
    cams = {}
    for cam_idx, label in CAM_LABELS.items():
        src = capture_dir / f"cam_{label}.png"
        if not src.is_file():
            raise FileNotFoundError(f"missing {src}")
        dest_dir = calib_train_dir / f"cam{cam_idx}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        rel_path = f"cam{cam_idx}/rgb_{event_index:05d}.png"
        dest = calib_train_dir / rel_path
        if not dest.exists():
            shutil.copyfile(src, dest)
        cams[str(cam_idx)] = {
            "saved": True,
            "is_gripper": bool(cam_idx == GRIPPER_CAM_IDX),
            "rgb_path": rel_path,
            "charuco_detect_n": compute_charuco_detect_n(src),
        }
    return cams


def build_capture(
    *,
    event_id: int,
    capture_index: int,
    set_index: int,
    cube_gripped: bool,
    grasp_id: Optional[int],
    capture_block: str,
    capture_dir: Path,
    calib_train_dir: Path,
    set_cube_center_6dof: Optional[List[float]] = None,
    canonical_set_cube_center_matrix_4x4: Optional[List[List[float]]] = None,
) -> dict:
    tcp_pose = load_robot_pose(capture_dir)
    T_base_gripper = tcp_pose_to_matrix(tcp_pose)
    cams = copy_camera_images(capture_dir, calib_train_dir, event_id)
    capture = {
        "event_id": int(event_id),
        "capture_index": int(capture_index),
        "capture_gate": {"capture_block": capture_block},
        "capture_block": capture_block,
        "set_index": int(set_index),
        "cube_gripped": bool(cube_gripped),
        "grasp_id": (None if grasp_id is None else int(grasp_id)),
        "robot_pose_matrix_4x4": T_base_gripper.tolist(),
        "cams": cams,
        "source_capture_dir": str(capture_dir),
    }
    if set_cube_center_6dof is not None:
        capture["set_cube_center_6dof"] = [float(v) for v in set_cube_center_6dof]
    if canonical_set_cube_center_matrix_4x4 is not None:
        capture["canonical_set_cube_center_matrix_4x4"] = [
            [float(v) for v in row] for row in canonical_set_cube_center_matrix_4x4
        ]
    return capture


def meta_header(calib_train_dir: Path) -> dict:
    return {
        "root_folder": str(calib_train_dir.resolve()),
        "session_allocation": None,
        "gripper_cam_idx": GRIPPER_CAM_IDX,
        "n_fixed_cams": 3,
        "n_gripper_cams": 1,
        "cam_indices": [0, 1, 2, 3],
        "charuco_board_config_source": CHARUCO_BOARD_SOURCE,
        "charuco_board_config": dict(CHARUCO_BOARD_CONFIG),
        "cube_config_source": CUBE_CONFIG_SOURCE,
        "capture_config": {
            "schema_version": "capture_config_v1",
            "charuco_board_config": dict(CHARUCO_BOARD_CONFIG),
            "intrinsics_dir": str((Path(__file__).resolve().parent / "intrinsics")),
            "width": 1280,
            "height": 720,
            "fps": 15,
            "save_depth": False,
            "capture_gate": {
                "schema_version": "capture_gate_profiles_v1",
                "profiles": {
                    "A_placement": {"expected_cube_gripped": False},
                    "B_eyetohand": {"expected_cube_gripped": True},
                },
            },
        },
    }


def convert_session12(
    out_root: Path,
    set_cube_center_by_set: Optional[Dict[str, List[float]]] = None,
    set_cube_center_matrix_by_set: Optional[Dict[str, List[List[float]]]] = None,
) -> Path:
    calib_train_dir = out_root / "calib_train"
    calib_train_dir.mkdir(parents=True, exist_ok=True)
    meta = meta_header(calib_train_dir)
    captures = []

    session1_dir = UR3_DATA_ROOT / "session1_handheld_fixed_cam_0909" / "capture"
    session1_indices = sorted(int(p.name) for p in session1_dir.iterdir() if p.is_dir())
    for event_id, idx in enumerate(session1_indices):
        capture_dir = session1_dir / f"{idx:03d}"
        captures.append(build_capture(
            event_id=event_id,
            capture_index=idx,
            set_index=0,
            cube_gripped=True,
            grasp_id=0,
            capture_block="B_eyetohand",
            capture_dir=capture_dir,
            calib_train_dir=calib_train_dir,
        ))

    session2_dir = UR3_DATA_ROOT / "session2_floor_board_dual_cam_0909" / "capture"
    session2_indices = sorted(int(p.name) for p in session2_dir.iterdir() if p.is_dir())
    event_offset = len(session1_indices)
    for order, idx in enumerate(session2_indices):
        capture_dir = session2_dir / f"{idx:03d}"
        set_index = idx + 1  # capture 000 -> set 1 ... capture 011 -> set 12
        set_key = str(set_index)
        set_cube_center_6dof = (
            set_cube_center_by_set.get(set_key) if set_cube_center_by_set else None)
        set_cube_center_matrix = (
            set_cube_center_matrix_by_set.get(set_key) if set_cube_center_matrix_by_set else None)
        captures.append(build_capture(
            event_id=event_offset + order,
            capture_index=idx,
            set_index=set_index,
            cube_gripped=False,
            grasp_id=None,
            capture_block="A_placement",
            capture_dir=capture_dir,
            calib_train_dir=calib_train_dir,
            set_cube_center_6dof=set_cube_center_6dof,
            canonical_set_cube_center_matrix_4x4=set_cube_center_matrix,
        ))

    meta["captures"] = captures
    meta_path = calib_train_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"wrote {meta_path}  ({len(captures)} captures: "
          f"{len(session1_indices)} session1 + {len(session2_indices)} session2)")
    return meta_path


def convert_session3(out_root: Path) -> Path:
    calib_train_dir = out_root / "calib_train"
    calib_train_dir.mkdir(parents=True, exist_ok=True)
    meta = meta_header(calib_train_dir)
    captures = []

    session3_dir = UR3_DATA_ROOT / "session3_wrist_motion_gripper_cam_0909" / "capture"
    session3_indices = sorted(int(p.name) for p in session3_dir.iterdir() if p.is_dir())
    for event_id, idx in enumerate(session3_indices):
        capture_dir = session3_dir / f"{idx:03d}"
        captures.append(build_capture(
            event_id=event_id,
            capture_index=idx,
            set_index=0,
            cube_gripped=False,
            grasp_id=None,
            capture_block="A_placement",
            capture_dir=capture_dir,
            calib_train_dir=calib_train_dir,
        ))

    meta["captures"] = captures
    meta_path = calib_train_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"wrote {meta_path}  ({len(captures)} captures, board-only, session3)")
    return meta_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root-12", default=str(REPO_ROOT / "data" / "session10_ur3_handheld_floor_meta_0909"))
    ap.add_argument("--out-root-3", default=str(REPO_ROOT / "data" / "session09_ur3_wrist_meta_0909"))
    ap.add_argument("--set-cube-center-json", default=None,
                    help=("Pass 2 only: JSON produced by fit_grasp_offset.py mapping "
                          "{set_index(str): {'pose6': [...], 'matrix4x4': [[...]]}}."))
    ap.add_argument("--skip-session3", action="store_true")
    args = ap.parse_args()

    set_cube_center_by_set = None
    set_cube_center_matrix_by_set = None
    if args.set_cube_center_json:
        payload = json.loads(Path(args.set_cube_center_json).read_text())
        set_cube_center_by_set = {k: v["pose6"] for k, v in payload.items()}
        set_cube_center_matrix_by_set = {k: v["matrix4x4"] for k, v in payload.items()}

    convert_session12(
        Path(args.out_root_12),
        set_cube_center_by_set=set_cube_center_by_set,
        set_cube_center_matrix_by_set=set_cube_center_matrix_by_set,
    )
    if not args.skip_session3:
        convert_session3(Path(args.out_root_3))


if __name__ == "__main__":
    main()
