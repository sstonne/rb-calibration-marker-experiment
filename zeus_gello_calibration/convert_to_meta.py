#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zeus_gello_calibration/convert_to_meta.py -- Zeus legacy 촬영(session1/2/3,
폴더별 cam_<label>.png + robot.json)을 04_filter_observations -> 05_calibrate ->
06_make_report 가 읽는 `data/session<NN>_<설명>_<MMDD>/calib_train/meta.json`
형식으로 변환한다. ur3_calibration/convert_to_meta.py 의 Zeus 판.

앞으로 모든 캘리브레이션은 01~06 파이프라인으로만 돌린다. 새 촬영은
03_capture.py 가 이 형식으로 바로 저장하고, 옛 데이터는 이 변환기를 한 번
거친다 -- 그래야 옛 결과와 새 결과가 같은 러너(05)로 비교된다.

세 세션을 **하나의 meta.json** 으로 합친다 (한 이벤트 = 한 로봇 자세, 카메라 4대):
  session1 (쥔 큐브, 고정캠)      -> cube_gripped=True,  grasp_id=0, set_index=null, block B_eyetohand
  session2 (놓인 큐브, 촬영 자세)  -> cube_gripped=False, set_index=1..N,   block A_placement
                                    + set_cube_center_6dof (A3 raw-FK용, 아래 참고)
                                    (기본: 고정캠 이벤트 + 그리퍼캠 이벤트 둘로 분리, 아래 참고)
  session3 (손목 보드)             -> cube_gripped=False, set_index=null,   block A_placement
set_index=null 인 이벤트는 placement 세트가 아니라 05 에서 train 전용 보조 관측
(grasp+FK 큐브, 정지 보드)으로만 쓰인다.
보드는 session2 사진에도 그대로 들어 있으므로 04 가 자동으로 검출한다.

A3(raw-FK hard fixed)용 set_cube_center: 05 는 "큐브 중심을 가리키는 로봇 FK
pose"를 세트마다 요구하고 거기에 RAW_FK_CUBE_CENTER_TO_OBJECT(축 뒤집기, 이동
0)를 곱해 object 프레임으로 만든다. Zeus 에는 기계적으로 티칭한 값이 없어서
**place 명령 flange pose @ 플랜지 z축 방향 nominal 160mm** 로 만든다
(flange->큐브 top datum 97.5 + 큐브 원점->top 62.5 = 160mm, 조립 도면 nominal,
비전 미사용). 이 값은 --nominal-flange-to-cube-center-mm 로 바꿀 수 있다.

카메라 index: LOCAL_CAM_IDS 순서(0:039422061216, 1:fixed2, 2:fixed3, 3:gripper)로
세션 폴더 안에 intrinsics/cam<N>.npz 를 같이 써 준다(04/05 가 index 로 읽으므로).

사용법:
  python zeus_gello_calibration/convert_to_meta.py
  python zeus_gello_calibration/convert_to_meta.py --out-root data/session11_zeus_handheld_floor_wrist_meta_0909
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "zeus_gello_calibration"))

from calibration_pipeline.board_config import charuco_config_from_dict  # noqa: E402
from calibration_pipeline.charuco import CharucoTarget  # noqa: E402
from calibration_pipeline.config import get_default_cube_config, get_default_cube_config_source  # noqa: E402
from calibration_pipeline.cube_config import cube_config_to_dict  # noqa: E402
from robot.backends.zeus_client import T_to_pose6, pose6_to_T  # noqa: E402

from fit_full_calibration import CHARUCO_BOARD_CONFIG  # noqa: E402
from fit_grasp_offset import LOCAL_CAM_IDS, load_intrinsics_by_label  # noqa: E402
from paths import SESSION1_DIR, SESSION2_DIR, SESSION3_DIR  # noqa: E402
from session2_pick_and_place import compute_ordered_targets  # noqa: E402

GRIPPER_CAM_IDX = LOCAL_CAM_IDS["gripper"]
NOMINAL_FLANGE_TO_CUBE_CENTER_MM = 160.0
OUT_ROOT_DEFAULT = REPO_ROOT / "data" / "session11_zeus_handheld_floor_wrist_meta_0909"


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)          # 같은 파일시스템이면 하드링크 (용량 0)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def build_cams(capture_dir: Path, calib_train_dir: Path, event_id: int, charuco: CharucoTarget,
               only_labels=None) -> dict:
    cams = {}
    for label, idx in LOCAL_CAM_IDS.items():
        src = capture_dir / f"cam_{label}.png"
        if not src.is_file() or (only_labels is not None and label not in only_labels):
            cams[str(idx)] = {"saved": False}
            continue
        rel = f"cam{idx}/rgb_{event_id:05d}.png"
        link_or_copy(src, calib_train_dir / rel)
        img = cv2.imread(str(src))
        n = 0
        if img is not None:
            _c, _i, n, _mc, _mi = charuco.detect(img)
        cams[str(idx)] = {"saved": True, "rgb_path": rel, "charuco_detect_n": int(n or 0)}
    return cams


def load_pose6(capture_dir: Path):
    rj = json.loads((capture_dir / "robot.json").read_text())
    return [float(v) for v in rj["pose"]], rj.get("joints")


def build_capture(*, event_id, capture_index, set_index, cube_gripped, grasp_id, capture_block,
                  capture_dir, calib_train_dir, charuco, session_tag, set_cube_center_6dof=None,
                  only_labels=None):
    pose6, joints = load_pose6(capture_dir)
    T = pose6_to_T(pose6)
    cap = {
        "event_id": int(event_id),
        "capture_index": int(capture_index),
        "capture_gate": {"capture_block": capture_block},
        "capture_block": capture_block,
        "set_index": (None if set_index is None else int(set_index)),   # null = placement 세트가 아닌 이벤트(쥔 큐브, 손목 보드)
        "cube_gripped": bool(cube_gripped),
        "grasp_id": (None if grasp_id is None else int(grasp_id)),
        "robot_pose_6dof": pose6,                       # [x,y,z mm, rz,ry,rx deg], Zeus i611 extrinsic ZYX
        "robot_pose_matrix_4x4": np.asarray(T, dtype=float).tolist(),   # metres
        "robot_joints_deg": joints,
        "cams": build_cams(capture_dir, calib_train_dir, event_id, charuco, only_labels),
        "source_capture_dir": str(capture_dir),
        "source_session": session_tag,
    }
    if set_cube_center_6dof is not None:
        cap["set_cube_center_6dof"] = [float(v) for v in set_cube_center_6dof]
    return cap


def write_intrinsics(session_root: Path, zeus_dir: Path, ur3_dir: Path, device_map: Path,
                     color_w: int, color_h: int) -> Path:
    K_map, D_map = load_intrinsics_by_label(zeus_dir, ur3_dir, device_map)
    out = session_root / "intrinsics"
    out.mkdir(parents=True, exist_ok=True)
    label_by_idx = {v: k for k, v in LOCAL_CAM_IDS.items()}
    for idx in sorted(label_by_idx):
        np.savez(out / f"cam{idx}.npz", color_K=np.asarray(K_map[idx], float), color_D=np.asarray(D_map[idx], float),
                 depth_scale_m_per_unit=np.float64(0.001), label=label_by_idx[idx], color_w=color_w, color_h=color_h)
    (out / "device_map.json").write_text(json.dumps({
        "note": "Zeus rig, index order = zeus_gello_calibration.fit_grasp_offset.LOCAL_CAM_IDS; "
                "cam0 (039422061216) intrinsics come from the UR3 rig's charuco calibration of the same unit",
        "label_to_idx": dict(LOCAL_CAM_IDS), "gripper_cam_idx": GRIPPER_CAM_IDX,
    }, indent=2))
    return out


def detect_capture_resolution(session1_dir: Path, capture_subdir: str) -> tuple[int, int]:
    """실제로 찍힌 사진 한 장의 크기를 재서 (width, height)를 돌려준다 -- intrinsics
    메타데이터에 하드코딩된 해상도를 안 믿고, 진짜 촬영 해상도를 그대로 쓰기 위함."""
    root = session1_dir / capture_subdir
    for cap_dir in sorted((p for p in root.iterdir() if p.is_dir() and p.name.isdigit()),
                          key=lambda p: int(p.name)):
        for png in sorted(cap_dir.glob("cam_*.png")):
            img = cv2.imread(str(png))
            if img is not None:
                h, w = img.shape[:2]
                return int(w), int(h)
    raise RuntimeError(f"{root}에서 해상도를 잴 사진을 하나도 못 찾았습니다.")


def convert(args):
    session_root = Path(args.out_root)
    calib_train_dir = session_root / "calib_train"
    calib_train_dir.mkdir(parents=True, exist_ok=True)
    charuco = CharucoTarget(charuco_config_from_dict(CHARUCO_BOARD_CONFIG))
    color_w, color_h = detect_capture_resolution(SESSION1_DIR, args.session1_capture_subdir)
    print(f"[해상도] 실제 촬영 사진 기준: {color_w}x{color_h}")
    intrinsics_dir = write_intrinsics(session_root, Path(args.zeus_intrinsics_dir), Path(args.ur3_intrinsics_dir),
                                      Path(args.device_map), color_w, color_h)

    meta = {
        "root_folder": str(calib_train_dir.resolve()),
        "session_allocation": None,
        "rig": "zeus_gello",
        "gripper_cam_idx": GRIPPER_CAM_IDX,
        "n_fixed_cams": 3, "n_gripper_cams": 1, "cam_indices": [0, 1, 2, 3],
        "cam_labels": {str(v): k for k, v in LOCAL_CAM_IDS.items()},
        "charuco_board_config_source": "zeus_gello_calibration.fit_full_calibration.CHARUCO_BOARD_CONFIG",
        "charuco_board_config": dict(CHARUCO_BOARD_CONFIG),
        "cube_config_source": get_default_cube_config_source(),
        # 05가 manifest(04)의 동결 cube config와 여기 값을 비교한다 -- 없으면 실패
        "cube_config": cube_config_to_dict(get_default_cube_config()),
        "pose_convention_note": "robot_pose_6dof = Zeus i611 [x,y,z mm, rz,ry,rx deg] extrinsic ZYX; matrix in metres",
        "set_cube_center_source": (
            f"place command flange pose @ nominal +z {args.nominal_flange_to_cube_center_mm} mm "
            "(flange->cube top datum 97.5 + cube origin->top 62.5; no vision)"),
        "capture_config": {
            "schema_version": "capture_config_v1",
            "charuco_board_config": dict(CHARUCO_BOARD_CONFIG),
            "intrinsics_dir": str(intrinsics_dir.resolve()),
            "width": color_w, "height": color_h, "fps": 15, "save_depth": False,
            "capture_gate": {"schema_version": "capture_gate_profiles_v1",
                             "profiles": {"A_placement": {"expected_cube_gripped": False},
                                          "B_eyetohand": {"expected_cube_gripped": True}}},
        },
        "source_sessions": {"session1": str(SESSION1_DIR), "session2": str(SESSION2_DIR), "session3": str(SESSION3_DIR)},
    }
    captures = []
    event_id = 0

    # session1: 쥔 큐브
    s1 = SESSION1_DIR / args.session1_capture_subdir
    for idx in sorted(int(p.name) for p in s1.iterdir() if p.is_dir() and p.name.isdigit()):
        captures.append(build_capture(event_id=event_id, capture_index=idx, set_index=None, cube_gripped=True, grasp_id=0,
                                      capture_block="B_eyetohand", capture_dir=s1 / f"{idx:03d}",
                                      calib_train_dir=calib_train_dir, charuco=charuco, session_tag="session1"))
        event_id += 1
    n1 = event_id

    # session2: 놓인 큐브 (촬영 자세에서 4대), set_cube_center = place FK @ nominal
    s2 = SESSION2_DIR / args.session2_capture_subdir
    items = compute_ordered_targets(SESSION2_DIR)
    T_nom = np.eye(4); T_nom[2, 3] = args.nominal_flange_to_cube_center_mm / 1000.0
    for idx in sorted(int(p.name) for p in s2.iterdir() if p.is_dir() and p.name.isdigit()):
        if idx >= len(items):
            continue
        T_place = pose6_to_T(items[idx]["target"])
        center6 = [float(v) for v in T_to_pose6(T_place @ T_nom)]
        # 05의 event-stratified split은 세트 안에서 이벤트를 train/test로 나눈다.
        # Zeus session2는 세트당 촬영 이벤트가 1개(한 자세에서 4대 동시)라 그대로
        # 넣으면 그 하나가 test로 가고 train이 비어 실패한다. 그래서 기본값은
        # 같은 로봇 자세·같은 순간의 사진을 "고정캠 3대 이벤트"와 "그리퍼캠
        # 이벤트" 둘로 나눠 넣는다 -- 물리적으로 같은 촬영이고, held-out은
        # "train 이벤트로 잡은 큐브 pose를 test 이벤트 카메라가 맞추는가"가 된다.
        groups = ([("fixed", ("039422061216", "fixed2", "fixed3")), ("gripper", ("gripper",))]
                  if args.placement_event_mode == "fixed_gripper_split" else [("all", None)])
        for tag, labels in groups:
            captures.append(build_capture(event_id=event_id, capture_index=idx, set_index=idx + 1, cube_gripped=False, grasp_id=None,
                                          capture_block="A_placement", capture_dir=s2 / f"{idx:03d}",
                                          calib_train_dir=calib_train_dir, charuco=charuco, session_tag="session2",
                                          set_cube_center_6dof=center6, only_labels=labels))
            captures[-1]["place_command_pose_6dof"] = [float(v) for v in items[idx]["target"]]
            captures[-1]["placement_event_group"] = tag
            event_id += 1
    n2 = event_id - n1

    # session3: 손목 보드
    s3 = SESSION3_DIR / args.session3_capture_subdir
    for idx in sorted(int(p.name) for p in s3.iterdir() if p.is_dir() and p.name.isdigit()):
        captures.append(build_capture(event_id=event_id, capture_index=idx, set_index=None, cube_gripped=False, grasp_id=None,
                                      capture_block="A_placement", capture_dir=s3 / f"{idx:03d}",
                                      calib_train_dir=calib_train_dir, charuco=charuco, session_tag="session3"))
        event_id += 1
    n3 = event_id - n1 - n2

    meta["captures"] = captures
    meta_path = calib_train_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    n_sets = len({c["set_index"] for c in captures if c["source_session"] == "session2"})
    print(f"wrote {meta_path}\n  {len(captures)} events: session1 {n1} (gripped) + session2 {n2} events / {n_sets} placement sets "
          f"({args.placement_event_mode}) + session3 {n3} (wrist board)")
    print(f"  intrinsics -> {intrinsics_dir}")
    return meta_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default=str(OUT_ROOT_DEFAULT))
    ap.add_argument("--session1-capture-subdir", default="capture_replayed")
    ap.add_argument("--session2-capture-subdir", default="capture_placed")
    ap.add_argument("--session3-capture-subdir", default="capture_replayed")
    ap.add_argument("--zeus-intrinsics-dir", default=str(REPO_ROOT / "intrinsics"))
    ap.add_argument("--ur3-intrinsics-dir", default=str(REPO_ROOT / "ur3_calibration" / "intrinsics"))
    ap.add_argument("--device-map", default=str(REPO_ROOT / "intrinsics" / "device_map.json"))
    ap.add_argument("--nominal-flange-to-cube-center-mm", type=float, default=NOMINAL_FLANGE_TO_CUBE_CENTER_MM)
    ap.add_argument("--placement-event-mode", choices=("fixed_gripper_split", "single"), default="fixed_gripper_split",
                    help="session2 placement 한 촬영을 고정캠 이벤트 + 그리퍼캠 이벤트 둘로 나눔(기본; 05 split 요건) / single = 이벤트 1개")
    args = ap.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
