#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zeus_gello_calibration/fit_full_calibration.py -- "통합" 캘리브레이션: 4대
카메라(고정 3 + 그리퍼 1) 전부를 하나의 조인트 최적화에 묶는다.

지금까지의 fit_grasp_offset.py / fit_placement_fk_ablation.py는 고정 카메라
3대만 썼다 -- 그리퍼 캠은 session1/session2 어디서도 큐브를 못 봐서(0/16,
0/15 검출) 낄 자리가 없었다. 그리퍼 캠을 넣으려면 그게 실제로 뭔가 보는
데이터가 있어야 하는데, 그게 session3(손목만 움직이며 바닥 마커보드를
그리퍼캠으로 촬영, eye-in-hand)다. UR3 쪽 charuco 보드 설정
(squares 11x7, square 25mm, marker 18mm, DICT_4X4_250, id_start=5)이
session3 그리퍼캠 이미지에서 그대로 검출된다(실측 확인, 같은 물리 보드를
재사용 중인 것으로 보임) -- 그래서 UR3 쪽 정의를 그대로 가져다 쓴다.

세 세션을 로봇의 같은 FK 체인으로 묶어서 하나로 푼다:
  session1 (16, 고정캠 3대, 큐브 쥔 채)   -> grasp+FK:  T_base_cube = FK(q)@T_gripper_cube
  session2 (15, 고정캠 3대, 큐브 바닥 배치) -> 조건별로 다름 (아래)
  session3 (15, 그리퍼캠, 바닥 마커보드)    -> eye-in-hand: T_base_cam = FK(q)@T_gripper_cam,
                                              보드 pose는 T_base_board 하나(전체 공유)

  no_fk    -- session2의 15개 placement 각각 이미지만으로 자유 pose(T_base_cube_by_set)
  fixed_fk -- session2도 grasp+FK로: 그 placement에서 실제로 명령했던 place 목표
              pose를 FK인 것처럼 넣어 T_gripper_cube(session1과 공유)로 계산

두 조건 모두 T_base_Ci(고정캠 3대) + T_gripper_cam + T_base_board +
T_gripper_cube_by_grasp는 전부 자유 변수로 같이 푼다 (session1+2+3 전체
관측치로 통합 최적화). fit_placement_fk_ablation.py의 fixed cam-only 버전을
그대로 확장한 것.

사용법:
  python fit_full_calibration.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from calibration_pipeline.apriltag_cube import AprilTagCubeTarget  # noqa: E402
from calibration_pipeline.board_config import (  # noqa: E402
    charuco_config_to_dict, resolve_charuco_config,
)
from calibration_pipeline.charuco import CharucoTarget  # noqa: E402
from calibration_pipeline.config import get_default_cube_config  # noqa: E402
from calibration_pipeline.observations import (  # noqa: E402
    load_board_pixel_observations, load_cube_pixel_observations,
)
from calibration_pipeline.reprojection import (  # noqa: E402
    PoseState, SolverOptions, solve_corner_reprojection, variable_keys,
)
from calibration_pipeline.table1 import estimate_board_handeye_initial  # noqa: E402
from robot.backends.zeus_client import pose6_to_T  # noqa: E402

from fit_grasp_offset import (  # noqa: E402
    LOCAL_CAM_IDS, build_synthetic_meta, load_intrinsics_by_label, load_robot_T,
)
from fit_placement_fk_ablation import (  # noqa: E402
    SESSION2_EVENT_OFFSET, build_synthetic_meta_placed, init_cube_poses_from_images,
)
from session2_pick_and_place import (  # noqa: E402
    SESSION2_DIR_DEFAULT, compute_ordered_targets,
)

SESSION1_DIR_DEFAULT = REPO_ROOT / "zeus_gello_calibration" / "data" / "session1_handheld_fixed_cam_0909"
SESSION3_DIR_DEFAULT = REPO_ROOT / "zeus_gello_calibration" / "data" / "session3_wrist_motion_gripper_cam_0909"
FIT_JSON_DEFAULT = REPO_ROOT / "zeus_gello_calibration" / "pass1_grasp_offset_replayed.json"
GRIPPER_LOCAL_ID = LOCAL_CAM_IDS["gripper"]  # 3
SESSION3_EVENT_OFFSET = 2000  # session1(0..15)/session2(1000..1014)와 안 겹치게

# UR3 세션과 같은 물리 보드 (targets/charuco_boards/board_11x7_id5.json) -- 실측으로
# session3 그리퍼캠 이미지에서 그대로 검출됨을 확인함.
CHARUCO_BOARD_NAME = "11x7_id5"
CHARUCO_BOARD_CFG, CHARUCO_BOARD_SOURCE = resolve_charuco_config(CHARUCO_BOARD_NAME)
CHARUCO_BOARD_CONFIG = charuco_config_to_dict(CHARUCO_BOARD_CFG)


def build_synthetic_meta_board(session3_dir: Path, capture_subdir: str, capture_indices,
                               charuco_target: CharucoTarget,
                               event_offset: int = SESSION3_EVENT_OFFSET) -> dict:
    """그리퍼캠뿐 아니라 고정캠 3대도 session3 보드를 본다(실측 확인: 15/15
    검출, 코너 23~56개) -- 그래서 4대 전부 meta에 넣는다. 예전엔 그리퍼캠만
    넣어서 고정캠의 보드 관측치를 통째로 빠뜨리고 있었다."""
    label_by_id = {v: k for k, v in LOCAL_CAM_IDS.items()}
    captures = []
    for idx in capture_indices:
        folder = f"{idx:03d}"
        cams = {}
        for local_id, label in label_by_id.items():
            img_path = session3_dir / capture_subdir / folder / f"cam_{label}.png"
            if not img_path.is_file():
                continue
            image = cv2.imread(str(img_path))
            n_corners = 0
            if image is not None:
                _corners, _ids, n_corners, _mc, _mi = charuco_target.detect(image)
                n_corners = int(n_corners or 0)
            rel = f"{capture_subdir}/{folder}/cam_{label}.png"
            cams[str(local_id)] = {"saved": True, "rgb_path": rel, "charuco_detect_n": n_corners}
        captures.append({
            "event_id": int(event_offset) + int(idx),
            "cams": cams,
        })
    return {"captures": captures, "charuco_board_config": CHARUCO_BOARD_CONFIG}


def summarize(state, diag, cam_init, gtc_init, board_init):
    return {
        "success": diag["success"],
        "train_reprojection_rmse_px": diag["train_reprojection_rmse_px"],
        "initial_reprojection_rmse_px": diag["initial_reprojection_rmse_px"],
        "n_parameters": diag["n_parameters"],
        "n_residuals": diag["n_residuals"],
        "T_gripper_cube_translation_mm": np.round(state.grasps[0][:3, 3] * 1000, 3).tolist(),
        "T_gripper_cam_translation_mm": np.round(state.gtc[:3, 3] * 1000, 3).tolist(),
        "T_gripper_cam_shift_from_init_mm": float(np.linalg.norm(state.gtc[:3, 3] - gtc_init[:3, 3]) * 1000),
        "T_base_board_shift_from_init_mm": float(np.linalg.norm(state.board[:3, 3] - board_init[:3, 3]) * 1000),
        "camera_shift_from_init_mm": {
            c: float(np.linalg.norm(state.cams[c][:3, 3] - cam_init[c][:3, 3]) * 1000)
            for c in state.cams
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session1-dir", default=str(SESSION1_DIR_DEFAULT))
    ap.add_argument("--session1-capture-subdir", default="capture_replayed")
    ap.add_argument("--session2-dir", default=str(SESSION2_DIR_DEFAULT))
    ap.add_argument("--session2-capture-subdir", default="capture_placed")
    ap.add_argument("--session3-dir", default=str(SESSION3_DIR_DEFAULT))
    ap.add_argument("--session3-capture-subdir", default="capture_replayed")
    ap.add_argument("--zeus-intrinsics-dir", default=str(REPO_ROOT / "intrinsics"))
    ap.add_argument("--ur3-intrinsics-dir", default=str(REPO_ROOT / "ur3_calibration" / "intrinsics"))
    ap.add_argument("--device-map", default=str(REPO_ROOT / "intrinsics" / "device_map.json"))
    ap.add_argument("--fit-json", default=str(FIT_JSON_DEFAULT),
                    help="고정캠 extrinsics/grasp offset 초기값 출처 (fit_grasp_offset.py 결과)")
    ap.add_argument("--fixed-min-corners", type=int, default=8)
    ap.add_argument("--cube-observation-policy", default="legacy", choices=("legacy", "core_multiface"))
    ap.add_argument("--out", default=str(REPO_ROOT / "zeus_gello_calibration" / "full_calibration_ablation.json"))
    args = ap.parse_args()

    session1_dir = Path(args.session1_dir)
    session2_dir = Path(args.session2_dir)
    session3_dir = Path(args.session3_dir)
    s1_root = session1_dir / args.session1_capture_subdir
    s2_root = session2_dir / args.session2_capture_subdir
    s3_root = session3_dir / args.session3_capture_subdir
    s1_indices = sorted(int(p.name) for p in s1_root.iterdir() if p.is_dir() and p.name.isdigit())
    s2_indices = sorted(int(p.name) for p in s2_root.iterdir() if p.is_dir() and p.name.isdigit())
    s3_indices = sorted(int(p.name) for p in s3_root.iterdir() if p.is_dir() and p.name.isdigit())
    print(f"session1: {s1_root} ({len(s1_indices)}개, 고정캠, grasp+FK)")
    print(f"session2: {s2_root} ({len(s2_indices)}개, 고정캠, 조건별)")
    print(f"session3: {s3_root} ({len(s3_indices)}개, 그리퍼캠, eye-in-hand 보드)")

    items = compute_ordered_targets(session2_dir)
    items_by_index = {idx: items[idx] for idx in s2_indices if idx < len(items)}

    K_map, D_map = load_intrinsics_by_label(
        Path(args.zeus_intrinsics_dir), Path(args.ur3_intrinsics_dir), Path(args.device_map))

    fit = json.loads(Path(args.fit_json).read_text())
    grasp_init = np.asarray(fit["T_gripper_cube"], dtype=np.float64)
    cam_init = {int(k.split("_", 1)[0]): np.asarray(v, dtype=np.float64)
                for k, v in fit["T_base_cam"].items()}
    print(f"초기값 로드 (from {args.fit_json}): 고정캠 {sorted(cam_init)}, T_gripper_cube "
          f"t_mm={np.round(grasp_init[:3,3]*1000,2).tolist()}")

    cube_cfg = get_default_cube_config()
    cube = AprilTagCubeTarget(cube_cfg)
    fixed_cam_ids = sorted(cam_init)  # 0,1,2
    all_cam_ids_cube = fixed_cam_ids  # 큐브 로더는 고정캠만 (그리퍼캠은 항상 0검출)
    NOT_A_CAM = -999  # 큐브 로더 호출 시 "그리퍼캠"을 존재하지 않는 id로 둬서 전부 고정캠 경로로

    # --- session1: grasp+FK (고정) ---
    meta_s1 = build_synthetic_meta(session1_dir, s1_indices, args.session1_capture_subdir)
    robot_T_s1 = load_robot_T(session1_dir, s1_indices, args.session1_capture_subdir)
    obs_s1, _ = load_cube_pixel_observations(
        str(session1_dir), meta_s1, cube, K_map, D_map, all_cam_ids_cube,
        gripper_cam_idx=NOT_A_CAM, exclude_gripped=False,
        fixed_min_corners=args.fixed_min_corners, image_scale=1.0,
        observation_policy=args.cube_observation_policy,
    )
    obs_s1 = [o for o in obs_s1 if int(o.cam) in cam_init]
    print(f"session1 관측치: {len(obs_s1)}개")

    # --- session2: 원본(no_fk 자유변수용) ---
    meta_s2 = build_synthetic_meta_placed(s2_root, s2_indices)
    obs_s2_raw, _ = load_cube_pixel_observations(
        str(session2_dir), meta_s2, cube, K_map, D_map, all_cam_ids_cube,
        gripper_cam_idx=NOT_A_CAM, exclude_gripped=False,
        fixed_min_corners=args.fixed_min_corners, image_scale=1.0,
        observation_policy=args.cube_observation_policy,
    )
    obs_s2_raw = [o for o in obs_s2_raw if int(o.cam) in cam_init]
    print(f"session2 관측치: {len(obs_s2_raw)}개")

    robot_T_s2_commanded = {
        SESSION2_EVENT_OFFSET + idx: pose6_to_T(item["target"])
        for idx, item in items_by_index.items()
    }
    obs_s2_as_grasp = [
        dataclasses.replace(o, set_idx=None, grasp_idx=0)
        for o in obs_s2_raw if int(o.event) - SESSION2_EVENT_OFFSET in items_by_index
    ]

    # --- session3: 그리퍼캠 eye-in-hand 보드 ---
    charuco_cfg = CHARUCO_BOARD_CFG
    charuco_target = CharucoTarget(charuco_cfg)
    meta_s3 = build_synthetic_meta_board(session3_dir, args.session3_capture_subdir, s3_indices, charuco_target)
    robot_T_s3 = {SESSION3_EVENT_OFFSET + k: v
                  for k, v in load_robot_T(session3_dir, s3_indices, args.session3_capture_subdir).items()}
    obs_s3 = load_board_pixel_observations(
        str(session3_dir), meta_s3, [GRIPPER_LOCAL_ID], gripper_cam_idx=GRIPPER_LOCAL_ID, image_scale=1.0)
    print(f"session3 보드 관측치: {len(obs_s3)}개 (그리퍼캠)")
    if len(obs_s3) < 5:
        print("[ERROR] eye-in-hand 초기화에 보드 관측치가 5개 미만 -- 중단합니다.")
        return

    gtc_init, board_init, eih_diag = estimate_board_handeye_initial(
        obs_s3, robot_T_s3, K_map, D_map, GRIPPER_LOCAL_ID)
    print(f"T_gripper_cam 초기값: t_mm={np.round(gtc_init[:3,3]*1000,2).tolist()} ({eih_diag})")

    robot_T_combined = {**robot_T_s1, **robot_T_s2_commanded, **robot_T_s3}

    options = SolverOptions()
    results = {}

    # --- no_fk ---
    set_ids = sorted(items_by_index)
    cube_init_no_fk, cube_init_diag = init_cube_poses_from_images(
        obs_s2_raw, K_map, D_map, cam_init, set_ids)
    missing = sorted(set(set_ids) - set(cube_init_no_fk))
    if missing:
        print(f"[WARN] 이미지만으로 초기 큐브 pose를 못 만든 set: {missing} (no_fk에서 제외)")
    obs_no_fk = (obs_s1
                 + [o for o in obs_s2_raw if o.set_idx is not None and int(o.set_idx) in cube_init_no_fk]
                 + obs_s3)
    state_no_fk = PoseState(
        cams=dict(cam_init), gtc=gtc_init.copy(), board=board_init.copy(),
        cubes=dict(cube_init_no_fk), grasps={0: grasp_init.copy()},
    )
    keys_no_fk = variable_keys(
        ["T_base_Ci", "T_gripper_cam", "T_base_board", "T_base_cube_by_set", "T_gripper_cube_by_grasp"],
        state_no_fk)
    final_no_fk, diag_no_fk = solve_corner_reprojection(
        observations=obs_no_fk, variable_keys_=keys_no_fk, reference_state=state_no_fk,
        robot_T=robot_T_combined, K_map=K_map, D_map=D_map,
        gripper_cam_idx=GRIPPER_LOCAL_ID, options=options,
    )
    results["no_fk"] = summarize(final_no_fk, diag_no_fk, cam_init, gtc_init, board_init)
    results["no_fk"]["n_session1_obs"] = len(obs_s1)
    results["no_fk"]["n_session2_sets_free"] = len(cube_init_no_fk)
    results["no_fk"]["n_session3_obs"] = len(obs_s3)

    # --- fixed_fk ---
    obs_fixed = obs_s1 + obs_s2_as_grasp + obs_s3
    state_fixed = PoseState(
        cams=dict(cam_init), gtc=gtc_init.copy(), board=board_init.copy(),
        cubes={}, grasps={0: grasp_init.copy()},
    )
    keys_fixed = variable_keys(
        ["T_base_Ci", "T_gripper_cam", "T_base_board", "T_gripper_cube_by_grasp"], state_fixed)
    final_fixed, diag_fixed = solve_corner_reprojection(
        observations=obs_fixed, variable_keys_=keys_fixed, reference_state=state_fixed,
        robot_T=robot_T_combined, K_map=K_map, D_map=D_map,
        gripper_cam_idx=GRIPPER_LOCAL_ID, options=options,
    )
    results["fixed_fk"] = summarize(final_fixed, diag_fixed, cam_init, gtc_init, board_init)
    results["fixed_fk"]["n_session1_obs"] = len(obs_s1)
    results["fixed_fk"]["n_session2_obs_as_grasp"] = len(obs_s2_as_grasp)
    results["fixed_fk"]["n_session3_obs"] = len(obs_s3)

    print()
    print(f"{'condition':>10} {'n_corners':>9} {'train_rmse_px':>15} {'initial_rmse_px':>17} {'success':>8}")
    for name, r in results.items():
        print(f"{name:>10} {r['n_residuals']//2:>9d} {r['train_reprojection_rmse_px']:>15.4f} "
              f"{r['initial_reprojection_rmse_px']:>17.4f} {str(r['success']):>8}")

    out = {
        "warning": ("train-pooled (session1+2+3 함께, 고정캠3+그리퍼캠1), held-out 분리 없음 "
                    "-- table1.py의 정식 지표가 아님. no_fk/fixed_fk 두 조건끼리 비교하는 용도."),
        "session1_dir": str(session1_dir), "session2_dir": str(session2_dir), "session3_dir": str(session3_dir),
        "fit_json_source": args.fit_json,
        "charuco_board_config": CHARUCO_BOARD_CONFIG,
        "eih_init_diag": eih_diag,
        "results": results,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
