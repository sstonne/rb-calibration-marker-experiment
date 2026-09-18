#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zeus_gello_calibration/fit_placement_fk_ablation.py -- no-FK vs fixed-FK 비교.

session1(그리퍼로 큐브를 쥐고 이리저리 움직이며 찍은 16장, capture_replayed/)과
구형 pick-and-place collector가 남긴 "큐브를 바닥에 놓고 찍은" 15장
(capture_placed/)을 **하나의 pooled fit**으로 같이 넣어서 카메라 extrinsics를
같이 최적화하고, train reprojection RMSE를 비교한다.

*** session1은 "flange-to-cube 상수를 미리 구해서 얼려두는 전처리 단계"가
아니다 -- session1 이미지 자체도 이 fit의 residual로 같이 들어간다. *** (이전
버전은 fit_grasp_offset.py로 session1만 따로 fit해서 나온 T_gripper_cube를
얼린 상수로 session2 fit에 꽂아넣었는데, 그러면 session1 데이터가 이번
fit에 전혀 기여하지 못한다. 이번 버전은 session1+session2 관측치를 합쳐서
한 번에 푼다. fit_grasp_offset.py의 결과(pass1_grasp_offset_replayed.json)는
카메라 extrinsics/grasp offset의 "초기값"으로만 쓰인다.)

두 조건 모두 session1의 16개는 항상 "grasp+FK" 모델
(T_base_cube[e] = FK(q_e) @ T_gripper_cube, T_gripper_cube는 자유 변수)로
다룬다 -- 쥔 채로 돌아다니는 시퀀스를 이미지만으로 설명하려면 이 모델 말고는
합리적인 대안이 없고, ur3_calibration의 기존 ablation도 이렇게 한다. 조건
차이는 session2의 15개 placement를 어떻게 다루느냐에만 있다:

  no_fk    -- session2 각 placement의 큐브 pose를 이미지(PnP)만으로 독립적인
              자유 변수로 최적화 (T_base_cube_by_set, 15개 각자 6DoF).
  fixed_fk -- session2 placement도 session1과 "같은" grasp+FK 모델을 강제로
              적용한다: 그 placement에서 실제로 명령했던 place 목표 pose
              (build_plan의 dest_pose)를 마치 그 순간의 로봇 FK인 것처럼 넣어서
              T_base_cube = (그 place 목표 pose) @ T_gripper_cube 로 계산한다
              (T_gripper_cube는 session1+session2 전체에서 공유되는 하나의 자유
              변수 -- 이 fit 안에서 같이 풀린다, 얼린 상수가 아니다). 즉
              "session1에서 확립한 것과 똑같은 FK 체인이, 각 placement마다
              따로 자유도를 안 줘도 그 이미지들까지 설명하는가?"를 본다.

train RMSE 차이가 "FK(grasp offset) 체인을 신뢰해도 되는지"에 대한 지표다.

주의: table1.py의 정식 held-out Table 1 지표가 아니다 (train-pooled, held-out
분리 없음) -- no_fk/fixed_fk 두 조건끼리 비교하는 용도.

사용법:
  python fit_placement_fk_ablation.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from calibration_pipeline import se3 as cp  # noqa: E402
from calibration_pipeline.apriltag_cube import AprilTagCubeTarget  # noqa: E402
from calibration_pipeline.config import get_default_cube_config  # noqa: E402
from calibration_pipeline.observations import load_cube_pixel_observations  # noqa: E402
from calibration_pipeline.path_evaluation import solve_observed_pose  # noqa: E402
from calibration_pipeline.reprojection import (  # noqa: E402
    PoseState, SolverOptions, solve_corner_reprojection, variable_keys,
)
from robot.backends.zeus_client import pose6_to_T  # noqa: E402

from fit_grasp_offset import (  # noqa: E402
    LOCAL_CAM_IDS, NOT_GRIPPER_SENTINEL, build_synthetic_meta, load_intrinsics_by_label,
    load_robot_T,
)
from session2_pick_and_place import (  # noqa: E402
    SESSION2_DIR_DEFAULT, compute_ordered_targets,
)

SESSION1_DIR_DEFAULT = REPO_ROOT / "zeus_gello_calibration" / "data" / "session1_handheld_fixed_cam_0909"
FIT_JSON_DEFAULT = REPO_ROOT / "zeus_gello_calibration" / "results" / "fits" / "pass1_grasp_offset_replayed.json"
SESSION2_EVENT_OFFSET = 1000  # session1과 event id가 안 겹치게 (robot_T 딕셔너리 키 충돌 방지)


def build_synthetic_meta_placed(capture_root: Path, capture_indices) -> dict:
    """load_cube_pixel_observations 스키마용 in-memory meta -- cube_gripped=False,
    각 캡처는 자기 자신의 set_index를 가짐. event_id는 session1과 안 겹치게 offset."""
    label_by_id = {v: k for k, v in LOCAL_CAM_IDS.items()}
    captures = []
    for idx in capture_indices:
        folder = f"{idx:03d}"
        cams = {}
        for local_id, label in label_by_id.items():
            # capture_root 의 실제 폴더명(capture_placed, capture_placed_0914 ...)을 써야
            # 다른 촬영분(--session2-capture-subdir)이 예전 사진을 읽는 일이 없다.
            rel = f"{capture_root.name}/{folder}/cam_{label}.png"
            if (capture_root.parent / rel).is_file():
                cams[str(local_id)] = {"saved": True, "rgb_path": rel}
        captures.append({
            "event_id": SESSION2_EVENT_OFFSET + int(idx),
            "cube_gripped": False,
            "set_index": int(idx),
            "cams": cams,
        })
    return {"captures": captures}


def init_cube_poses_from_images(observations, K_map, D_map, cam_init, set_ids):
    """각 placement(set)마다, 그 set을 본 카메라들의 PnP + 초기 카메라
    extrinsics로 T_base_cube 후보를 만들어 robust 평균 (no_fk 초기값)."""
    by_set = {s: [] for s in set_ids}
    for obs in observations:
        s = obs.set_idx
        if s is None or int(s) not in by_set:
            continue
        cam_id = int(obs.cam)
        if cam_id not in cam_init:
            continue
        T_cam_cube = solve_observed_pose(obs, K_map, D_map)
        if T_cam_cube is None:
            continue
        by_set[int(s)].append(cam_init[cam_id] @ T_cam_cube)
    init, diag = {}, {}
    for s, cands in by_set.items():
        if not cands:
            continue
        if len(cands) == 1:
            init[s], diag[s] = cands[0], {"n_cams": 1}
        else:
            T_avg, d = cp.robust_se3_average(cands, None)
            init[s], diag[s] = T_avg, {"n_cams": len(cands), **d}
    return init, diag


def summarize(state, diag, cam_init):
    return {
        "success": diag["success"],
        "train_reprojection_rmse_px": diag["train_reprojection_rmse_px"],
        "initial_reprojection_rmse_px": diag["initial_reprojection_rmse_px"],
        "n_parameters": diag["n_parameters"],
        "n_residuals": diag["n_residuals"],
        "T_gripper_cube_translation_mm": np.round(state.grasps[0][:3, 3] * 1000, 3).tolist(),
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
    ap.add_argument("--zeus-intrinsics-dir", default=str(REPO_ROOT / "intrinsics"))
    ap.add_argument("--ur3-intrinsics-dir", default=str(REPO_ROOT / "ur3_calibration" / "intrinsics"))
    ap.add_argument("--device-map", default=str(REPO_ROOT / "intrinsics" / "device_map.json"))
    ap.add_argument("--fit-json", default=str(FIT_JSON_DEFAULT),
                    help="카메라 extrinsics/grasp offset 초기값 출처 (fit_grasp_offset.py 결과)")
    ap.add_argument("--fixed-min-corners", type=int, default=8)
    ap.add_argument("--cube-observation-policy", default="legacy", choices=("legacy", "core_multiface"))
    ap.add_argument("--out", default=str(REPO_ROOT / "zeus_gello_calibration" / "results" / "heldout" / "placement_fk_ablation.json"))
    args = ap.parse_args()

    session1_dir = Path(args.session1_dir)
    session2_dir = Path(args.session2_dir)
    s1_capture_root = session1_dir / args.session1_capture_subdir
    s2_capture_root = session2_dir / args.session2_capture_subdir
    s1_indices = sorted(int(p.name) for p in s1_capture_root.iterdir() if p.is_dir() and p.name.isdigit())
    s2_indices = sorted(int(p.name) for p in s2_capture_root.iterdir() if p.is_dir() and p.name.isdigit())
    print(f"session1: {s1_capture_root}  ({len(s1_indices)}개, grasp+FK 고정 모델)")
    print(f"session2: {s2_capture_root}  ({len(s2_indices)}개, 조건별로 다르게 다룸)")

    items = compute_ordered_targets(session2_dir)
    if len(items) != len(s2_indices):
        print(f"[WARN] compute_ordered_targets()가 {len(items)}개, 캡처 폴더가 "
              f"{len(s2_indices)}개 -- legacy pick-and-place 실행 순서와 안 맞을 수 있습니다.")
    items_by_index = {idx: items[idx] for idx in s2_indices if idx < len(items)}

    K_map, D_map = load_intrinsics_by_label(
        Path(args.zeus_intrinsics_dir), Path(args.ur3_intrinsics_dir), Path(args.device_map))

    fit = json.loads(Path(args.fit_json).read_text())
    grasp_init = np.asarray(fit["T_gripper_cube"], dtype=np.float64)
    cam_init = {int(k.split("_", 1)[0]): np.asarray(v, dtype=np.float64)
                for k, v in fit["T_base_cam"].items()}
    print(f"초기값 로드 (from {args.fit_json}): 카메라 {sorted(cam_init)}, T_gripper_cube "
          f"t_mm={np.round(grasp_init[:3,3]*1000,2).tolist()}\n")

    cube_cfg = get_default_cube_config()
    cube = AprilTagCubeTarget(cube_cfg)
    all_cam_ids = sorted(LOCAL_CAM_IDS.values())

    # --- session1: 항상 grasp+FK 모델 (양쪽 조건 공통) ---
    meta_s1 = build_synthetic_meta(session1_dir, s1_indices, args.session1_capture_subdir)
    robot_T_s1 = load_robot_T(session1_dir, s1_indices, args.session1_capture_subdir)
    obs_s1, _ = load_cube_pixel_observations(
        str(session1_dir), meta_s1, cube, K_map, D_map, all_cam_ids,
        gripper_cam_idx=NOT_GRIPPER_SENTINEL, exclude_gripped=False,
        fixed_min_corners=args.fixed_min_corners, image_scale=1.0,
        observation_policy=args.cube_observation_policy,
    )
    obs_s1 = [o for o in obs_s1 if int(o.cam) in cam_init]
    print(f"session1 관측치: {len(obs_s1)}개 (사용 가능 카메라만)")

    # --- session2: 원본(set 자유 변수용, no_fk) ---
    meta_s2 = build_synthetic_meta_placed(s2_capture_root, s2_indices)
    obs_s2_raw, _ = load_cube_pixel_observations(
        str(session2_dir), meta_s2, cube, K_map, D_map, all_cam_ids,
        gripper_cam_idx=NOT_GRIPPER_SENTINEL, exclude_gripped=False,
        fixed_min_corners=args.fixed_min_corners, image_scale=1.0,
        observation_policy=args.cube_observation_policy,
    )
    obs_s2_raw = [o for o in obs_s2_raw if int(o.cam) in cam_init]
    print(f"session2 관측치: {len(obs_s2_raw)}개 (사용 가능 카메라만)")

    # 같은 코너 데이터를 grasp 모델로 재태깅 (fixed_fk용): event -> place 목표 pose를
    # robot_T에 넣어, session1과 똑같은 "FK(q) @ T_gripper_cube" 잔차 경로를 태운다.
    robot_T_s2_commanded = {
        SESSION2_EVENT_OFFSET + idx: pose6_to_T(item["target"])
        for idx, item in items_by_index.items()
    }
    obs_s2_as_grasp = [
        dataclasses.replace(o, set_idx=None, grasp_idx=0)
        for o in obs_s2_raw if int(o.event) - SESSION2_EVENT_OFFSET in items_by_index
    ]
    dropped = len(obs_s2_raw) - len(obs_s2_as_grasp)
    if dropped:
        print(f"[WARN] {dropped}개 session2 관측치는 대응하는 place 목표를 못 찾아 fixed_fk에서 제외")

    robot_T_combined = {**robot_T_s1, **robot_T_s2_commanded}

    options = SolverOptions()
    results = {}

    # --- no_fk ---
    set_ids = sorted(items_by_index)
    cube_init_no_fk, cube_init_diag = init_cube_poses_from_images(
        obs_s2_raw, K_map, D_map, cam_init, set_ids)
    missing = sorted(set(set_ids) - set(cube_init_no_fk))
    if missing:
        print(f"[WARN] 이미지만으로 초기 큐브 pose를 못 만든 set: {missing} (no_fk에서 제외)")
    obs_no_fk = obs_s1 + [o for o in obs_s2_raw if o.set_idx is not None and int(o.set_idx) in cube_init_no_fk]
    state_no_fk = PoseState(
        cams=dict(cam_init), gtc=np.eye(4), board=None,
        cubes=dict(cube_init_no_fk), grasps={0: grasp_init.copy()},
    )
    keys_no_fk = variable_keys(["T_base_Ci", "T_base_cube_by_set", "T_gripper_cube_by_grasp"], state_no_fk)
    final_no_fk, diag_no_fk = solve_corner_reprojection(
        observations=obs_no_fk, variable_keys_=keys_no_fk, reference_state=state_no_fk,
        robot_T=robot_T_combined, K_map=K_map, D_map=D_map,
        gripper_cam_idx=NOT_GRIPPER_SENTINEL, options=options,
    )
    results["no_fk"] = summarize(final_no_fk, diag_no_fk, cam_init)
    results["no_fk"]["n_session1_obs"] = len(obs_s1)
    results["no_fk"]["n_session2_sets_free"] = len(cube_init_no_fk)

    # --- fixed_fk ---
    obs_fixed = obs_s1 + obs_s2_as_grasp
    state_fixed = PoseState(
        cams=dict(cam_init), gtc=np.eye(4), board=None,
        cubes={}, grasps={0: grasp_init.copy()},
    )
    keys_fixed = variable_keys(["T_base_Ci", "T_gripper_cube_by_grasp"], state_fixed)
    final_fixed, diag_fixed = solve_corner_reprojection(
        observations=obs_fixed, variable_keys_=keys_fixed, reference_state=state_fixed,
        robot_T=robot_T_combined, K_map=K_map, D_map=D_map,
        gripper_cam_idx=NOT_GRIPPER_SENTINEL, options=options,
    )
    results["fixed_fk"] = summarize(final_fixed, diag_fixed, cam_init)
    results["fixed_fk"]["n_session1_obs"] = len(obs_s1)
    results["fixed_fk"]["n_session2_obs_as_grasp"] = len(obs_s2_as_grasp)

    print()
    print(f"{'condition':>10} {'n_corners':>9} {'train_rmse_px':>15} {'initial_rmse_px':>17} {'success':>8}")
    for name, r in results.items():
        print(f"{name:>10} {r['n_residuals']//2:>9d} {r['train_reprojection_rmse_px']:>15.4f} "
              f"{r['initial_reprojection_rmse_px']:>17.4f} {str(r['success']):>8}")

    out = {
        "warning": ("train-pooled (session1+session2 함께), held-out 분리 없음 -- table1.py의 "
                    "정식 지표가 아님. no_fk/fixed_fk 두 조건끼리 비교하는 용도."),
        "session1_dir": str(session1_dir),
        "session2_dir": str(session2_dir),
        "fit_json_source": args.fit_json,
        "results": results,
        "cube_init_diag_no_fk": cube_init_diag,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
