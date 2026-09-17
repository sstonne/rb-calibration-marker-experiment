#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zeus_gello_calibration/fit_calibration_methods.py -- 통합/독립 x raw-fk/no-fk
3가지 방식을 "같은 데이터, 다른 solve 전략"으로 비교한다.

*** 핵심 원칙: 통합이든 독립이든 캘리브레이션에 쓰는 데이터의 총량은 완전히
같아야 한다. 차이는 그 데이터를 하나로 묶어서 푸느냐(통합) vs 두 그룹으로
나눠서 완전히 따로 푸느냐(독립)뿐이다. ***

데이터 풀 (통합/독립 공통, 총 4가지 소스):
  session1        -- 고정캠 3대가 그리퍼로 쥔 큐브를 봄 (grasp+FK 모델)
  session2-고정캠  -- 고정캠 3대가 바닥에 놓인 큐브를 봄 (세트별)
  session2-그리퍼캠 -- ★그리퍼캠도 바닥에 놓인 큐브를 본다★ (capture_placed의
                      새 파킹 위치에서 실측 확인: PnP 15/15 성공, err<1px).
                      이전 버전은 이걸 빠뜨렸다.
  session3-그리퍼캠 -- 그리퍼캠이 바닥 마커보드를 봄 (eye-in-hand)

  no_fk    : session2(고정캠+그리퍼캠 둘 다)의 큐브 pose를 세트별 자유 변수로.
  raw-fk   : session2의 큐브 pose를 "그 placement에서 실제 명령한 place pose
             @ T_gripper_cube" FK값으로 고정(정적 상수, 최적화 중 안 바뀜).

  통합(unified)     -- 위 4개 소스를 전부 하나의 최소제곱에 넣어 한 번에 푼다.
  독립(independent) -- 고정캠 그룹(session1+session2-고정캠+session3-고정캠)과
                       그리퍼 그룹(session2-그리퍼캠+session3-그리퍼캠)을 정보
                       교환 전혀 없이 완전히 따로 푼다(solve_parallel_fixed/
                       gripper). 어느 쪽도 다른 쪽 값을 넘겨받지 않는다.
                       session2 큐브는 양쪽 다 학습에 쓰되 큐브 pose 변수를
                       공유하지 않는다(각 그룹이 별개의 자유 변수로 따로 추정).
                       다 끝난 뒤 "두 그룹이 같은 session2 큐브에 대해 각자
                       추정한 pose가 얼마나 일치하는가"를 사후 합의
                       (consensus_check)로 확인한다 -- 결과를 바꾸지 않는
                       순수 진단 지표.

raw-fk는 no_fk에서만 통합/독립 구분이 의미가 있다 -- 큐브 위치를 상수로
고정하면 고정캠/그리퍼캠 블록이 항상 수학적으로 분리되어(block-separable)
통합과 독립이 완전히 같은 답을 내므로, 독립_raw-fk는 따로 안 만든다.

table1.py의 정식 held-out 지표 아님 (train-pooled). 세 조건(통합_no-fk,
통합_raw-fk, 독립_no-fk)끼리 비교하는 용도. table1.py의 공식
sequential_frozen_stage(A1) 알고리즘 자체는 더 이상 여기 없다 -- 그건
"독립"의 정의가 아니라고 판단해서(한쪽이 다른 쪽에 일방적으로 맞추는 구조라
"서로 정보 교환 없음"이라는 독립의 정의와 다름) 제거했다.

사용법:
  python fit_calibration_methods.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from calibration_pipeline.apriltag_cube import AprilTagCubeTarget, inv_T  # noqa: E402
from calibration_pipeline.board_config import charuco_config_from_dict  # noqa: E402
from calibration_pipeline.charuco import CharucoTarget  # noqa: E402
from calibration_pipeline.config import get_default_cube_config  # noqa: E402
from calibration_pipeline.observations import (  # noqa: E402
    load_board_pixel_observations, load_cube_pixel_observations,
)
from calibration_pipeline.path_evaluation import solve_observed_pose  # noqa: E402
from calibration_pipeline.reprojection import (  # noqa: E402
    PoseState, SolverOptions, pose_delta, project_points, solve_corner_reprojection, variable_keys,
)
from calibration_pipeline.table1 import estimate_board_handeye_initial  # noqa: E402
from robot.backends.zeus_client import pose6_to_T  # noqa: E402

from calibration_pipeline import se3 as cp  # noqa: E402
from fit_grasp_offset import LOCAL_CAM_IDS, build_synthetic_meta, load_intrinsics_by_label, load_robot_T  # noqa: E402
from fit_placement_fk_ablation import SESSION2_EVENT_OFFSET, build_synthetic_meta_placed  # noqa: E402
from fit_full_calibration import CHARUCO_BOARD_CONFIG, GRIPPER_LOCAL_ID, build_synthetic_meta_board  # noqa: E402
from session2_pick_and_place import SESSION2_DIR_DEFAULT, compute_ordered_targets  # noqa: E402

SESSION1_DIR_DEFAULT = REPO_ROOT / "zeus_gello_calibration" / "data" / "session1_handheld_fixed_cam_0909"
SESSION3_DIR_DEFAULT = REPO_ROOT / "zeus_gello_calibration" / "data" / "session3_wrist_motion_gripper_cam_0909"
FIT_JSON_DEFAULT = REPO_ROOT / "zeus_gello_calibration" / "pass1_grasp_offset_replayed.json"

# 이벤트 id 네임스페이스 충돌 방지 (robot_T 딕셔너리 키). session2는
# build_synthetic_meta_placed가 이미 event_id에 SESSION2_EVENT_OFFSET(1000)을
# 박아서 만들기 때문에(고정캠/그리퍼캠 공용 meta), 그리퍼캠 쪽도 그대로
# 재사용한다 -- 별도 offset을 또 더하면 이중 offset 버그가 난다.
SESSION3_EVENT_OFFSET = 2000
FIT_SUFFIX = ""   # --tag 로 설정; fit_<조건><suffix>.json


def export_fit_json(path: Path, T_gripper_cube, T_base_cam: dict, T_gripper_cam=None):
    """gt_pick_test.py/gt_compare_fits.py의 load_fit()이 읽는 스키마로 저장.
    T_gripper_cam은 그리퍼캠 추론에 필요 (없으면 그 방식은 그리퍼캠 추론 불가)."""
    label_by_id = {v: k for k, v in LOCAL_CAM_IDS.items()}
    payload = {
        "T_gripper_cube": np.asarray(T_gripper_cube, dtype=np.float64).tolist(),
        "T_base_cam": {
            f"{c}_{label_by_id[c]}": np.asarray(T, dtype=np.float64).tolist()
            for c, T in T_base_cam.items()
        },
    }
    if T_gripper_cam is not None:
        payload["T_gripper_cam"] = np.asarray(T_gripper_cam, dtype=np.float64).tolist()
    path.write_text(json.dumps(payload, indent=2))
    print(f"  -> {path}")


def rmse_px(errs):
    errs = np.asarray(errs, dtype=np.float64)
    return float(np.sqrt(np.mean(errs ** 2))) if errs.size else float("nan")


def fk_anchor_cubes(items_by_index, T_gripper_cube):
    return {idx: pose6_to_T(item["target"]) @ T_gripper_cube for idx, item in items_by_index.items()}


def init_cube_poses(obs_list, K_map, D_map, cam_init, gtc_init, robot_T, gripper_id, set_ids):
    """세트별 T_base_cube 초기값 -- 고정캠은 cam_init[c]@T_cam_cube, 그리퍼캠은
    robot_T[event]@gtc_init@T_cam_cube (그 사진 찍은 순간의 실제 로봇 pose 사용)."""
    by_set = {s: [] for s in set_ids}
    for o in obs_list:
        if o.set_idx is None or int(o.set_idx) not in by_set:
            continue
        T_cam_cube = solve_observed_pose(o, K_map, D_map)
        if T_cam_cube is None:
            continue
        c = int(o.cam)
        if c == gripper_id:
            if int(o.event) not in robot_T:
                continue
            cand = robot_T[int(o.event)] @ gtc_init @ T_cam_cube
        else:
            if c not in cam_init:
                continue
            cand = cam_init[c] @ T_cam_cube
        by_set[int(o.set_idx)].append(cand)
    init = {}
    for s, cands in by_set.items():
        if not cands:
            continue
        init[s] = cands[0] if len(cands) == 1 else cp.robust_se3_average(cands, None)[0]
    return init


def load_all_data(args):
    session1_dir, session2_dir, session3_dir = Path(args.session1_dir), Path(args.session2_dir), Path(args.session3_dir)
    s1_root = session1_dir / args.session1_capture_subdir
    s2_root = session2_dir / args.session2_capture_subdir
    s3_root = session3_dir / args.session3_capture_subdir
    s1_idx = sorted(int(p.name) for p in s1_root.iterdir() if p.is_dir() and p.name.isdigit())
    s2_idx = sorted(int(p.name) for p in s2_root.iterdir() if p.is_dir() and p.name.isdigit())
    s3_idx = sorted(int(p.name) for p in s3_root.iterdir() if p.is_dir() and p.name.isdigit())
    items = compute_ordered_targets(session2_dir)
    items_by_index = {idx: items[idx] for idx in s2_idx if idx < len(items)}

    K_map, D_map = load_intrinsics_by_label(
        Path(args.zeus_intrinsics_dir), Path(args.ur3_intrinsics_dir), Path(args.device_map))
    fit = json.loads(Path(args.fit_json).read_text())
    grasp_init = np.asarray(fit["T_gripper_cube"], dtype=np.float64)
    cam_init = {int(k.split("_", 1)[0]): np.asarray(v, dtype=np.float64) for k, v in fit["T_base_cam"].items()}
    fixed_ids = sorted(cam_init)
    all_cam_ids = sorted(set(fixed_ids) | {GRIPPER_LOCAL_ID})

    cube_config_path = getattr(args, "cube_config", None)
    if cube_config_path:
        from calibration_pipeline.cube_config import load_cube_config_from_json_file
        cube_cfg, cube_src = load_cube_config_from_json_file(cube_config_path)
        if cube_cfg is None:
            raise SystemExit(f"cube config를 못 읽었습니다: {cube_config_path}")
        print(f"cube config: {cube_config_path} ({cube_src})")
        cube = AprilTagCubeTarget(cube_cfg)
    else:
        cube = AprilTagCubeTarget(get_default_cube_config())

    # session1 (고정캠, grasp+FK) -- 그리퍼캠은 여기선 항상 0검출이라 굳이 안 실음
    meta_s1 = build_synthetic_meta(session1_dir, s1_idx, args.session1_capture_subdir)
    robot_T_s1 = load_robot_T(session1_dir, s1_idx, args.session1_capture_subdir)
    obs_s1, _ = load_cube_pixel_observations(
        str(session1_dir), meta_s1, cube, K_map, D_map, fixed_ids, gripper_cam_idx=-999,
        exclude_gripped=False, fixed_min_corners=args.fixed_min_corners, image_scale=1.0,
        observation_policy=args.cube_observation_policy)
    obs_s1 = [o for o in obs_s1 if int(o.cam) in cam_init]

    # session2 (고정캠 3대 + 그리퍼캠 1대, 전부 큐브-세트) -- 하나의 meta로 같이 로드
    meta_s2 = build_synthetic_meta_placed(s2_root, s2_idx)
    obs_s2_all, _ = load_cube_pixel_observations(
        str(session2_dir), meta_s2, cube, K_map, D_map, all_cam_ids, gripper_cam_idx=-999,
        exclude_gripped=False, fixed_min_corners=args.fixed_min_corners, image_scale=1.0,
        observation_policy=args.cube_observation_policy)
    obs_s2_fixed = [o for o in obs_s2_all if int(o.cam) in cam_init]
    obs_s2_gripper_raw = [o for o in obs_s2_all if int(o.cam) == GRIPPER_LOCAL_ID]
    # 그리퍼캠 관측치의 event id를 그 사진을 실제로 찍은 순간(파킹 pose)의 로봇
    # pose와 연결 -- 큐브를 놓은 순간이 아니라 촬영한 순간의 FK가 카메라 pose다.
    robot_T_s2_photo = load_robot_T(session2_dir, s2_idx, args.session2_capture_subdir)
    obs_s2_gripper = obs_s2_gripper_raw  # event id는 이미 build_synthetic_meta_placed가 offset해서 나옴
    robot_T_s2_gripper = {SESSION2_EVENT_OFFSET + k: v for k, v in robot_T_s2_photo.items()}
    print(f"session2 그리퍼캠 큐브 관측치: {len(obs_s2_gripper)}개 (신규 -- 예전엔 빠뜨렸음)")

    # session3 (보드) -- 고정캠 3대도 그리퍼캠만큼 잘 본다(실측 확인: 15/15
    # 검출, 코너 23~56개). 예전엔 그리퍼캠만 불러와서 고정캠의 보드 관측치를
    # 통째로 빠뜨리고 있었다.
    charuco_target = CharucoTarget(charuco_config_from_dict(CHARUCO_BOARD_CONFIG))
    meta_s3 = build_synthetic_meta_board(session3_dir, args.session3_capture_subdir, s3_idx, charuco_target)
    robot_T_s3 = {SESSION3_EVENT_OFFSET + k: v
                  for k, v in load_robot_T(session3_dir, s3_idx, args.session3_capture_subdir).items()}
    obs_s3_all = load_board_pixel_observations(
        str(session3_dir), meta_s3, all_cam_ids, gripper_cam_idx=GRIPPER_LOCAL_ID, image_scale=1.0)
    obs_s3_fixed = [o for o in obs_s3_all if int(o.cam) in cam_init]
    obs_s3_gripper = [o for o in obs_s3_all if int(o.cam) == GRIPPER_LOCAL_ID]
    if getattr(args, "s3_gripper_only", False):
        # session3는 그리퍼캠(eye-in-hand) 전용으로만 쓴다 -- 고정캠 보드 관측은
        # session2 사진에서 이미 들어오므로 여기 고정캠 관측을 뺀다.
        print(f"[s3-gripper-only] session3 고정캠 보드 관측 {len(obs_s3_fixed)}개 제외")
        obs_s3_fixed = []
        obs_s3_all = list(obs_s3_gripper)
    print(f"session3 고정캠 보드 관측치: {len(obs_s3_fixed)}개 (신규 -- 예전엔 빠뜨렸음)")

    # session2 사진에도 같은 보드가 그대로 바닥에 있다 (실측: 고정캠 8~60코너,
    # 그리퍼캠 14/15장 39~60코너). 그리고 session2/session3 사이에 보드가 안
    # 움직였다(같은 고정캠으로 본 base 좌표 차이 0.25~0.42mm) -- 그래서 session3와
    # 같은 T_base_board 변수를 공유한다. 예전엔 session2에서 큐브만 뽑고 보드는
    # 통째로 버리고 있었다. event id는 session2 것(SESSION2_EVENT_OFFSET+idx)이라
    # 그리퍼캠 보드 관측은 robot_T_s2_gripper(촬영 순간 pose)와 연결된다.
    meta_s2_board = build_synthetic_meta_board(session2_dir, args.session2_capture_subdir, s2_idx,
                                               charuco_target, event_offset=SESSION2_EVENT_OFFSET)
    obs_s2_board = load_board_pixel_observations(
        str(session2_dir), meta_s2_board, all_cam_ids, gripper_cam_idx=GRIPPER_LOCAL_ID, image_scale=1.0)
    obs_s2_board_fixed = [o for o in obs_s2_board if int(o.cam) in cam_init]
    obs_s2_board_gripper = [o for o in obs_s2_board if int(o.cam) == GRIPPER_LOCAL_ID]
    print(f"session2 보드 관측치: 고정캠 {len(obs_s2_board_fixed)}개 / 그리퍼캠 {len(obs_s2_board_gripper)}개 (신규 -- 예전엔 빠뜨렸음)")

    print(f"session1 {len(obs_s1)}개 / session2-고정캠 {len(obs_s2_fixed)}개 / "
          f"session2-그리퍼캠 {len(obs_s2_gripper)}개 / session2-보드 {len(obs_s2_board)}개 / "
          f"session3-고정캠 {len(obs_s3_fixed)}개 / session3-그리퍼캠 {len(obs_s3_gripper)}개")
    total = (len(obs_s1) + len(obs_s2_fixed) + len(obs_s2_gripper) + len(obs_s2_board)
             + len(obs_s3_fixed) + len(obs_s3_gripper))
    print(f"총 관측치: {total}개 (통합/독립 공통, 같은 양)\n")

    observation_groups = {
        "session1_cube": obs_s1,
        "session2_fixed_cube": obs_s2_fixed,
        "session2_gripper_cube": obs_s2_gripper,
        "session2_board": obs_s2_board,
        "session3_board": obs_s3_all,
    }

    def file_source(path):
        source = Path(path).resolve()
        return {"path": str(source),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}

    # Capture-directory names alone are insufficient: 0909 and 0914 share
    # parent session directories but use different images, rigs and geometry.
    # Preserve the exact selected roots and numerical intrinsics for callers
    # producing a result artifact, without changing the loaded observations.
    source_data_provenance = {
        "schema": "zeus_calibration_loader_provenance_v1",
        "capture_sources": {
            name: {"root": str(root.resolve()), "capture_indices": indices}
            for name, root, indices in (
                ("session1", s1_root, s1_idx),
                ("session2", s2_root, s2_idx),
                ("session3", s3_root, s3_idx),
            )
        },
        "cube_config": (file_source(cube_config_path) if cube_config_path else {
            "path": None, "source": "calibration_pipeline.config.get_default_cube_config"}),
        "initial_grasp_fit": file_source(args.fit_json),
        "device_map": file_source(args.device_map),
        "intrinsics": {
            str(camera): {
                "K": np.asarray(K_map[camera]).tolist(),
                "D": np.asarray(D_map[camera]).tolist(),
                "values_sha256": hashlib.sha256(
                    np.asarray(K_map[camera], dtype="<f8").tobytes()
                    + np.asarray(D_map[camera], dtype="<f8").tobytes()).hexdigest(),
            }
            for camera in sorted(K_map)
        },
        "cube_observation_policy": str(args.cube_observation_policy),
        "fixed_min_corners": int(args.fixed_min_corners),
        "s3_gripper_only": bool(getattr(args, "s3_gripper_only", False)),
        "observation_counts": {name: len(group) for name, group in observation_groups.items()},
        "session2_event_identity": {
            str(SESSION2_EVENT_OFFSET + index): {
                "placement_id": index,
                "capture_directory": str((s2_root / f"{index:03d}").resolve()),
            }
            for index in s2_idx
        },
    }

    return dict(
        cam_init=cam_init, grasp_init=grasp_init, K_map=K_map, D_map=D_map,
        obs_s1=obs_s1, robot_T_s1=robot_T_s1,
        obs_s2_fixed=obs_s2_fixed, obs_s2_gripper=obs_s2_gripper, robot_T_s2_gripper=robot_T_s2_gripper,
        obs_s2_board=obs_s2_board, obs_s2_board_fixed=obs_s2_board_fixed, obs_s2_board_gripper=obs_s2_board_gripper,
        obs_s3=obs_s3_all, obs_s3_fixed=obs_s3_fixed, obs_s3_gripper=obs_s3_gripper, robot_T_s3=robot_T_s3,
        items_by_index=items_by_index,
        source_data_provenance=source_data_provenance,
    )


# ------------------------------------------------------------- cube-only (보드 제외)
def strip_board_data(data):
    """보드 관측(session3 전체 + session2 보드)을 전부 뺀 데이터 -- late_table1 B2
    (-board, cube only)에 대응. session3는 아예 안 쓰는 조건."""
    d = dict(data)
    for k in ("obs_s3", "obs_s3_fixed", "obs_s3_gripper", "obs_s2_board", "obs_s2_board_fixed", "obs_s2_board_gripper"):
        d[k] = []
    return d


def init_gtc_from_cubes(data):
    """보드 없이 T_gripper_cam 초기값: 고정캠으로 초기화한 세트별 큐브 pose(base)와
    그리퍼캠 단일 이미지 PnP(T_cam_cube), 촬영 순간 FK로
    gtc = inv(robot_T) @ T_base_cube @ inv(T_cam_cube) 후보들을 robust 평균."""
    cam_init, K_map, D_map = data["cam_init"], data["K_map"], data["D_map"]
    set_ids = sorted(data["items_by_index"])
    cubes_fixed = init_cube_poses(data["obs_s2_fixed"], K_map, D_map, cam_init, np.eye(4), {}, -999, set_ids)
    robot_T = data["robot_T_s2_gripper"]
    cands = []
    for o in data["obs_s2_gripper"]:
        s = int(o.set_idx) if o.set_idx is not None else None
        if s not in cubes_fixed or int(o.event) not in robot_T:
            continue
        T_cam_cube = solve_observed_pose(o, K_map, D_map)
        if T_cam_cube is None:
            continue
        cands.append(inv_T(robot_T[int(o.event)]) @ cubes_fixed[s] @ inv_T(T_cam_cube))
    if not cands:
        raise RuntimeError("cube-only gtc 초기화 실패: 그리퍼캠 큐브 관측이 없음")
    T, diag = cp.robust_se3_average(cands, None)
    return T, diag


# ------------------------------------------------------------------ 통합(unified)
def solve_unified(data, fk_mode, gtc_init, board_init):
    cam_init, grasp_init = data["cam_init"], data["grasp_init"]
    K_map, D_map = data["K_map"], data["D_map"]
    obs_s2 = data["obs_s2_fixed"] + data["obs_s2_gripper"]
    set_ids = sorted(data["items_by_index"])
    robot_T = {**data["robot_T_s1"], **data["robot_T_s2_gripper"], **data["robot_T_s3"]}
    observations = data["obs_s1"] + obs_s2 + data.get("obs_s2_board", []) + data["obs_s3"]
    has_board = any(o.marker == "board" for o in observations)

    if fk_mode == "no_fk":
        cubes = init_cube_poses(obs_s2, K_map, D_map, cam_init, gtc_init, robot_T, GRIPPER_LOCAL_ID, set_ids)
        cube_key = ["T_base_cube_by_set"]
    else:
        cubes = fk_anchor_cubes(data["items_by_index"], grasp_init)
        cube_key = []
    observations = [o for o in observations
                    if o.set_idx is None or int(o.set_idx) in cubes or o.grasp_idx is not None]

    state = PoseState(cams=dict(cam_init), gtc=gtc_init.copy(),
                      board=(board_init.copy() if (has_board and board_init is not None) else None),
                      cubes=dict(cubes), grasps={0: grasp_init.copy()})
    keys = variable_keys(["T_base_Ci", "T_gripper_cam"] + (["T_base_board"] if has_board else [])
                         + cube_key + ["T_gripper_cube_by_grasp"], state)
    final_state, diag = solve_corner_reprojection(
        observations=observations, variable_keys_=keys, reference_state=state,
        robot_T=robot_T, K_map=K_map, D_map=D_map,
        gripper_cam_idx=GRIPPER_LOCAL_ID, options=SolverOptions(),
    )
    return final_state, diag, len(observations)


# --------------------------------------------------------- 독립(parallel, 최종 합의)
#
# "독립"의 정의: 고정캠
# 그룹과 그리퍼 그룹이 서로 residual을 전혀 공유하지 않고 **완전히 따로**
# 풀고(한쪽이 다른 쪽에 값을 넘겨주는 handoff가 아예 없음), 맨 마지막에
# 두 그룹이 같은 session2 세트에 대해 각자 계산한 큐브 pose를 서로 비교해서
# "합의(consensus)"를 본다. Simulation/core/methods.py의 solve_independent +
# _rigid_align이 원래 하던 방식과 같은 개념.
#
# Zeus는 두 그룹 다 로봇 FK로 base 좌표계에 이미 묶여 있다(그룹A는 session1의
# grasp+FK, 그룹B는 session3의 board eye-in-hand+FK) -- 그래서 Simulation 쪽처럼
# 임의의 gauge를 맞추는 rigid-align이 필요 없고, 그냥 두 그룹의 큐브 pose를
# 직접 비교하면 된다(같은 좌표계이므로). 이 비교 자체가 "완전히 독립적으로
# 계산한 두 답이 서로 얼마나 맞는가"라는 유의미한 검증 지표가 된다.
def init_board_pose(obs_board, K_map, D_map, cam_init):
    """고정캠들의 board 관측치만으로 T_base_board 초기값 (그리퍼 정보 전혀 안 씀)."""
    cands = []
    for o in obs_board:
        c = int(o.cam)
        if c not in cam_init:
            continue
        T_cam_board = solve_observed_pose(o, K_map, D_map)
        if T_cam_board is not None:
            cands.append(cam_init[c] @ T_cam_board)
    if not cands:
        return np.eye(4)
    return cands[0] if len(cands) == 1 else cp.robust_se3_average(cands, None)[0]


def solve_parallel_fixed(data):
    """고정캠 그룹: session1(grasp+FK) + session2-고정캠 + session3-고정캠(보드),
    그리퍼 정보 전혀 안 씀. 고정캠도 session3 보드를 잘 본다(실측 확인)."""
    cam_init, grasp_init = data["cam_init"], data["grasp_init"]
    K_map, D_map = data["K_map"], data["D_map"]
    obs_s2 = data["obs_s2_fixed"]
    obs_s3f = data["obs_s3_fixed"]
    set_ids = sorted(data["items_by_index"])
    cubes = init_cube_poses(obs_s2, K_map, D_map, cam_init, np.eye(4), {}, -999, set_ids)
    board_obs = data.get("obs_s2_board_fixed", []) + obs_s3f
    board_init_fixed = init_board_pose(board_obs, K_map, D_map, cam_init) if board_obs else None
    observations = (data["obs_s1"]
                    + [o for o in obs_s2 if o.set_idx is not None and int(o.set_idx) in cubes]
                    + board_obs)
    state = PoseState(cams=dict(cam_init), gtc=np.eye(4), board=board_init_fixed,
                      cubes=dict(cubes), grasps={0: grasp_init.copy()})
    keys = variable_keys(["T_base_Ci", "T_base_cube_by_set", "T_gripper_cube_by_grasp"]
                         + (["T_base_board"] if board_obs else []), state)
    final_state, diag = solve_corner_reprojection(
        observations=observations, variable_keys_=keys, reference_state=state,
        robot_T={**data["robot_T_s1"], **data["robot_T_s3"]}, K_map=K_map, D_map=D_map,
        gripper_cam_idx=-999, options=SolverOptions(),
    )
    return final_state, diag, observations


def solve_parallel_gripper(data, gtc_init, board_init):
    """그리퍼 그룹: session2-그리퍼캠 + session3-그리퍼캠만, 고정캠 정보 전혀 안 씀."""
    K_map, D_map = data["K_map"], data["D_map"]
    obs_s2g = data["obs_s2_gripper"]
    obs_s3g = data["obs_s3_gripper"]
    set_ids = sorted(data["items_by_index"])
    robot_T = {**data["robot_T_s2_gripper"], **data["robot_T_s3"]}
    cubes = init_cube_poses(obs_s2g, K_map, D_map, {}, gtc_init, robot_T, GRIPPER_LOCAL_ID, set_ids)
    observations = ([o for o in obs_s2g if o.set_idx is not None and int(o.set_idx) in cubes]
                    + data.get("obs_s2_board_gripper", []) + obs_s3g)
    state = PoseState(cams={}, gtc=gtc_init.copy(), board=board_init.copy(), cubes=dict(cubes), grasps={})
    keys = variable_keys(["T_gripper_cam", "T_base_board", "T_base_cube_by_set"], state)
    final_state, diag = solve_corner_reprojection(
        observations=observations, variable_keys_=keys, reference_state=state,
        robot_T=robot_T, K_map=K_map, D_map=D_map,
        gripper_cam_idx=GRIPPER_LOCAL_ID, options=SolverOptions(),
    )
    return final_state, diag, observations


def consensus_check(state_fixed, state_gripper):
    """두 그룹이 완전히 독립적으로 계산한 값들을 직접 비교(같은 base 좌표계라
    rigid-align 불필요): session2 큐브(세트별) + session3 보드(전체 하나)."""
    common = sorted(set(state_fixed.cubes) & set(state_gripper.cubes))
    per_set = {}
    for s in common:
        d_mm, d_deg = pose_delta(state_fixed.cubes[s], state_gripper.cubes[s])
        per_set[s] = {"translation_mm": d_mm, "rotation_deg": d_deg}
    board_mm, board_deg = None, None
    if state_fixed.board is not None and state_gripper.board is not None:
        board_mm, board_deg = pose_delta(state_fixed.board, state_gripper.board)
    return per_set, {"translation_mm": board_mm, "rotation_deg": board_deg}


def per_corner_errors(state, observations, robot_T, K_map, D_map, gripper_id):
    """(dx, dy) 성분별 잔차를 펴서 반환 -- solve_corner_reprojection의
    train_reprojection_rmse_px(=sqrt(mean(raw_residual**2)), 코너별 유클리드
    거리가 아니라 x/y 성분 각각을 표본으로 취급)와 정의를 맞추기 위함.
    코너별 유클리드 거리로 RMS를 내면 등방 오차 기준 sqrt(2)배 부풀려진다
    (실제로 이 차이 때문에 raw-fk에서 통합/독립 RMSE가 다르게 나온 적 있음)."""
    errs = []
    for o in observations:
        if o.marker == "board":
            target = state.board
        elif o.grasp_idx is not None:
            target = robot_T[int(o.event)] @ state.grasps[int(o.grasp_idx)]
        else:
            target = state.cubes[int(o.set_idx)]
        if int(o.cam) == gripper_id:
            T_base_cam = robot_T[int(o.event)] @ state.gtc
        else:
            T_base_cam = state.cams[int(o.cam)]
        pred = project_points(inv_T(T_base_cam) @ target, o.object_points, K_map[int(o.cam)], D_map[int(o.cam)])
        errs.extend((pred - o.image_points).reshape(-1).tolist())
    return errs


def main_cube_only(args, data):
    """보드 없이(=session3 미사용, session2 보드 관측 제외) 큐브만으로 통합 2조건.
    독립_no-fk는 이 조건에서 정의 불가: 그리퍼 그룹 관측이 파킹 자세 1개에서 찍은
    큐브 15장뿐이고 큐브 pose가 자유 변수라, T_gripper_cam이 어떤 값이든 큐브
    pose가 흡수해서 식별이 안 된다(보드가 그 역할을 하고 있었음). 통합은 큐브가
    고정캠과 공유 변수라 식별된다."""
    data = strip_board_data(data)
    K_map, D_map = data["K_map"], data["D_map"]
    gtc_init, diag = init_gtc_from_cubes(data)
    print(f"[cube-only] T_gripper_cam 초기값 (session2 큐브만): t_mm={np.round(gtc_init[:3,3]*1000,2).tolist()} "
          f"(n={diag['num_total']}, inlier={diag['num_inliers']}, std={diag['translation_std_mm']:.2f}mm)\n")
    fit_out_dir = REPO_ROOT / "zeus_gello_calibration"
    results = {}
    for fk_mode, label in (("no_fk", "통합_no-fk_cubeonly"), ("fixed_fk", "통합_raw-fk_cubeonly")):
        state, diag, n_obs = solve_unified(data, fk_mode, gtc_init, None)
        results[label] = {"success": diag["success"], "rmse_px": diag["train_reprojection_rmse_px"],
                          "n_corners": diag["n_residuals"] // 2, "n_observations": n_obs}
        export_fit_json(fit_out_dir / f"fit_{label}{FIT_SUFFIX}.json", state.grasps[0], state.cams, state.gtc)
    results["독립_no-fk_cubeonly"] = {"success": False, "reason": "gripper group not identifiable without board (cube poses free, single parking view)"}
    total_n = len(data["obs_s1"]) + len(data["obs_s2_fixed"]) + len(data["obs_s2_gripper"])
    print(f"{'condition':>24} {'rmse_px':>10} {'n_corners':>10} {'n_obs':>7}")
    for name, r in results.items():
        if "rmse_px" in r:
            print(f"{name:>24} {r['rmse_px']:>10.4f} {r['n_corners']:>10d} {r['n_observations']:>7d}")
        else:
            print(f"{name:>24} {'N/A':>10}   ({r['reason']})")
    print(f"\n(cube-only 데이터 풀 총 observation 수 = {total_n})")
    out_path = Path(args.out).with_name(Path(args.out).stem + "_cubeonly.json")
    out_path.write_text(json.dumps({"cube_only": True, "total_observations": total_n, "results": results}, indent=2, default=str))
    print(f"wrote {out_path}")


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
    ap.add_argument("--fit-json", default=str(FIT_JSON_DEFAULT))
    ap.add_argument("--fixed-min-corners", type=int, default=8)
    ap.add_argument("--cube-observation-policy", default="legacy", choices=("legacy", "core_multiface"))
    ap.add_argument("--out", default=str(REPO_ROOT / "zeus_gello_calibration" / "calibration_methods_comparison.json"))
    ap.add_argument("--cube-only", action="store_true",
                    help="보드 관측 전부 제외(session3 미사용 + session2 보드 제외). late_table1 B2(-board) 대응")
    ap.add_argument("--tag", default="", help="출력 파일 접미사 (예: _0914 -> fit_통합_no-fk_0914.json). 다른 촬영분 결과를 덮어쓰지 않게")
    ap.add_argument("--cube-config", default=None,
                    help="큐브 마커 config JSON (기본: config.py 메인 큐브). GT 큐브로 찍은 촬영이면 targets/gt_cube/cube_config.json")
    ap.add_argument("--s3-gripper-only", action="store_true", help="session3는 그리퍼캠 관측만 사용 (고정캠 보드 관측 제외)")
    args = ap.parse_args()
    global FIT_SUFFIX
    FIT_SUFFIX = args.tag
    if args.tag and args.out == str(REPO_ROOT / "zeus_gello_calibration" / "calibration_methods_comparison.json"):
        args.out = str(REPO_ROOT / "zeus_gello_calibration" / f"calibration_methods_comparison{args.tag}.json")

    data = load_all_data(args)
    K_map, D_map = data["K_map"], data["D_map"]

    if args.cube_only:
        return main_cube_only(args, data)

    gtc_init, board_init, eih_diag = estimate_board_handeye_initial(
        data["obs_s3"], data["robot_T_s3"], K_map, D_map, GRIPPER_LOCAL_ID)
    print(f"T_gripper_cam 초기값 (session3만): t_mm={np.round(gtc_init[:3,3]*1000,2).tolist()} ({eih_diag})\n")

    fit_out_dir = REPO_ROOT / "zeus_gello_calibration"
    results = {}
    print("각 방식의 T_gripper_cube/T_base_cam을 gt_pick_test.py용 JSON으로 저장:")
    for fk_mode, label in (("no_fk", "통합_no-fk"), ("fixed_fk", "통합_raw-fk")):
        state, diag, n_obs = solve_unified(data, fk_mode, gtc_init, board_init)
        results[label] = {"success": diag["success"], "rmse_px": diag["train_reprojection_rmse_px"],
                          "n_corners": diag["n_residuals"] // 2, "n_observations": n_obs}
        export_fit_json(fit_out_dir / f"fit_{label}{FIT_SUFFIX}.json", state.grasps[0], state.cams, state.gtc)

    # 진짜 독립: 고정캠 그룹과 그리퍼 그룹을 서로 정보 교환 없이 완전히 따로
    # 캘리브레이션한다 (고정캠은 session1 grasp+FK로, 그리퍼는 session3 board
    # eye-in-hand+FK로 각자 로봇 FK에 연결). session2 큐브는 양쪽 다 학습에
    # 쓰되 큐브 pose 변수를 공유하지 않는다 -- 각 그룹이 별개의 자유 변수로
    # 따로 추정하고, 다 끝난 뒤 그 두 추정치가 얼마나 일치하는가를 사후
    # 검증(합의)으로 본다. 결과를 바꾸지 않는 순수 진단 지표.
    label2 = "독립_no-fk (진짜 독립, 큐브 pose 비공유)"
    state_fixed, diag_fixed, obs_fixed = solve_parallel_fixed(data)
    state_gripper, diag_gripper, obs_gripper = solve_parallel_gripper(data, gtc_init, board_init)
    export_fit_json(fit_out_dir / f"fit_독립_no-fk{FIT_SUFFIX}.json", state_fixed.grasps[0], state_fixed.cams, state_gripper.gtc)
    errs_fixed = per_corner_errors(state_fixed, obs_fixed, data["robot_T_s1"], K_map, D_map, -999)
    errs_gripper = per_corner_errors(state_gripper, obs_gripper,
                                     {**data["robot_T_s2_gripper"], **data["robot_T_s3"]},
                                     K_map, D_map, GRIPPER_LOCAL_ID)
    agreement, board_agreement = consensus_check(state_fixed, state_gripper)
    agree_mm = [v["translation_mm"] for v in agreement.values()]
    agree_deg = [v["rotation_deg"] for v in agreement.values()]
    results[label2] = {
        "success": bool(diag_fixed["success"] and diag_gripper["success"]),
        "fixed_group_rmse_px": rmse_px(errs_fixed),
        "gripper_group_rmse_px": rmse_px(errs_gripper),
        "n_observations": len(obs_fixed) + len(obs_gripper),
        "consensus_per_set": agreement,
        "consensus_mean_mm": float(np.mean(agree_mm)) if agree_mm else float("nan"),
        "consensus_mean_deg": float(np.mean(agree_deg)) if agree_deg else float("nan"),
        "consensus_max_mm": float(np.max(agree_mm)) if agree_mm else float("nan"),
        "consensus_max_deg": float(np.max(agree_deg)) if agree_deg else float("nan"),
        "consensus_board_mm": board_agreement["translation_mm"],
        "consensus_board_deg": board_agreement["rotation_deg"],
    }

    total_n = (len(data["obs_s1"]) + len(data["obs_s2_fixed"]) + len(data["obs_s2_gripper"])
              + len(data.get("obs_s2_board", [])) + len(data["obs_s3_fixed"]) + len(data["obs_s3_gripper"]))
    print(f"{'condition':>34} {'rmse_px':>10} {'n_corners':>10} {'n_obs':>7}   detail")
    for name in ("통합_raw-fk", "통합_no-fk"):
        r = results[name]
        print(f"{name:>34} {r['rmse_px']:>10.4f} {r['n_corners']:>10d} {r['n_observations']:>7d}   ")
    r2 = results[label2]
    print(f"{label2:>34} {'고정캠:'+format(r2['fixed_group_rmse_px'],'.4f'):>10} {'-':>10} "
          f"{r2['n_observations']:>7d}   그리퍼:{r2['gripper_group_rmse_px']:.4f}")
    print(f"  -> session2 큐브 사후 합의: 평균 {r2['consensus_mean_mm']:.2f}mm/{r2['consensus_mean_deg']:.2f}deg, "
          f"최대 {r2['consensus_max_mm']:.2f}mm/{r2['consensus_max_deg']:.2f}deg")
    print(f"  -> session3 보드 사후 합의: {r2['consensus_board_mm']:.2f}mm/{r2['consensus_board_deg']:.2f}deg")
    print(f"\n(참고: 데이터 풀 총 observation 수 = {total_n})")
    print("\n주의: raw-fk(통합_raw-fk)는 table1.py A3와 이름만 같지 실제로는 다른 조건입니다 --")
    print("A3는 비전 개입 0인 순수 기계적 상수를 쓰는데, 여긴 session1 비전 fit값(T_gripper_cube)을 씁니다.")
    print("Zeus엔 A3에 해당하는 독립 측정된 기계적 상수가 없어서 진짜 A3 재현은 지금 불가능합니다.")

    out = {
        "warning": ("train-pooled, held-out 분리 없음 -- table1.py 정식 지표 아님. "
                    "'통합_raw-fk'는 이름만 raw-fk고 A3의 정의(비전 개입 0)를 만족하지 않음 -- "
                    "session1에서 비전으로 fit한 T_gripper_cube를 앵커로 쓰기 때문."),
        "total_observations": total_n,
        "results": results,
    }
    Path(args.out).write_text(json.dumps(out, indent=2, default=lambda o: str(o)))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
