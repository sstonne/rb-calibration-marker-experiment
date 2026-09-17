#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zeus_gello_calibration/table1_zeus.py -- Zeus 데이터로 Table 1을 계산한다.

핵심 차이: A3는 nominal 기계값, A5는 P1 VISION corrected-FK를 쓴다
----------------------------------------------------------------
``pass1_grasp_offset_replayed.json``의 ``T_gripper_cube``는 P1 Cube 영상과
Robot FK를 함께 사용해 적합한 train VISION artifact다. P1은 Cube를 한 번
강체 파지한 채 공중에서 위치와 회전을 바꾼 16개 pose이며, 이 운동 다양성으로
거리, 축 관계와 실제 장착 오차를 함께 식별한다. 따라서

    T^B_cube(s) = T^B_flange(place_s) · T^flange_cube,P1-VISION

은 corrected-FK를 hard fixed하는 A5에 해당한다. A3는 영상값 대신 사용자 확정
nominal 조립 가정인 flange-to-top datum 97.5mm와 Cube object origin-to-top plane
62.5mm를 합친 160.0mm, 그리고 이상적인 Ry(180deg)를 hard fixed한다.

수식이 달라지는 지점
--------------------
1. A4/B1/B2는 P1 VISION artifact를 corrected-FK soft factor 중심으로 사용한다.
2. A5는 같은 corrected-FK pose를 hard fixed한다.
3. A3는 ``[0, 0, 160.0mm] + Ry(180deg)`` nominal 기계 변환을 hard fixed한다.
4. External GT는 현재 비어 있으며 결과 JSON/Markdown에 Pending으로 기록한다.
5. P1의 단일 파지 다중 회전 fit은 A5 학습에만 쓰며 A3 기계값으로 재사용하지 않는다.

목적함수는 late_table1과 동일하다. 잔차는 corner 재투영 하나뿐이다.

    r_k = π( K_c, D_c, (T^B_Cc(e))⁻¹ · T^B_O , X_k ) − x_k          [px]
    고정캠 i : T^B_Cc = T^B_Ci
    그리퍼캠 g: T^B_Cc(e) = T^B_G(e) · T^G_Cg
    session1(큐브를 쥔 상태): T^B_cube(e) = T^B_G(e) · T^gripper_cube  (grasp 모델)

평가지표
--------
Cube 재투영은 모든 row에서 같은 FK reference pose를 사용한다. 따라서 ALL/Train/
Held-out Test를 같은 수식으로 비교할 수 있지만, FK를 사용하는 row에 구조적으로
유리하므로 내부 보조 지표다.

Cross-view는 source camera 한 대의 PnP pose만 destination camera로 전달한다.
Destination 관측은 오직 재투영 오차 계산에만 사용한다. ALL은 full-data fit 진단,
Train은 fold별 in-sample 진단, Held-out Test는 leave-one-placement-out 내부 일반화
지표다. 모든 pixel 값은 코너별 sqrt(mean(dx² + dy²))로 통일하며,
placement와 fold를 합칠 때도 평가한 코너 수로 가중한다.

최종 순위는 calibration/FK와 독립적으로 측정한 External GT의 TRE(mm), rotation
error(deg), P95 TRE와 failure rate로 결정한다.

사용법:
  python zeus_gello_calibration/table1_zeus.py --report-dir <결과폴더>
  python zeus_gello_calibration/table1_zeus.py --rows A2,A3 --folds 5
  # 저장된 JSON으로 보고서만 다시 만들기 (재fit 없음, 결과 JSON 폴더에 저장)
  python zeus_gello_calibration/table1_zeus.py --report-only <결과폴더>/ABLATION_TEST_table1_methods.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibration_pipeline.fk_factor import (  # noqa: E402
    FKFactorSpec, FK_MODE_FACTOR, FK_MODE_NONE, diagonal_covariance,
    solve_factorized_fk,
)
from calibration_pipeline.path_evaluation import solve_observed_pose  # noqa: E402
from calibration_pipeline.reprojection import (  # noqa: E402
    PoseState, SolverOptions, project_points,
    solve_corner_reprojection, variable_keys,
)
from calibration_pipeline.apriltag_cube import inv_T  # noqa: E402
from calibration_pipeline.evaluation import serialize_state  # noqa: E402
from calibration_pipeline.config import TOP_MARKER_PLANE_Z_M  # noqa: E402
from calibration_pipeline.table1 import estimate_board_handeye_initial  # noqa: E402

# fit_calibration_methods 는 import 사슬로 capture_session -> pyrealsense2 를
# 끌고 온다.  이 파일은 저장된 영상만 읽어 적합하므로 카메라 SDK 가 필요 없다.
# 하드웨어 없는 환경에서 돌리기 위해 없을 때만 빈 모듈로 대체한다.
try:  # pragma: no cover - 환경에 따라 갈림
    import pyrealsense2  # noqa: F401
except ModuleNotFoundError:
    import types
    sys.modules["pyrealsense2"] = types.ModuleType("pyrealsense2")

import fit_calibration_methods as fcm  # noqa: E402
from paths import ZEUS_DATA_ROOT, require_zeus_data_path  # noqa: E402

GRIPPER = fcm.GRIPPER_LOCAL_ID

# late_table1의 A4가 쓰는 preflight 사전값과 같은 크기.  측정된 로봇 공분산이
# 아니므로 A4는 확정 근거가 아니라 민감도 점검으로만 읽는다.
SIGMA_FK_MM, SIGMA_FK_DEG = 2.0, 0.30

# A3 nominal FK.  tool1=0으로 저장된 pose는 T_base_flange다. 사용자가 확정한
# 중앙 파지 조립 가정에 따라 flange-to-Cube-top datum 97.5mm와 Cube 모델의
# origin-to-top plane 62.5mm를 합친다. 97.5mm는 물리 실측값이 아니라 nominal
# hardware datum이므로 결과 provenance에서 measured=False로 명시한다.
ZEUS_FLANGE_TO_CUBE_TOP_DATUM_MM = 97.5
CUBE_ORIGIN_TO_TOP_PLANE_MM = TOP_MARKER_PLANE_Z_M * 1000.0
MECHANICAL_FLANGE_CUBE_DISTANCE_MM = (
    ZEUS_FLANGE_TO_CUBE_TOP_DATUM_MM + CUBE_ORIGIN_TO_TOP_PLANE_MM
)


def mechanical_flange_cube_transform():
    """Nominal A3 T_flange_cube: centered translation and ideal Ry(180deg)."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.diag([-1.0, 1.0, -1.0])
    transform[:3, 3] = [0.0, 0.0, MECHANICAL_FLANGE_CUBE_DISTANCE_MM / 1000.0]
    return transform


def load_mechanical_transform(path):
    """Load a separately specified, vision-free flange-to-object transform."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("vision_used") is not False or not payload.get("source"):
        raise ValueError("mechanical transform requires vision_used=false and source")
    transform = np.asarray(payload["T_flange_cube"], dtype=np.float64)
    if (transform.shape != (4, 4) or not np.all(np.isfinite(transform))
            or not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1.0)):
        raise ValueError("mechanical T_flange_cube must be SE(3), translation in metres")
    return transform, payload


# ---------------------------------------------------------------- row 정의
# targets : calibration에 넣는 표적.  cube를 빼면 그 row는 큐브를 평가에만 쓴다.
# opt     : "uni" = 하나의 최소제곱, "seq" = eye-in-hand 먼저 풀고 동결 후 고정캠.
# fk      : "none"              큐브 자세 자유 (VISION)
#           "corrected_fixed"   P1 VISION corrected-FK hard fixed
#           "corrected_factor"  큐브 자유 + corrected-FK soft factor
#           "mechanical_fixed"  nominal 160mm + Ry(180deg) FK hard fixed
ROWS = {
    "A0": dict(label="board-only, sequential VISION", targets=("board",), opt="seq", fk="none"),
    "A1": dict(label="+cube, sequential", targets=("board", "cube"), opt="seq", fk="none"),
    "A2": dict(label="+unified, VISION", targets=("board", "cube"), opt="uni", fk="none"),
    "A3": dict(
        label="FK hard fixed (nominal 160mm)", targets=("board", "cube"),
        opt="uni", fk="mechanical_fixed",
    ),
    "A4": dict(label="corrected-FK soft factor", targets=("board", "cube"), opt="uni", fk="corrected_factor"),
    "A5": dict(label="corrected-FK hard fixed (P1 VISION-aligned)", targets=("board", "cube"), opt="uni", fk="corrected_fixed"),
    "B1": dict(label="-Unified (corrected-FK soft factor, sequential)", targets=("board", "cube"), opt="seq", fk="corrected_factor"),
    "B2": dict(label="-board (cube only, corrected-FK soft factor)", targets=("cube",), opt="uni", fk="corrected_factor"),
    "B3": dict(label="-cube (board only, unified)", targets=("board",), opt="uni", fk="none"),
}
ROW_ORDER = ("A0", "A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3")
SCHEMA = "table1_zeus_cube_and_cross_view_v4"


def fk_cube_poses(items_by_index, T_flange_cube):
    """Apply one frozen nominal or corrected flange-to-Cube transform."""
    return {int(s): fcm.pose6_to_T(item["target"]) @ T_flange_cube
            for s, item in items_by_index.items()}


def split_observations(data, targets, drop_set=None):
    """row가 쓰는 관측을 고른다.  drop_set은 held-out placement를 뺀다."""
    cube_obs, board_obs = [], []
    if "cube" in targets:
        cube_obs = list(data["obs_s1"])  # 쥔 큐브 (grasp 모델, placement 아님)
        # session3 큐브(마지막 placement 자리에 정지)는 그 세트 변수를 공유하므로
        # drop_set 규칙이 그대로 적용된다 (held-out 채점에는 안 들어감).
        for o in data["obs_s2_fixed"] + data["obs_s2_gripper"] + data.get("obs_s3_cube", []):
            if drop_set is not None and o.set_idx is not None and int(o.set_idx) == int(drop_set):
                continue
            cube_obs.append(o)
    if "board" in targets:
        board_obs = list(data["obs_s3"])
        if data.get("include_session2_board", False):
            board_obs.extend(
                observation for observation in data.get("obs_s2_board", [])
                if drop_set is None
                or int(observation.event) != fcm.SESSION2_EVENT_OFFSET + int(drop_set)
            )
    return cube_obs, board_obs


def held_out_observations(data, set_index):
    return [o for o in data["obs_s2_fixed"] + data["obs_s2_gripper"]
            if o.set_idx is not None and int(o.set_idx) == int(set_index)]


def make_state(data, cubes, grasp_init, board_init, gtc_init):
    return PoseState(cams=dict(data["cam_init"]), gtc=gtc_init.copy(),
                     board=board_init.copy(), cubes=dict(cubes),
                     grasps={0: grasp_init.copy()})


def grasp_variable_families(spec):
    if "cube" not in spec["targets"] or spec["fk"] in (
        "mechanical_fixed", "corrected_fixed"
    ):
        return []
    return ["T_gripper_cube_by_grasp"]


def fit_row(row, data, drop_set, robot_T, board_init, gtc_init):
    """한 row를 적합한다.  반환: (final_state, train_rmse_px, n_obs, ok)."""
    spec = ROWS[row]
    if not spec.get("available", True):
        raise ValueError(f"{row} is pending: {spec['pending_reason']}")
    corrected_grasp = data["grasp_init"]
    grasp_init = (
        data.get("mechanical_grasp", mechanical_flange_cube_transform())
        if spec["fk"] == "mechanical_fixed"
        else corrected_grasp
    )
    cube_obs, board_obs = split_observations(data, spec["targets"], drop_set)
    K_map, D_map = data["K_map"], data["D_map"]
    options = SolverOptions()

    # 큐브 자세의 출처
    if "cube" not in spec["targets"]:
        cubes, cube_key = {}, []
    elif spec["fk"] in ("mechanical_fixed", "corrected_fixed"):
        cubes = fk_cube_poses(data["items_by_index"], grasp_init)
        cube_key = []                      # 자유도 없음 -- 하드 고정
    else:
        set_ids = sorted(data["items_by_index"])
        cubes = fcm.init_cube_poses(cube_obs, K_map, D_map, data["cam_init"],
                                    gtc_init, robot_T, GRIPPER, set_ids)
        cube_key = ["T_base_cube_by_set"]

    observations = cube_obs + board_obs
    if "cube" in spec["targets"]:
        observations = [o for o in observations
                        if o.set_idx is None or int(o.set_idx) in cubes or o.grasp_idx is not None]
    if not observations:
        return None, float("nan"), 0, False

    board_families = ["T_base_board"] if "board" in spec["targets"] else []
    grasp_families = grasp_variable_families(spec)
    state = make_state(data, cubes, grasp_init, board_init, gtc_init)

    if spec["opt"] == "uni":
        families = ["T_base_Ci", "T_gripper_cam"] + board_families + cube_key + grasp_families
        keys = variable_keys(families, state)
        if spec["fk"] == "corrected_factor":
            cov = diagonal_covariance(SIGMA_FK_MM, SIGMA_FK_DEG)
            targets_fk = fk_cube_poses(data["items_by_index"], corrected_grasp)
            final, diag = solve_factorized_fk(
                observations=observations, variable_keys_=keys, reference_state=state,
                robot_T=robot_T, K_map=K_map, D_map=D_map, gripper_cam_idx=GRIPPER,
                options=options,
                fk_targets={s: t for s, t in targets_fk.items() if s in cubes},
                fk_covariances={s: cov for s in cubes},
                fk_spec=FKFactorSpec(mode=FK_MODE_FACTOR, loss="huber", robust_scale=3.0))
        else:
            final, diag = solve_corner_reprojection(
                observations=observations, variable_keys_=keys, reference_state=state,
                robot_T=robot_T, K_map=K_map, D_map=D_map, gripper_cam_idx=GRIPPER,
                options=options)
        ok = bool(diag.get("success", False))
    else:
        # sequential: eye-in-hand 먼저 (그리퍼캠 관측), 그 결과를 동결하고 고정캠.
        eih = [o for o in observations if int(o.cam) == GRIPPER]
        e2h = [o for o in observations if int(o.cam) != GRIPPER]
        if not eih or not e2h:
            return None, float("nan"), 0, False
        keys1 = variable_keys(["T_gripper_cam"] + board_families + cube_key, state)
        if spec["fk"] == "corrected_factor":
            # B1: soft FK factor 는 큐브 자세가 자유변수인 stage1 에 건다.
            # stage2 는 큐브를 동결하므로 FK 항이 걸릴 자유도가 없다.
            cov = diagonal_covariance(SIGMA_FK_MM, SIGMA_FK_DEG)
            targets_fk = fk_cube_poses(data["items_by_index"], corrected_grasp)
            stage1, d1 = solve_factorized_fk(
                observations=eih, variable_keys_=keys1, reference_state=state,
                robot_T=robot_T, K_map=K_map, D_map=D_map, gripper_cam_idx=GRIPPER,
                options=options,
                fk_targets={s: t for s, t in targets_fk.items() if s in cubes},
                fk_covariances={s: cov for s in cubes},
                fk_spec=FKFactorSpec(mode=FK_MODE_FACTOR, loss="huber", robust_scale=3.0))
        else:
            stage1, d1 = solve_corner_reprojection(
                observations=eih, variable_keys_=keys1, reference_state=state,
                robot_T=robot_T, K_map=K_map, D_map=D_map, gripper_cam_idx=GRIPPER, options=options)
        keys2 = variable_keys(["T_base_Ci"] + grasp_families, stage1)
        final, d2 = solve_corner_reprojection(
            observations=e2h, variable_keys_=keys2, reference_state=stage1,
            robot_T=robot_T, K_map=K_map, D_map=D_map, gripper_cam_idx=GRIPPER, options=options)
        ok = bool(d1.get("success", False) and d2.get("success", False))

    errs = fcm.per_corner_errors(final, observations, robot_T, K_map, D_map, GRIPPER)
    # Keep the optimizer unchanged; report its residuals with the same
    # two-dimensional corner-distance convention as the evaluation metrics.
    residuals = np.asarray(errs, dtype=np.float64).reshape(-1, 2)
    train_px = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1)))) if residuals.size else float("nan")
    return final, train_px, len(observations), ok


# ------------------------------------------------------------- 공통 평가
def session2_observations(data, *, exclude_set=None):
    observations = data["obs_s2_fixed"] + data["obs_s2_gripper"]
    if exclude_set is None:
        return list(observations)
    return [
        o for o in observations
        if o.set_idx is None or int(o.set_idx) != int(exclude_set)
    ]


def base_camera_pose(observation, state, robot_T):
    camera = int(observation.cam)
    if camera == GRIPPER:
        event = int(observation.event)
        if state.gtc is None or event not in robot_T:
            return None
        return np.asarray(robot_T[event]) @ np.asarray(state.gtc)
    if camera not in state.cams:
        return None
    return np.asarray(state.cams[camera])


def _support_bucket():
    return {
        "events": set(),
        "observation_ids": set(),
        "n_pairs": 0,
        "n_directions": 0,
        "n_corners": 0,
        "n_residual_components": 0,
    }


def _pixel_error_stats(squared_by_set, support_by_set):
    """Pool squared 2-D corner distances, retaining per-placement diagnostics."""
    per_set = []
    for set_index in sorted(squared_by_set):
        squared = np.asarray(squared_by_set[set_index], dtype=np.float64)
        if not squared.size:
            continue
        support = support_by_set[set_index]
        mse = float(np.mean(squared))
        per_set.append({
            "set": int(set_index),
            "mse_px2": mse,
            "rmse_px": float(np.sqrt(mse)),
            "sum_squared_error_px2": float(np.sum(squared)),
            "n_events": len(support["events"]),
            "n_observations": len(support["observation_ids"]),
            **{key: int(support[key]) for key in (
                "n_pairs", "n_directions", "n_corners",
                "n_residual_components",
            )},
        })

    n_corners = sum(row["n_corners"] for row in per_set)
    squared_sum = float(sum(row["sum_squared_error_px2"] for row in per_set))
    mse = squared_sum / n_corners if n_corners else float("nan")
    return {
        "rmse_px": float(np.sqrt(mse)),
        "mse_px2": mse,
        "sum_squared_error_px2": squared_sum,
        "aggregation": "pooled_corner_squared_distance_mean_then_sqrt",
        "pixel_rmse_definition": "sqrt(mean(dx^2 + dy^2))",
        "n_sets": len(per_set),
        "n_events": sum(row["n_events"] for row in per_set),
        **{key: sum(row[key] for row in per_set) for key in (
            "n_observations", "n_pairs", "n_directions", "n_corners",
            "n_residual_components",
        )},
        "per_set": per_set,
    }


def cube_reprojection_stats(obs_list, cube_poses, state, robot_T, K_map, D_map):
    """Reproject placement cubes from one common, method-independent pose map."""
    squared_by_set = defaultdict(list)
    support_by_set = defaultdict(_support_bucket)
    for observation in obs_list:
        if observation.set_idx is None:
            continue
        set_index = int(observation.set_idx)
        if set_index not in cube_poses:
            continue
        T_base_camera = base_camera_pose(observation, state, robot_T)
        if T_base_camera is None:
            continue
        camera = int(observation.cam)
        prediction = project_points(
            inv_T(T_base_camera) @ np.asarray(cube_poses[set_index]),
            observation.object_points,
            K_map[camera],
            D_map[camera],
        )
        measured = np.asarray(observation.image_points, dtype=np.float64).reshape(-1, 2)
        if prediction.shape != measured.shape or not np.all(np.isfinite(prediction)):
            continue
        squared = np.sum(np.square(prediction - measured), axis=1)
        squared_by_set[set_index].extend(squared.tolist())
        support = support_by_set[set_index]
        support["events"].add(int(observation.event))
        support["observation_ids"].add(
            (int(observation.event), int(observation.cam)))
        support["n_corners"] += len(measured)
        support["n_residual_components"] += 2 * len(squared)
    return _pixel_error_stats(squared_by_set, support_by_set)


def cross_view_transfer_stats(obs_list, state, robot_T, K_map, D_map):
    """Bidirectional source-only PnP transfer on the supplied event population."""
    by_event = defaultdict(dict)
    for observation in obs_list:
        if observation.set_idx is None:
            continue
        key = (int(observation.set_idx), int(observation.event))
        camera = int(observation.cam)
        if camera in by_event[key]:
            raise ValueError(
                f"duplicate cube observation for set={key[0]} event={key[1]} camera={camera}"
            )
        by_event[key][camera] = observation

    squared_by_type = {
        "overall": defaultdict(list),
        "fixed_fixed": defaultdict(list),
        "fixed_gripper": defaultdict(list),
    }
    support_by_type = {
        key: defaultdict(_support_bucket) for key in squared_by_type
    }

    for (set_index, event), camera_observations in sorted(by_event.items()):
        solved = {}
        for camera, observation in camera_observations.items():
            T_camera_cube = solve_observed_pose(observation, K_map, D_map)
            T_base_camera = base_camera_pose(observation, state, robot_T)
            if T_camera_cube is not None and T_base_camera is not None:
                solved[camera] = (observation, T_camera_cube, T_base_camera)

        for left, right in combinations(sorted(solved), 2):
            pair_type = "fixed_gripper" if GRIPPER in (left, right) else "fixed_fixed"
            pair_squared = []
            n_corners = 0
            valid = True
            for source, destination in ((left, right), (right, left)):
                _source_obs, T_source_cube, T_base_source = solved[source]
                destination_obs, _T_destination_cube_measured, T_base_destination = solved[destination]
                T_destination_cube = (
                    inv_T(T_base_destination) @ T_base_source @ T_source_cube
                )
                prediction = project_points(
                    T_destination_cube,
                    destination_obs.object_points,
                    K_map[destination],
                    D_map[destination],
                )
                measured = np.asarray(
                    destination_obs.image_points, dtype=np.float64).reshape(-1, 2)
                if prediction.shape != measured.shape or not np.all(np.isfinite(prediction)):
                    valid = False
                    break
                pair_squared.extend(np.sum(np.square(prediction - measured), axis=1).tolist())
                n_corners += len(measured)
            if not valid:
                continue

            for key in ("overall", pair_type):
                squared_by_type[key][set_index].extend(pair_squared)
                support = support_by_type[key][set_index]
                support["events"].add(event)
                support["observation_ids"].update(((event, left), (event, right)))
                support["n_pairs"] += 1
                support["n_directions"] += 2
                support["n_corners"] += n_corners
                support["n_residual_components"] += 2 * len(pair_squared)

    overall = _pixel_error_stats(
        squared_by_type["overall"], support_by_type["overall"])
    overall["by_pair_type"] = {
        pair_type: _pixel_error_stats(
            squared_by_type[pair_type], support_by_type[pair_type])
        for pair_type in ("fixed_fixed", "fixed_gripper")
    }
    overall["definition"] = (
        "source camera measurement-only PnP transferred to destination; "
        "destination observation used only for scoring; both directions"
    )
    return overall


def evaluate_fold(row, data, held_set, robot_T, board_init, gtc_init):
    final, solver_train_px, n_obs, ok = fit_row(
        row, data, held_set, robot_T, board_init, gtc_init)
    if final is None:
        return None
    K_map, D_map = data["K_map"], data["D_map"]
    obs_h = held_out_observations(data, held_set)
    if not obs_h:
        return None
    obs_train = session2_observations(data, exclude_set=held_set)
    cube_reference = fk_cube_poses(data["items_by_index"], data["grasp_init"])
    return {
        "set": int(held_set),
        "converged": ok,
        "transforms": serialize_state(final),
        "training_placement_ids": sorted(
            int(s) for s in data["items_by_index"] if int(s) != int(held_set)),
        "heldout_placement_ids": [int(held_set)],
        "n_solver_train_observations": n_obs,
        "solver_train_rmse_px": solver_train_px,
        "train": {
            "cube_reprojection": cube_reprojection_stats(
                obs_train, cube_reference, final, robot_T, K_map, D_map),
            "cross_view": cross_view_transfer_stats(
                obs_train, final, robot_T, K_map, D_map),
        },
        "heldout_test": {
            "cube_reprojection": cube_reprojection_stats(
                obs_h, cube_reference, final, robot_T, K_map, D_map),
            "cross_view": cross_view_transfer_stats(
                obs_h, final, robot_T, K_map, D_map),
        },
    }


def _fold_job(arguments):
    return evaluate_fold(*arguments)


def aggregate_fold_metric(folds, split, metric):
    valid = [
        fold[split][metric]
        for fold in folds if fold is not None
        and np.isfinite(fold[split][metric]["mse_px2"])
    ]
    def pooled_summary(items):
        n_corners = sum(item["n_corners"] for item in items)
        squared_sum = float(sum(item["sum_squared_error_px2"] for item in items))
        mse = squared_sum / n_corners if n_corners else float("nan")
        return {
            "rmse_px": float(np.sqrt(mse)),
            "mse_px2": mse,
            "sum_squared_error_px2": squared_sum,
            "n_corners": n_corners,
            "n_residual_components": 2 * n_corners,
            "aggregation": "pooled_fold_corner_squared_distance_mean_then_sqrt",
            "pixel_rmse_definition": "sqrt(mean(dx^2 + dy^2))",
            "n_fold_evaluations": len(items),
        }

    result = pooled_summary(valid)
    if metric == "cross_view":
        result["by_pair_type"] = {}
        for pair_type in ("fixed_fixed", "fixed_gripper"):
            values = [
                item["by_pair_type"][pair_type]
                for item in valid
                if np.isfinite(item["by_pair_type"][pair_type]["mse_px2"])
            ]
            result["by_pair_type"][pair_type] = pooled_summary(values)
    return result


def aggregate_folds(folds):
    valid = [fold for fold in folds if fold is not None]
    return {
        "n_folds": len(valid),
        "n_converged": sum(1 for fold in valid if fold["converged"]),
        "train_cube_reprojection": aggregate_fold_metric(
            valid, "train", "cube_reprojection"),
        "heldout_test_cube_reprojection": aggregate_fold_metric(
            valid, "heldout_test", "cube_reprojection"),
        "train_cross_view": aggregate_fold_metric(
            valid, "train", "cross_view"),
        "heldout_test_cross_view": aggregate_fold_metric(
            valid, "heldout_test", "cross_view"),
    }


def external_gt_pending():
    return {
        "status": "pending",
        "reason": "독립 6-DoF T_base_cube_GT와 각 방법의 frozen prediction 미확보; 기존 flange/yaw 진단은 대체 불가",
        "failure_definition": "missing_or_failed_predictions_over_all_independent_GT_poses",
        "mean_tre_mm": None,
        "median_tre_mm": None,
        "p95_tre_mm": None,
        "mean_rotation_error_deg": None,
        "p95_rotation_error_deg": None,
        "failure_rate": None,
    }


def save_result(result, path):
    """Keep JSON portable: missing metrics are null, never nonstandard NaN."""
    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(result), indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def empty_metric_summary():
    return {
        "all_cube_rmse_px": None,
        "train_cube_rmse_px": None,
        "heldout_test_cube_rmse_px": None,
        "all_cross_view_cube_rmse_px": None,
        "train_cross_view_cube_rmse_px": None,
        "heldout_test_cross_view_cube_rmse_px": None,
        "n_folds": 0,
        "n_converged": 0,
        "full_fit_converged": False,
    }


def corrected_fk_training_contract(fit_json_path):
    """Summarize the P1-only VISION fit that defines corrected-FK.

    P1 uses one rigid grasp (grasp_id=0) observed over multiple airborne robot
    poses.  It identifies one constant flange-to-Cube transform, including the
    real mounting offset.  Because Cube images participate in that estimate,
    the result belongs to corrected-FK and must never be reused as A3 FK.
    """
    fit_path = Path(fit_json_path)
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    transform = np.asarray(fit["T_gripper_cube"], dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"invalid T_gripper_cube shape in {fit_path}: {transform.shape}")

    nominal_rotation = np.diag([-1.0, 1.0, -1.0])  # Ry(180 deg)
    rotation_delta = nominal_rotation.T @ transform[:3, :3]
    cosine = np.clip((np.trace(rotation_delta) - 1.0) / 2.0, -1.0, 1.0)
    dispersion = fit.get("grasp_init_dispersion_across_cams", {})
    solve = fit.get("solve_diagnostics", {})
    return {
        "role": "P1 training artifact for initialization, A4/A5/B1/B2 corrected-FK and common Cube reference; forbidden as A3 mechanical FK",
        "source": str(fit_path.resolve()),
        "collection": "one rigid Cube grasp, multiple airborne positions and rotations",
        "robot_pose_frame": "T_base_flange from tool1=0",
        "grasp_model": "one constant T_flange_cube (grasp_id=0); regrasp repeatability not measured",
        "n_captures": int(fit.get("n_captures", 0)),
        "translation_mm": (transform[:3, 3] * 1000.0).tolist(),
        "nominal_axis_map": "Ry(180 deg): Cube +X=-flange +X, +Y=+Y, +Z=-Z",
        "rotation_deviation_from_nominal_deg": float(np.degrees(np.arccos(cosine))),
        "pnp_accepted_per_camera": fit.get("pnp_accepted_per_camera", {}),
        "cross_camera_initial_translation_std_mm": dispersion.get("translation_std_mm"),
        "cross_camera_initial_rotation_std_deg": dispersion.get("rotation_std_deg"),
        "train_reprojection_rmse_px": (
            float(solve["train_reprojection_rmse_px"]) * np.sqrt(2.0)
            if solve.get("train_reprojection_rmse_px") is not None else None
        ),
        "train_reprojection_rmse_definition": "sqrt(mean(dx^2 + dy^2))",
        "train_reprojection_source_conversion": "component-wise solver RMSE multiplied by sqrt(2)",
        "jacobian_rank_deficient": solve.get("jacobian_rank_deficient"),
    }


def mechanical_fk_contract():
    transform = mechanical_flange_cube_transform()
    return {
        "role": "A3 FK hard fixed only",
        "source": "nominal geometry: 97.5mm flange-to-top datum + CubeConfig top plane 62.5mm",
        "measured": False,
        "vision_used": False,
        "centered_grasp_assumption": True,
        "flange_to_cube_top_datum_mm": ZEUS_FLANGE_TO_CUBE_TOP_DATUM_MM,
        "cube_origin_to_top_plane_mm": CUBE_ORIGIN_TO_TOP_PLANE_MM,
        "translation_mm": (transform[:3, 3] * 1000.0).tolist(),
        "rotation": "Ry(180 deg)",
        "transform": transform.tolist(),
    }


# ------------------------------------------------------------- 보고서
# 저장된 v4 JSON만으로 Markdown·요약 CSV·fold CSV를 만든다 (재fit 없음).
DEFAULT_CONDITIONS = {
    "A0": ("board", "Sequential", "VISION"),
    "A1": ("board + cube", "Sequential", "VISION"),
    "A2": ("board + cube", "Unified", "VISION"),
    "A3": ("board + cube", "Unified", "raw-FK hard fixed"),
    "A4": ("board + cube", "Unified", "corrected-FK soft factor"),
    "A5": ("board + cube", "Unified", "corrected-FK hard fixed"),
    "B1": ("board + cube", "Sequential", "corrected-FK soft factor"),
    "B2": ("cube", "Unified", "corrected-FK soft factor"),
    "B3": ("board", "Unified", "VISION"),
}
PIXEL_COLUMNS = {
    "all_cube_rmse_px": "ALL Cube RMSE px (Train + Held-out Test)",
    "train_cube_rmse_px": "Train Cube RMSE px",
    "heldout_test_cube_rmse_px": "Held-out Test Cube RMSE px",
    "all_cross_view_cube_rmse_px": "ALL Cross-view Cube RMSE px (Train + Held-out Test)",
    "train_cross_view_cube_rmse_px": "Train Cross-view Cube RMSE px",
    "heldout_test_cross_view_cube_rmse_px": "Held-out Test Cross-view Cube RMSE px",
}
EXTERNAL_COLUMNS = {
    "mean_tre_mm": "TRE mm",
    "mean_rotation_error_deg": "Rotation Error deg",
    "p95_tre_mm": "P95 TRE mm",
    "failure_rate": "Failure Rate",
}
METRIC_COLUMNS = {**PIXEL_COLUMNS, **EXTERNAL_COLUMNS}
SUPPORT_COLUMNS = (
    "n_sets", "n_events", "n_observations", "n_pairs", "n_directions",
    "n_corners", "n_residual_components",
)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _metric(value):
    return float(value) if _finite(value) else "Pending"


def _portable(value):
    """Display repository paths without machine-specific prefixes."""
    if isinstance(value, dict):
        return {key: _portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        marker = "rb-calibration-marker-experiment/"
        if marker in normalized:
            return normalized.split(marker, 1)[1]
        if normalized.startswith("/") or re.match(r"^[A-Za-z]:/", normalized):
            return "외부 입력: " + normalized.rstrip("/").rsplit("/", 1)[-1]
    return value


def _display(value):
    value = _portable(value)
    if value is None:
        return "Pending"
    if isinstance(value, float):
        return f"{value:.4f}" if math.isfinite(value) else "Pending"
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False)
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _table(lines, headers, records):
    lines.extend(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"])
    lines.extend("| " + " | ".join(_display(value) for value in record) + " |" for record in records)
    lines.append("")


def _highlight_minimum(value, minimum):
    formatted = _display(value)
    return f"**{formatted}**" if _finite(value) and value == minimum else formatted


def _rows(result):
    """Include unavailable and unrequested baselines as explicit pending rows."""
    bodies = result.get("rows", {})
    pending = result.get("pending_rows", {})
    for row in ROW_ORDER:
        body = bodies.get(row, {})
        pending_info = pending.get(row, {})
        if isinstance(pending_info, str):
            pending_info = {"pending_reason": pending_info}
        reason = body.get("pending_reason") or pending_info.get("pending_reason") or pending_info.get("reason")
        if not body and not reason:
            reason = "이 결과 JSON에 실행 결과가 없음"
        status = body.get("status", "pending")
        yield row, body, status, reason or ""


def _conditions(row, body):
    condition = body.get("condition", {})
    target, optimization, fk = DEFAULT_CONDITIONS[row]
    if condition.get("targets"):
        target = " + ".join(condition["targets"])
    optimization = {"seq": "Sequential", "uni": "Unified"}.get(condition.get("opt"), optimization)
    fk = {
        "none": "VISION", "mechanical_fixed": DEFAULT_CONDITIONS["A3"][2],
        "corrected_factor": "corrected-FK soft factor",
        "corrected_fixed": "corrected-FK hard fixed",
    }.get(condition.get("fk"), fk)
    return target, optimization, fk


def _summary_records(result):
    records = []
    for row, body, status, reason in _rows(result):
        target, optimization, fk = _conditions(row, body)
        summary = body.get("summary", {})
        external = body.get("external_gt", {})
        record = {"row": row, "status": status, "targets": target,
                  "optimization": optimization, "fk": fk, "pending_reason": reason}
        for key in PIXEL_COLUMNS:
            record[key] = _metric(summary.get(key)) if status != "pending" else "Pending"
        for key in EXTERNAL_COLUMNS:
            record[key] = (_metric(external.get(key))
                           if status != "pending" and external.get("status") == "complete" else "Pending")
        record["external_gt_status"] = external.get("status", "pending")
        for key in ("n_folds", "n_converged", "full_fit_converged"):
            record[key] = summary.get(key, "Pending") if status != "pending" else "Pending"
        records.append(record)
    return records


def _method_label(record):
    targets = record["targets"].replace(" + ", "+")
    optimization = {"Sequential": "Seq"}.get(record["optimization"], record["optimization"])
    return f"{record['row']}<br>({targets}+{optimization}+{record['fk']})"


def _pooled(items):
    """Use sufficient statistics; never average per-fold RMSE values."""
    valid = [item for item in items if item.get("n_corners", 0) > 0
             and _finite(item.get("sum_squared_error_px2"))]
    n_corners = sum(item["n_corners"] for item in valid)
    squared = sum(item["sum_squared_error_px2"] for item in valid)
    return {"rmse_px": math.sqrt(squared / n_corners) if n_corners else None,
            "n_corners": n_corners, "sum_squared_error_px2": squared,
            "n_fold_evaluations": len(valid)}


def _pair_records(result):
    records = []
    for row, body, status, _ in _rows(result):
        for split, label in (("all", "ALL"), ("train", "Train"), ("heldout_test", "Held-out Test")):
            for pair in ("fixed_fixed", "fixed_gripper"):
                if split == "all":
                    stats = body.get("all", {}).get("cross_view", {}).get("by_pair_type", {}).get(pair, {})
                else:
                    stats = body.get("fold_summary", {}).get(f"{split}_cross_view", {}).get("by_pair_type", {}).get(pair)
                    if stats is None:
                        stats = _pooled([fold.get(split, {}).get("cross_view", {}).get("by_pair_type", {}).get(pair, {})
                                         for fold in body.get("folds", [])])
                records.append((row, label, pair.replace("_", "↔"),
                                _metric(stats.get("rmse_px")) if status != "pending" else "Pending",
                                stats.get("n_corners", "Pending") if status != "pending" else "Pending"))
    return records


def _fold_records(result):
    records = []
    for row, body, status, _ in _rows(result):
        if status == "pending":
            continue
        evaluations = [("ALL", "all", body.get("all", {}), body.get("summary", {}).get("full_fit_converged"))]
        for fold in body.get("folds", []):
            evaluations.extend((fold["set"], split, fold, fold.get("converged")) for split in ("train", "heldout_test"))
        for fold_id, split, evaluation, converged in evaluations:
            payload = evaluation if split == "all" else evaluation.get(split, {})
            for metric in ("cube_reprojection", "cross_view"):
                primary = payload.get(metric, {})
                stats_groups = [("overall", primary)]
                if metric == "cross_view":
                    stats_groups.extend((pair, primary.get("by_pair_type", {}).get(pair, {}))
                                        for pair in ("fixed_fixed", "fixed_gripper"))
                for pair, stats in stats_groups:
                    record = {"row": row, "fold": fold_id, "split": split, "metric": metric,
                              "pair_type": pair, "converged": converged if converged is not None else "Pending",
                              "training_placement_ids": json.dumps(evaluation.get("training_placement_ids", [])),
                              "heldout_placement_ids": json.dumps(evaluation.get("heldout_placement_ids", [])),
                              "rmse_px": _metric(stats.get("rmse_px")),
                              "sum_squared_error_px2": _metric(stats.get("sum_squared_error_px2"))}
                    record.update({key: stats.get(key, "Pending") for key in SUPPORT_COLUMNS})
                    records.append(record)
    return records


def _write_csv(path, fields, records):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def write_reports(result, output_dir):
    """Create Markdown, summary CSV and long-form per-fold CSV from v4 JSON."""
    if result.get("schema") != SCHEMA:
        raise ValueError(f"Expected {SCHEMA}; got {result.get('schema')!r}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"markdown": output_dir / "ABLATION_TEST_TABLE1_RESULTS.md",
             "csv": output_dir / "ABLATION_TEST_table1_results.csv",
             "fold_metrics": output_dir / "fold_metrics.csv"}
    records = _summary_records(result)
    minima = {key: min((record[key] for record in records if _finite(record[key])), default=None)
              for key in METRIC_COLUMNS}
    fold_records = _fold_records(result)
    _write_csv(paths["csv"], list(records[0]), records)
    fold_fields = ["row", "fold", "split", "metric", "pair_type", "converged",
                   "training_placement_ids", "heldout_placement_ids", "rmse_px",
                   "sum_squared_error_px2", *SUPPORT_COLUMNS]
    _write_csv(paths["fold_metrics"], fold_fields, fold_records)
    source = result.get("source_data", {})
    complete = [record for record in records if _finite(record["all_cube_rmse_px"])]
    source_rows = []
    directories = source.get("session_directories", {})
    subdirs = source.get("capture_subdirectories", {})
    for phase in ("p1", "p2", "p3"):
        if phase in directories:
            source_rows.append((phase.upper(), str(Path(directories[phase]) / subdirs.get(phase, ""))))
    if not source_rows and source.get("root"):
        source_rows.append(("입력", source["root"]))
    source_rows.extend((key, source[key]) for key in ("cube_config",) if key in source)
    counts = source.get("observations", {})
    placements = result.get("split", {}).get("placement_ids", [])
    n_folds = len(placements) if placements else max((record["n_folds"] for record in complete), default=0)
    all_converged = sum(record["full_fit_converged"] is True for record in complete)
    converged_folds = sum(record["n_converged"] for record in complete if isinstance(record["n_converged"], int))
    total_folds = sum(record["n_folds"] for record in complete if isinstance(record["n_folds"], int))
    lines = ["# Table 1 결과 요약", "",
             f"**계산된 방법 {len(complete)}개 · placement {n_folds}개 · ALL 수렴 {all_converged}/{len(complete)} · fold 수렴 {converged_folds}/{total_folds}.**", "",
             "내부 주 지표는 **Held-out Test Cross-view Cube RMSE px**. 최종 물리 순위는 독립 External GT로 결정한다.", "",
             "## 1. 전체 평가지표", "",
             "각 열은 작을수록 좋으며 **최솟값**을 굵게 표시한다. 반올림 전 값으로 비교하고 동률은 모두 표시한다. 미산출 값은 Pending이며, 소수점 4자리로 표시한다.", ""]
    _table(lines, ["실험 (구성)", *METRIC_COLUMNS.values()],
           [(_method_label(record), *(_highlight_minimum(record[key], minima[key])
                                     for key in METRIC_COLUMNS)) for record in records])
    lines.extend(["TRE·Rotation Error는 평균, P95 TRE는 95백분위수다. Failure Rate는 전체 GT pose 중 예측 실패·누락 비율(0–1)이며 solver 수렴률과 다르다.", "",
                  "## 2. 비교 구성과 미산출 항목", ""])
    _table(lines, ["Row", "학습 표적", "최적화", "FK 처리", "상태"],
           [(record["row"], record["targets"], record["optimization"], record["fk"], record["status"]) for record in records])
    pending = [(record["row"], record["pending_reason"]) for record in records if record["pending_reason"]]
    if pending:
        _table(lines, ["미산출 행", "사유"], pending)
    if any(record[key] == "Pending" for record in records for key in EXTERNAL_COLUMNS):
        lines.extend(["External GT가 미산출인 행은 독립 6-DoF GT와 방법별 frozen prediction이 필요하다. P1 비전으로 추정한 flange→Cube 변환은 corrected-FK이며 raw-FK의 독립 기계값이나 External GT가 아니다.", ""])
    lines.extend(["Sequential은 gripper 단계 후 결과를 동결하고 fixed 카메라를 적합한다. Unified는 함께 적합한다. 모든 행은 P1 Cube 기반 카메라 초기값과 P3 Board 기반 hand-eye 초기값을 공유한다. board-only/cube-only는 최적화 잔차 구성을 뜻하며 초기화까지 표적을 제외한 비교는 아니다.", "",
                  "## 3. 모든 데이터에 적용하는 평가 원리", "",
                  "1. **동일 LOPO:** placement 하나의 모든 P2 Cube·Board 관측을 학습에서 제외한다. 모든 방법과 지표에 같은 split을 쓴다.",
                  "2. **ALL / Train / Test:** ALL은 전체 placement로 별도 재fit한다. Train은 각 fold의 학습 placement, Test는 제외 placement를 평가한다.",
                  "3. **Cube:** 공통 FK-reference를 투영하여 코너 오차를 구한다. FK 계열에 유리할 수 있어 보조 지표로 사용한다.",
                  "4. **Cross-view:** A만으로 PnP → B로 전달한다. A→B·B→A 모두 계산하고, 각 방향의 destination 코너는 채점에만 쓴다.",
                  "5. **동일 RMSE:** `sqrt(mean(dx² + dy²))`. 코너 제곱오차 합 ÷ 코너 수의 제곱근이며 fold RMSE를 단순 평균하지 않는다.", "",
                  "fixed↔fixed 전달에는 FK가 필요 없고, fixed↔gripper 전달에는 촬영 순간 FK가 포함된다. Train은 fold별 평가 코너를 모아 집계한다.", "",
                  "## 4. 카메라 쌍별 Cross-view", ""])
    pairs = {(row, split, pair): value for row, split, pair, value, _count in _pair_records(result)}
    pair_columns = [(split, pair) for pair in ("fixed↔fixed", "fixed↔gripper")
                    for split in ("ALL", "Train", "Held-out Test")]
    _table(lines, ["Row", *(f"{pair} {split} px" for split, pair in pair_columns)],
           [(record["row"], *(pairs[(record["row"], split, pair)] for split, pair in pair_columns)) for record in records])
    lines.extend(["전체 카메라 쌍을 합친 값은 1절 Cross-view 열이다. 쌍·방향·코너 수와 fold별 수렴 상태는 [fold_metrics.csv](fold_metrics.csv)에 있다.", "",
                  "## 5. 입력·해석 범위·재생성", ""])
    _table(lines, ["입력", "저장소 기준 경로"], source_rows)
    if counts:
        lines.extend(["관측 수: " + "; ".join(f"{key}={value}" for key, value in counts.items()) + ".", ""])
    geometry = result.get("evaluation_limitations", {}).get("geometry")
    lines.extend(["**기하·스케일의 독립성:** " + (_display(geometry) if geometry else
                  "보정 자료의 출처가 기록되지 않았다. 기하 추정까지 독립적인 end-to-end 검증으로 해석하지 않는다."), "",
                  "원본 JSON에 입력 해시·K/D·변환행렬·split을, CSV에 반올림 전 값을 보존한다. 초기화 횟수·수렴·관측 수가 다른 실행끼리는 수치만으로 우열을 확정하지 않는다.", "",
                  "산출물: [원본 JSON](ABLATION_TEST_table1_methods.json) · [요약 CSV](ABLATION_TEST_table1_results.csv) · [fold CSV](fold_metrics.csv).", "",
                  "같은 JSON을 보고서로 다시 내보내기(재fit 없음):", "",
                  "```bash", "python zeus_gello_calibration/table1_zeus.py --report-only <결과폴더>/ABLATION_TEST_table1_methods.json", "```", ""])
    paths["markdown"].write_text("\n".join(lines), encoding="utf-8")
    return paths


def write_markdown_report(result, output_path):
    """Compatibility entry point using the same renderer for every dataset."""
    output_path = Path(output_path)
    paths = write_reports(result, output_path.parent)
    if output_path.resolve() != paths["markdown"].resolve():
        output_path.write_text(paths["markdown"].read_text(encoding="utf-8"), encoding="utf-8")


def regenerate_reports(json_path, output_dir):
    """Re-render reports from a saved v4 JSON; the raw JSON is copied byte-for-byte."""
    json_path, output_dir = Path(json_path), Path(output_dir)
    source_bytes = json_path.read_bytes()
    paths = write_reports(json.loads(source_bytes), output_dir)
    raw_json = output_dir / "ABLATION_TEST_table1_methods.json"
    if json_path.resolve() != raw_json.resolve():
        raw_json.write_bytes(source_bytes)
    return {"raw_json": raw_json, **paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    fcm_parser_defaults = dict(
        session1_dir=str(fcm.SESSION1_DIR_DEFAULT),
        session1_capture_subdir="capture_replayed",
        session2_dir=str(fcm.SESSION2_DIR_DEFAULT),
        session2_capture_subdir="capture_placed",
        session3_dir=str(fcm.SESSION3_DIR_DEFAULT),
        session3_capture_subdir="capture_replayed",
        zeus_intrinsics_dir=str(REPO_ROOT / "intrinsics"),
        ur3_intrinsics_dir=str(REPO_ROOT / "ur3_calibration" / "intrinsics"),
        device_map=str(REPO_ROOT / "intrinsics" / "device_map.json"),
        fit_json=str(fcm.FIT_JSON_DEFAULT),   # capture_replayed 와 짝인 replayed 아티팩트
        fixed_min_corners=8,
        cube_observation_policy="legacy",
    )
    for key, value in fcm_parser_defaults.items():
        parser.add_argument(f"--{key.replace('_', '-')}", default=value,
                            type=type(value) if not isinstance(value, bool) else str)
    parser.add_argument("--cube-config", default=None,
                        help="Cube geometry JSON for the captured target")
    parser.add_argument("--s3-gripper-only", action="store_true",
                        help="Use only gripper-camera board observations from session3")
    parser.add_argument("--no-s3-cube", action="store_true",
                        help="Do not use the (static) cube seen in session3; default uses it as extra observations of its placement set")
    parser.add_argument("--include-session2-board", action="store_true",
                        help="Include session2 board observations and exclude the held-out placement's board event")
    parser.add_argument("--rows", default=",".join(r for r in ROW_ORDER if r in ROWS))
    parser.add_argument("--folds", type=int, default=0, help="0 = 모든 placement")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel LOPO fit processes; full-data fit remains separate")
    parser.add_argument("--mechanical-transform-json",
                        help="Vision-free T_flange_cube for custom cubes; translation in metres")
    parser.add_argument("--report-dir",
                        help="Write detailed Markdown, summary CSV and per-fold CSV here")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "ABLATION_TEST_table1_zeus.json"),
    )
    parser.add_argument(
        "--md-out",
        default=None, help="Optional additional Markdown copy; standard report names are always generated",
    )
    parser.add_argument(
        "--report-only", type=Path, metavar="JSON",
        help="저장된 v4 JSON으로 보고서만 재생성 (재fit 없음); 출력은 --report-dir 또는 JSON 폴더",
    )
    args = parser.parse_args()
    if args.report_only:
        report_dir = Path(args.report_dir) if args.report_dir else args.report_only.parent
        for path in regenerate_reports(args.report_only, report_dir).values():
            print(f"[SAVE] {_display(str(path))}")
        return
    if args.workers < 1 or args.folds < 0:
        parser.error("--workers must be positive; --folds must be nonnegative")

    for attribute in ("session1_dir", "session2_dir", "session3_dir"):
        value = require_zeus_data_path(
            getattr(args, attribute), label=f"--{attribute.replace('_', '-')}"
        )
        setattr(args, attribute, str(value))

    data = fcm.load_all_data(args)
    data["include_session2_board"] = bool(args.include_session2_board)
    mechanical_fk = mechanical_fk_contract()
    mechanical_pending = None
    if args.mechanical_transform_json:
        data["mechanical_grasp"], mechanical_fk = load_mechanical_transform(
            args.mechanical_transform_json)
    elif args.cube_config:
        mechanical_pending = (
            "선택한 Cube의 비전 미사용 T_flange_cube 입력이 없음. "
            "robot.json pose는 T_base_flange이며 flange-to-Cube 장착 변환이 아님. "
            "다른 Cube의 nominal 기계 변환은 재사용하지 않음."
        )
        mechanical_fk = {"status": "pending", "reason": mechanical_pending,
                         "vision_used": False, "T_flange_cube": None}
    corrected_fk = corrected_fk_training_contract(args.fit_json)
    if args.cube_config:
        corrected_fk["nominal_axis_map"] = None
        corrected_fk["rotation_deviation_from_nominal_deg"] = None
        corrected_fk["rotation_matrix"] = np.asarray(data["grasp_init"])[:3, :3].tolist()
        corrected_fk["nominal_rotation_status"] = (
            "선택한 Cube에 다른 Cube의 nominal 회전을 적용하지 않음")
    robot_T = {**data["robot_T_s1"], **data["robot_T_s2_gripper"], **data["robot_T_s3"]}
    # fit_calibration_methods.main() 과 동일한 초기화: session3 보드만으로 hand-eye 초기값
    gtc_init, board_init, eih_diag = estimate_board_handeye_initial(
        data["obs_s3"], data["robot_T_s3"], data["K_map"], data["D_map"], GRIPPER)
    print(f"T_gripper_cam 초기값 t_mm={np.round(gtc_init[:3,3]*1000,2).tolist()} ({eih_diag})")

    set_ids = sorted(int(s) for s in data["items_by_index"])
    if args.folds:
        set_ids = set_ids[:args.folds]
    rows = [r.strip() for r in args.rows.split(",") if r.strip()]
    if not rows or set(rows) - set(ROWS):
        parser.error("--rows must contain known Table 1 row IDs")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print(f"\nplacement {len(set_ids)}개 leave-one-out x row {len(rows)}개\n")
    result = {
        "schema": SCHEMA,
        "source_data": {
            "root": str(ZEUS_DATA_ROOT.resolve()),
            "session_directories": {
                "p1": args.session1_dir,
                "p2": args.session2_dir,
                "p3": args.session3_dir,
            },
            "capture_subdirectories": {
                "p1": args.session1_capture_subdir,
                "p2": args.session2_capture_subdir,
                "p3": args.session3_capture_subdir,
            },
            "cube_config": str(Path(args.cube_config).resolve()) if args.cube_config else "default CubeConfig",
            "s3_gripper_only": bool(args.s3_gripper_only),
            "include_session2_board": bool(args.include_session2_board),
            "cube_observation_policy": args.cube_observation_policy,
            "fixed_min_corners": int(args.fixed_min_corners),
            "observations": {
                "p1_cube": len(data["obs_s1"]),
                "p2_fixed_cube": len(data["obs_s2_fixed"]),
                "p2_gripper_cube": len(data["obs_s2_gripper"]),
                "p3_board": len(data["obs_s3"]),
                "p3_cube": len(data.get("obs_s3_cube", [])),
                "p3_cube_set": data.get("s3_cube_set"),
                "p2_board": len(data.get("obs_s2_board", [])) if args.include_session2_board else 0,
                "total": (
                    len(data["obs_s1"]) + len(data["obs_s2_fixed"])
                    + len(data["obs_s2_gripper"]) + len(data["obs_s3"]) + len(data.get("obs_s3_cube", []))
                    + (len(data.get("obs_s2_board", [])) if args.include_session2_board else 0)
                ),
            },
        },
        "loader_provenance": data.get("source_data_provenance", {}),
        "initialization_contract": {
            "fixed_cameras_and_corrected_grasp": "P1-only fit, shared across rows/folds",
            "handeye": "P3 board observations only, shared across rows/folds",
            "cube_pose_initialization": "current fold's train Cube observations only",
            "B2_scope": "board residual ablation; P3 board-derived initialization retained",
            "n_initializations_per_fit": 1,
        },
        "split": {
            "strategy": "leave_one_placement_out",
            "placement_ids": sorted(int(s) for s in data["items_by_index"]),
            "evaluated_fold_ids": set_ids,
            "same_split_for_all_rows_and_pixel_metrics": True,
            "heldout_session2_board_observations_excluded": True,
            "all_is_separate_full_data_refit": True,
        },
        "corrected_fk_definition": (
            "T_base_cube[s] = T_base_flange(place_s) @ "
            "T_flange_cube_from_P1_train_VISION"
        ),
        "mechanical_fk": mechanical_fk,
        "corrected_fk_training": corrected_fk,
        "T_flange_cube_corrected_source": args.fit_json,
        "T_flange_cube_translation_mm": (np.asarray(data["grasp_init"])[:3, 3] * 1000.0).tolist(),
        "pending_rows": {},
        "metric_contracts": {
            "all": (
                "all placements를 한 번에 fit한 calibration으로 전체 session2 cube를 평가; "
                "train+test 평균이 아니라 full-data descriptive fit"
            ),
            "train": "각 leave-one-placement-out fold의 calibration-train placements 평가; 모든 fold의 평가 corner를 합쳐 RMSE 계산",
            "heldout_test": (
                "각 fold에서 calibration에 넣지 않은 한 placement만 평가; "
                "해당 P2 Cube와 Board 관측을 모두 fit에서 제외; test-time calibration refit 없음"
            ),
            "cube_reprojection": (
                "모든 row가 같은 P1 train-VISION corrected-FK reference T_base_cube를 "
                "사용한 sqrt(mean(dx^2 + dy^2)), pooled-corner pixel RMSE; corrected-FK 방법에 구조적으로 "
                "유리하므로 내부 보조 지표"
            ),
            "cross_view": (
                "source camera PnP만 destination으로 양방향 전달; destination corner는 scoring에만 사용; "
                "sqrt(mean(dx^2 + dy^2)), pooled-corner pixel RMSE; "
                "fixed-fixed uses no Robot FK, fixed-gripper includes Robot FK"
            ),
            "external_gt": (
                "별도 blind pose에서 frozen prediction과 독립 T_base_cube_GT를 비교한 "
                "TRE_mm, rotation_error_deg, P95_TRE_mm, failure_rate; 최종 순위 지표"
            ),
        },
        "fk_factor_prior": {"translation_std_mm": SIGMA_FK_MM, "rotation_std_deg": SIGMA_FK_DEG,
                            "measured": False},
        "rows": {},
    }
    cube_reference = fk_cube_poses(data["items_by_index"], data["grasp_init"])
    all_observations = session2_observations(data)
    for row in rows:
        t0 = time.time()
        if row == "A3" and mechanical_pending:
            result["rows"][row] = {
                "condition": {**ROWS[row], "label": "FK hard fixed (custom Cube mechanical transform pending)"},
                "status": "pending",
                "pending_reason": mechanical_pending, "folds": [],
                "summary": empty_metric_summary(), "external_gt": external_gt_pending(),
            }
            continue
        if not ROWS[row].get("available", True):
            result["rows"][row] = {
                "condition": ROWS[row],
                "status": "pending",
                "pending_reason": ROWS[row]["pending_reason"],
                "folds": [],
                "summary": empty_metric_summary(),
                "external_gt": external_gt_pending(),
            }
            print(f"  {row:<3} {ROWS[row]['label']:<40} [Pending]")
            continue
        full_state, full_solver_px, full_n_obs, full_ok = fit_row(
            row, data, None, robot_T, board_init, gtc_init)
        if full_state is None:
            print(f"  {row:<3} full-data fit failed")
            result["rows"][row] = {
                "condition": ROWS[row], "status": "failed",
                "pending_reason": "full-data calibration fit failed",
                "folds": [], "summary": empty_metric_summary(),
                "external_gt": external_gt_pending(),
            }
            continue
        all_cube = cube_reprojection_stats(
            all_observations, cube_reference, full_state, robot_T,
            data["K_map"], data["D_map"])
        all_cross = cross_view_transfer_stats(
            all_observations, full_state, robot_T,
            data["K_map"], data["D_map"])
        jobs = [(row, data, s, robot_T, board_init, gtc_init) for s in set_ids]
        folds = []
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                for fold in executor.map(_fold_job, jobs):
                    folds.append(fold)
                    print(f"      {row} fold {len(folds)}/{len(set_ids)}", flush=True)
        else:
            for job in jobs:
                folds.append(_fold_job(job))
                print(f"      {row} fold {len(folds)}/{len(set_ids)}", flush=True)
        fold_summary = aggregate_folds(folds)
        summary = {
            "all_cube_rmse_px": all_cube["rmse_px"],
            "train_cube_rmse_px": fold_summary["train_cube_reprojection"]["rmse_px"],
            "heldout_test_cube_rmse_px": fold_summary["heldout_test_cube_reprojection"]["rmse_px"],
            "all_cross_view_cube_rmse_px": all_cross["rmse_px"],
            "train_cross_view_cube_rmse_px": fold_summary["train_cross_view"]["rmse_px"],
            "heldout_test_cross_view_cube_rmse_px": fold_summary["heldout_test_cross_view"]["rmse_px"],
            "n_folds": fold_summary["n_folds"],
            "n_converged": fold_summary["n_converged"],
            "full_fit_converged": full_ok,
        }
        result["rows"][row] = {
            "condition": ROWS[row],
            "status": "complete" if (
                full_ok and fold_summary["n_converged"] == len(data["items_by_index"])
                and fold_summary["n_folds"] == len(data["items_by_index"])
            ) else "incomplete",
            "all": {
                "transforms": serialize_state(full_state),
                "training_placement_ids": sorted(int(s) for s in data["items_by_index"]),
                "solver_train_rmse_px_all_targets": full_solver_px,
                "n_solver_observations": full_n_obs,
                "cube_reprojection": all_cube,
                "cross_view": all_cross,
            },
            "folds": [fold for fold in folds if fold],
            "fold_summary": fold_summary,
            "summary": summary,
            "external_gt": external_gt_pending(),
        }
        save_result(result, args.out)
        print(f"  {row:<3} {ROWS[row]['label']:<40} [{time.time()-t0:.0f}s]")
        print(
            "      Cube(corrected-FK-ref) "
            f"ALL={summary['all_cube_rmse_px']:.3f}  "
            f"Train={summary['train_cube_rmse_px']:.3f}  "
            f"Heldout={summary['heldout_test_cube_rmse_px']:.3f} px"
        )
        print(
            "      Cross-view  "
            f"ALL={summary['all_cross_view_cube_rmse_px']:.3f}  "
            f"Train={summary['train_cross_view_cube_rmse_px']:.3f}  "
            f"Heldout={summary['heldout_test_cross_view_cube_rmse_px']:.3f} px"
        )

    save_result(result, args.out)
    report_dir = Path(args.report_dir) if args.report_dir else Path(args.out).parent
    save_result(result, report_dir / "ABLATION_TEST_table1_methods.json")
    reports = write_reports(result, report_dir)
    if args.md_out:
        destination = Path(args.md_out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.resolve() != reports["markdown"].resolve():
            destination.write_text(reports["markdown"].read_text(encoding="utf-8"), encoding="utf-8")
    print(f"\n[SAVE] {_display(str(report_dir / 'ABLATION_TEST_table1_methods.json'))}")
    print(f"[SAVE] {_display(str(reports['markdown']))}")



if __name__ == "__main__":
    main()
