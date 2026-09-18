#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zeus_gello_calibration/fit_calibration_methods_mm.py -- 학습 목적함수를
px 재투영이 아니라 mm/deg pose 오차로 바꿔서 fit_calibration_methods.py의
3가지 방식(통합_no-fk, 통합_raw-fk, 독립_no-fk)을 다시 풀고, 같은 데이터를
px로 학습했을 때와 mm으로 학습했을 때 표가 어떻게 달라지는지 비교한다.

*** 왜 이걸 만들었는가 ***: fit_calibration_methods.py/eval_heldout_and_
consistency.py는 전부 solve_corner_reprojection(픽셀 재투영 최소제곱)으로
학습한다. "raw-fk가 학습 목적함수(px)와 검증 기준(px)이 같아서 유리하게
나온다"는 순환성 문제를 논의하다가, "그럼 학습 자체를 mm으로 하면 표가 어떻게
달라지는가"를 직접 확인하기 위해 만들었다.

*** mm 학습 방식 ***: 각 관측치(한 카메라가 한 순간 찍은 한 장의 사진)마다
먼저 solve_observed_pose()로 "그 사진 한 장만 갖고" 계산한 T_cam_target(단일
이미지 PnP pose, mm/deg)을 구해서 고정 관측값으로 삼는다. 그 다음
자유 변수(T_base_Ci, T_gripper_cam, T_base_board, T_base_cube_by_set,
T_gripper_cube_by_grasp)는 px 재투영이 아니라 "예측한 T_cam_target vs 그
단일-이미지 PnP pose" 사이의 SE(3) 오차(rotvec rad + translation m)를
직접 최소화하도록 최적화한다. 파라미터화(PoseState/variable_keys/retract)는
px판과 완전히 동일하고 잔차 정의만 다르다.

한계: 단일 이미지 PnP pose 자체도 그 한 장의 코너 검출 노이즈에서 나온
값이라 px에서 완전히 자유롭지는 않다. 다만 재투영 최적화 없이 "한 번 계산해서
고정"해두고 쓰기 때문에, 카메라들이 서로의 노이즈에 맞춰 왜곡되는 정도(raw-fk
논의에서 나온 "오차가 카메라 extrinsics로 흡수되는" 현상)는 px 학습보다 줄어들
것으로 기대된다 -- 그 기대가 맞는지 확인하는 게 이 스크립트의 목적이다.

사용법:
  python fit_calibration_methods_mm.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from calibration_pipeline import se3 as cp  # noqa: E402
from calibration_pipeline.apriltag_cube import inv_T  # noqa: E402
from calibration_pipeline.path_evaluation import solve_observed_pose  # noqa: E402
from calibration_pipeline.reprojection import (  # noqa: E402
    PoseState, SE3Scaling, pose_delta, retract, set_state_transform, state_transform, variable_keys,
)
from calibration_pipeline.table1 import estimate_board_handeye_initial  # noqa: E402

from fit_calibration_methods import (  # noqa: E402
    GRIPPER_LOCAL_ID, SESSION1_DIR_DEFAULT, SESSION3_DIR_DEFAULT,
    export_fit_json, fk_anchor_cubes, init_board_pose, init_cube_poses, load_all_data,
    per_corner_errors, rmse_px,
)
from eval_heldout_and_consistency import camera_cube_estimate, evaluate_heldout, fit_frozen  # noqa: E402
from session2_pick_and_place import SESSION2_DIR_DEFAULT  # noqa: E402


# --------------------------------------------------------------- mm 잔차 코어
@dataclass(frozen=True)
class PoseObs:
    marker: str
    cam: int
    event: int
    set_idx: Optional[int]
    grasp_idx: Optional[int]
    T_cam_target: np.ndarray  # 그 한 장의 사진만으로 계산한 단일-이미지 PnP pose


def build_pose_obs(pixel_obs_list, K_map, D_map):
    """PixelObs 리스트 -> (한 장씩 PnP로 미리 풀어놓은) PoseObs 리스트.
    PnP가 실패하는 관측치(코너 too few/degenerate)는 조용히 버린다."""
    out = []
    for o in pixel_obs_list:
        T = solve_observed_pose(o, K_map, D_map)
        if T is None:
            continue
        out.append(PoseObs(o.marker, int(o.cam), int(o.event), o.set_idx, o.grasp_idx, T))
    return out


def _predict_T_cam_target(state, o, robot_T, gripper_id):
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
    return inv_T(T_base_cam) @ target


class PoseDeltaProblem:
    """CornerReprojectionProblem과 같은 파라미터화(PoseState/retract), 잔차만
    px 재투영이 아니라 SE(3) pose 오차(rotvec rad 3 + translation m 3)."""

    def __init__(self, pose_obs, variable_keys_, reference_state, robot_T, gripper_cam_idx,
                 scaling: SE3Scaling = SE3Scaling()):
        self.obs = list(pose_obs)
        self.variable_keys = list(variable_keys_)
        if len(self.variable_keys) != len(set(self.variable_keys)):
            raise ValueError("duplicate optimization variable")
        self.reference_state = reference_state.clone()
        self.robot_T = robot_T
        self.gripper = int(gripper_cam_idx)
        self.scaling = scaling
        self.slices = {key: slice(6 * i, 6 * (i + 1)) for i, key in enumerate(self.variable_keys)}
        self.n_params = 6 * len(self.variable_keys)
        self.x0 = np.zeros(self.n_params, dtype=np.float64)
        if not self.obs:
            raise RuntimeError("pose-delta problem has no observations")
        if not self.n_params:
            raise RuntimeError("pose-delta problem has no free variables")

    def unpack(self, x):
        state = self.reference_state.clone()
        for key, sl in self.slices.items():
            set_state_transform(
                state, key, retract(state_transform(self.reference_state, key), x[sl], self.scaling))
        return state

    def residual_vector(self, x):
        state = self.unpack(x)
        chunks = []
        for o in self.obs:
            T_pred = _predict_T_cam_target(state, o, self.robot_T, self.gripper)
            err = inv_T(o.T_cam_target) @ T_pred
            rotvec = Rotation.from_matrix(err[:3, :3]).as_rotvec()
            trans = err[:3, 3]
            chunks.append(np.concatenate([rotvec, trans]))
        return np.concatenate(chunks).astype(np.float64)


def solve_pose_delta(pose_obs, variable_keys_, reference_state, robot_T, gripper_cam_idx):
    """px판 solve_corner_reprojection에 대응하는 mm판 solver."""
    problem = PoseDeltaProblem(pose_obs, variable_keys_, reference_state, robot_T, gripper_cam_idx)
    solution = least_squares(
        problem.residual_vector, problem.x0, method="trf", loss="soft_l1",
        f_scale=0.01,  # rotvec(rad)/translation(m) 공용 robust threshold, 대략 0.6deg/10mm
        x_scale="jac", max_nfev=300, xtol=1e-10, ftol=1e-10, gtol=1e-10,
    )
    state = problem.unpack(solution.x)
    final = problem.residual_vector(solution.x).reshape(-1, 6)
    rot_deg = np.degrees(np.linalg.norm(final[:, :3], axis=1))
    trans_mm = np.linalg.norm(final[:, 3:], axis=1) * 1000.0
    diag = {
        "success": bool(solution.success),
        "n_obs": len(problem.obs),
        "n_params": problem.n_params,
        "train_rmse_mm": float(np.sqrt(np.mean(trans_mm ** 2))),
        "train_rmse_deg": float(np.sqrt(np.mean(rot_deg ** 2))),
    }
    return state, diag


def score_mm(state, observations, K_map, D_map, robot_T, gripper_id):
    """어느 state든(px로 학습했든 mm으로 학습했든) 주어진 관측치 집합에 대해
    mm/deg pose 오차로 채점 -- px/mm 학습을 공정하게 비교하기 위한 공용 지표."""
    pose_obs = build_pose_obs(observations, K_map, D_map)
    if not pose_obs:
        return float("nan"), float("nan")
    trans_mm, rot_deg = [], []
    for o in pose_obs:
        T_pred = _predict_T_cam_target(state, o, robot_T, gripper_id)
        d_mm, d_deg = pose_delta(o.T_cam_target, T_pred)
        trans_mm.append(d_mm)
        rot_deg.append(d_deg)
    return float(np.sqrt(np.mean(np.square(trans_mm)))), float(np.sqrt(np.mean(np.square(rot_deg))))


def score_px(state, observations, K_map, D_map, robot_T, gripper_id):
    errs = per_corner_errors(state, observations, robot_T, K_map, D_map, gripper_id)
    return rmse_px(errs)


# ------------------------------------------------------------------ 통합(mm)
def select_unified_observations(data, fk_mode, gtc_init):
    """solve_unified(px판)와 완전히 같은 큐브 선택/필터 로직 -- px/mm이 정확히
    같은 학습 데이터를 쓰도록 강제(공정 비교의 전제조건)."""
    cam_init, grasp_init = data["cam_init"], data["grasp_init"]
    K_map, D_map = data["K_map"], data["D_map"]
    obs_s2 = data["obs_s2_fixed"] + data["obs_s2_gripper"]
    set_ids = sorted(data["items_by_index"])
    robot_T = {**data["robot_T_s1"], **data["robot_T_s2_gripper"], **data["robot_T_s3"]}
    observations = data["obs_s1"] + obs_s2 + data.get("obs_s2_board", []) + data["obs_s3"]

    if fk_mode == "no_fk":
        cubes = init_cube_poses(obs_s2, K_map, D_map, cam_init, gtc_init, robot_T, GRIPPER_LOCAL_ID, set_ids)
        cube_key = ["T_base_cube_by_set"]
    else:
        cubes = fk_anchor_cubes(data["items_by_index"], grasp_init)
        cube_key = []
    observations = [o for o in observations
                    if o.set_idx is None or int(o.set_idx) in cubes or o.grasp_idx is not None]
    return cubes, cube_key, observations, robot_T


def solve_unified_mm(data, fk_mode, gtc_init, board_init):
    cam_init, grasp_init = data["cam_init"], data["grasp_init"]
    cubes, cube_key, observations, robot_T = select_unified_observations(data, fk_mode, gtc_init)
    K_map, D_map = data["K_map"], data["D_map"]
    pose_obs = build_pose_obs(observations, K_map, D_map)

    state0 = PoseState(cams=dict(cam_init), gtc=gtc_init.copy(), board=board_init.copy(),
                       cubes=dict(cubes), grasps={0: grasp_init.copy()})
    keys = variable_keys(["T_base_Ci", "T_gripper_cam", "T_base_board"] + cube_key + ["T_gripper_cube_by_grasp"], state0)
    final_state, diag = solve_pose_delta(pose_obs, keys, state0, robot_T, GRIPPER_LOCAL_ID)
    return final_state, diag, observations, robot_T


# --------------------------------------------------------------- 독립(mm)
def solve_parallel_fixed_mm(data):
    cam_init, grasp_init = data["cam_init"], data["grasp_init"]
    K_map, D_map = data["K_map"], data["D_map"]
    obs_s2 = data["obs_s2_fixed"]
    obs_s3f = data["obs_s3_fixed"]
    set_ids = sorted(data["items_by_index"])
    cubes = init_cube_poses(obs_s2, K_map, D_map, cam_init, np.eye(4), {}, -999, set_ids)
    board_init_fixed = init_board_pose(obs_s3f, K_map, D_map, cam_init)
    observations = (data["obs_s1"]
                    + [o for o in obs_s2 if o.set_idx is not None and int(o.set_idx) in cubes]
                    + obs_s3f)
    robot_T = {**data["robot_T_s1"], **data["robot_T_s3"]}
    pose_obs = build_pose_obs(observations, K_map, D_map)
    state0 = PoseState(cams=dict(cam_init), gtc=np.eye(4), board=board_init_fixed,
                       cubes=dict(cubes), grasps={0: grasp_init.copy()})
    keys = variable_keys(["T_base_Ci", "T_base_cube_by_set", "T_gripper_cube_by_grasp", "T_base_board"], state0)
    final_state, diag = solve_pose_delta(pose_obs, keys, state0, robot_T, -999)
    return final_state, diag, observations, robot_T


def solve_parallel_gripper_mm(data, gtc_init, board_init):
    K_map, D_map = data["K_map"], data["D_map"]
    obs_s2g = data["obs_s2_gripper"]
    obs_s3g = data["obs_s3_gripper"]
    set_ids = sorted(data["items_by_index"])
    robot_T = {**data["robot_T_s2_gripper"], **data["robot_T_s3"]}
    cubes = init_cube_poses(obs_s2g, K_map, D_map, {}, gtc_init, robot_T, GRIPPER_LOCAL_ID, set_ids)
    observations = ([o for o in obs_s2g if o.set_idx is not None and int(o.set_idx) in cubes]
                    + obs_s3g)
    pose_obs = build_pose_obs(observations, K_map, D_map)
    state0 = PoseState(cams={}, gtc=gtc_init.copy(), board=board_init.copy(), cubes=dict(cubes), grasps={})
    keys = variable_keys(["T_gripper_cam", "T_base_board", "T_base_cube_by_set"], state0)
    final_state, diag = solve_pose_delta(pose_obs, keys, state0, robot_T, GRIPPER_LOCAL_ID)
    return final_state, diag, observations, robot_T


def fit_frozen_mm(method, data_fold, fk_mode, gtc_init, board_init):
    """eval_heldout_and_consistency.fit_frozen과 같은 인터페이스, mm로 학습."""
    if method == "통합":
        state, _, _, _ = solve_unified_mm(data_fold, fk_mode, gtc_init, board_init)
        return state.cams, state.gtc
    state_fixed, _, _, _ = solve_parallel_fixed_mm(data_fold)
    state_gripper, _, _, _ = solve_parallel_gripper_mm(data_fold, gtc_init, board_init)
    return state_fixed.cams, state_gripper.gtc


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
    ap.add_argument("--fit-json", default=str(REPO_ROOT / "zeus_gello_calibration" / "pass1_grasp_offset_replayed.json"))
    ap.add_argument("--fixed-min-corners", type=int, default=8)
    ap.add_argument("--cube-observation-policy", default="legacy", choices=("legacy", "core_multiface"))
    ap.add_argument("--skip-heldout", action="store_true", help="held-out(느림)은 건너뛰고 train표만")
    ap.add_argument("--out", default=str(REPO_ROOT / "zeus_gello_calibration" / "results" / "heldout" / "px_vs_mm_training.json"))
    args = ap.parse_args()

    data = load_all_data(args)
    K_map, D_map = data["K_map"], data["D_map"]
    set_ids = sorted(data["items_by_index"])

    gtc_init, board_init, eih_diag = estimate_board_handeye_initial(
        data["obs_s3"], data["robot_T_s3"], K_map, D_map, GRIPPER_LOCAL_ID)
    print(f"T_gripper_cam 초기값 (session3만): t_mm={np.round(gtc_init[:3,3]*1000,2).tolist()} ({eih_diag})\n")

    from fit_calibration_methods import solve_parallel_fixed as solve_parallel_fixed_px
    from fit_calibration_methods import solve_parallel_gripper as solve_parallel_gripper_px
    from fit_calibration_methods import solve_unified as solve_unified_px

    fit_out_dir = REPO_ROOT / "zeus_gello_calibration"
    conditions = [("통합", "no_fk", "통합_no-fk"), ("통합", "fixed_fk", "통합_raw-fk"), ("독립_true", "no_fk", "독립_no-fk")]
    rows = {}
    print("mm 학습판 T_gripper_cube/T_base_cam을 gt_pick_test.py/gt_compare_fits.py용 JSON으로 저장 "
          "(px 학습판은 fit_calibration_methods.py가 이미 fit_<조건>.json으로 저장해둠):")

    for method, fk_mode, label in conditions:
        print(f"=== {label} ===")
        if method == "통합":
            state_px, _diag_px, _n_obs_px = solve_unified_px(data, fk_mode, gtc_init, board_init)
            # solve_unified(px)는 (state, diag, n_obs)만 반환 -- 학습에 쓴 실제
            # observation 리스트는 mm판과 동일한 select_unified_observations로 재구성
            # (같은 fk_mode/data이면 결정론적으로 같은 선택이 나오므로 px/mm이
            # 정확히 같은 학습 데이터를 쓰는 게 보장된다).
            _, _, obs_px, robot_T_px = select_unified_observations(data, fk_mode, gtc_init)
            state_mm, _diag_mm, obs_mm, robot_T_mm = solve_unified_mm(data, fk_mode, gtc_init, board_init)
            export_fit_json(fit_out_dir / f"fit_{label}_mm.json", state_mm.grasps[0], state_mm.cams, state_mm.gtc)
            gripper_id = GRIPPER_LOCAL_ID
        else:
            state_fixed_px, _, obs_fixed_px = solve_parallel_fixed_px(data)
            state_gripper_px, _, obs_gripper_px = solve_parallel_gripper_px(data, gtc_init, board_init)
            state_fixed_mm, _, obs_fixed_mm, robot_T_fixed = solve_parallel_fixed_mm(data)
            state_gripper_mm, _, obs_gripper_mm, robot_T_gripper = solve_parallel_gripper_mm(data, gtc_init, board_init)
            export_fit_json(fit_out_dir / f"fit_{label}_mm.json",
                            state_fixed_mm.grasps[0], state_fixed_mm.cams, state_gripper_mm.gtc)

            row = {}
            for tag, state, obs, robot_T, gid in (
                ("fixed_px", state_fixed_px, obs_fixed_px, data["robot_T_s1"], -999),
                ("gripper_px", state_gripper_px, obs_gripper_px,
                 {**data["robot_T_s2_gripper"], **data["robot_T_s3"]}, GRIPPER_LOCAL_ID),
                ("fixed_mm", state_fixed_mm, obs_fixed_mm, robot_T_fixed, -999),
                ("gripper_mm", state_gripper_mm, obs_gripper_mm, robot_T_gripper, GRIPPER_LOCAL_ID),
            ):
                px = score_px(state, obs, K_map, D_map, robot_T, gid)
                mm, deg = score_mm(state, obs, K_map, D_map, robot_T, gid)
                row[tag] = {"train_rmse_px": px, "train_rmse_mm": mm, "train_rmse_deg": deg}
                print(f"  {tag:>11}: train px={px:.4f}  mm={mm:.3f}  deg={deg:.3f}")
            rows[label] = row
            continue

        row = {}
        for tag, state, obs, robot_T in (("px", state_px, obs_px, robot_T_px), ("mm", state_mm, obs_mm, robot_T_mm)):
            px = score_px(state, obs, K_map, D_map, robot_T, gripper_id)
            mm, deg = score_mm(state, obs, K_map, D_map, robot_T, gripper_id)
            row[tag] = {"train_rmse_px": px, "train_rmse_mm": mm, "train_rmse_deg": deg}
            print(f"  {tag:>11}: train px={px:.4f}  mm={mm:.3f}  deg={deg:.3f}")
        rows[label] = row

    if not args.skip_heldout:
        print("\nheld-out (leave-one-out, FK 기준 GT) 계산 중 -- px학습 vs mm학습 각각...")
        robot_T_all = {**data["robot_T_s1"], **data["robot_T_s2_gripper"], **data["robot_T_s3"]}
        for method, fk_mode, label in conditions:
            for tag, fn in (("px", fit_frozen), ("mm", fit_frozen_mm)):
                heldout_px, _per_set, per_set_mm_deg, _heldout_cross = _evaluate_heldout_with(
                    fn, method, data, fk_mode, gtc_init, board_init, robot_T_all, K_map, D_map, set_ids)
                heldout_mm = [v["translation_mm"] for v in per_set_mm_deg.values()]
                heldout_deg = [v["rotation_deg"] for v in per_set_mm_deg.values()]
                rows[label].setdefault(tag, {})["heldout_mm"] = float(np.mean(heldout_mm)) if heldout_mm else float("nan")
                rows[label][tag]["heldout_deg"] = float(np.mean(heldout_deg)) if heldout_deg else float("nan")
                rows[label][tag]["heldout_px"] = heldout_px
                print(f"  [{label}/{tag}] heldout px={heldout_px:.3f} mm={rows[label][tag]['heldout_mm']:.3f} "
                      f"deg={rows[label][tag]['heldout_deg']:.3f}")

    Path(args.out).write_text(json.dumps(rows, indent=2, default=lambda o: str(o)))
    print(f"\nwrote {args.out}")


def _evaluate_heldout_with(fit_frozen_fn, method, data, fk_mode, gtc_init, board_init, robot_T_all, K_map, D_map, set_ids):
    """evaluate_heldout과 완전히 같은 GT/leave-one-out 로직이되 fit_frozen을
    fit_frozen_fn으로 교체 (mm 학습판 held-out을 위해 임시로 monkeypatch)."""
    import eval_heldout_and_consistency as _m
    original = _m.fit_frozen
    _m.fit_frozen = fit_frozen_fn
    try:
        return evaluate_heldout(method, data, fk_mode, gtc_init, board_init, robot_T_all, K_map, D_map, set_ids)
    finally:
        _m.fit_frozen = original


if __name__ == "__main__":
    main()
