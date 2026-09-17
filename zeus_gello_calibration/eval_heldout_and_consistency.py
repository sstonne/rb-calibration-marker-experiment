#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zeus_gello_calibration/eval_heldout_and_consistency.py -- 통합/독립 x
raw-fk/no-fk 3가지 방식(통합_no-fk, 통합_raw-fk, 독립_no-fk)에 대해
(1) session2 세트 leave-one-out held-out
reprojection RMSE, (2) 카메라 간 큐브 pose 일치도(cross-camera consistency,
mm/deg)를 계산한다. late_table1(ABLATION_TEST_result_<MMDD>/session02_NOUSE_session04_0814)의 "Heldout Cube RMSE"/
"Cross-view Cube px"/"Cam-common Cube mm/deg" 지표를 Zeus 데이터로 재현한 것.

Held-out: session2의 15개 세트를 하나씩 빼고(session1+session3는 항상 포함)
나머지로 다시 fit한 뒤, 뺐던 세트의 코너들을 재투영해서 오차를 잰다 -- 학습에
전혀 안 쓰인 세트에 대한 일반화 성능.

*** 정답(ground truth) 값 ***: 예전 버전은 "빠진 세트를 그 세트 자신의
사진들로 삼각측량"해서 정답을 만들었는데, 이러면 정답을 만드는 재료와 검증할
때 쓰는 재료가 똑같은 카메라의 같은 사진이라 카메라들의 공통 편향을 못 잡는다
(4명한테 물어보고 평균 내서 정답 삼은 뒤 다시 그 4명한테 맞는지 물어보는 것과
같음). 지금은 카메라를 전혀 안 쓰고, **그 세트에서 로봇이 실제로 명령받아
이동한 FK 위치 @ session1에서 구한 T_gripper_cube**로 정답을 만든다 --
완전히 비전과 무관한 값이라 카메라들의 공통 편향까지 잡아낼 수 있다.

Cross-camera consistency: (held-out 아닌) 전체 데이터로 한 fit에서, 같은
세트를 본 카메라들이 각자 독립적으로 계산한 큐브 pose끼리 얼마나 다른지
(평행이동 mm, 회전 deg) -- 카메라들끼리 서로 동의하는 정도.

사용법:
  python eval_heldout_and_consistency.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from calibration_pipeline import se3 as cp  # noqa: E402
from calibration_pipeline.apriltag_cube import inv_T  # noqa: E402
from calibration_pipeline.path_evaluation import solve_observed_pose  # noqa: E402
from calibration_pipeline.reprojection import pose_delta, project_points  # noqa: E402
from calibration_pipeline.table1 import estimate_board_handeye_initial  # noqa: E402

from fit_calibration_methods import (  # noqa: E402
    GRIPPER_LOCAL_ID, SESSION1_DIR_DEFAULT, SESSION3_DIR_DEFAULT, fk_anchor_cubes, load_all_data,
    rmse_px, solve_parallel_fixed, solve_parallel_gripper, solve_unified,
)
from session2_pick_and_place import SESSION2_DIR_DEFAULT  # noqa: E402


def camera_cube_estimate(obs, cams, gtc, robot_T, K_map, D_map, gripper_id):
    T_cam_cube = solve_observed_pose(obs, K_map, D_map)
    if T_cam_cube is None:
        return None
    c = int(obs.cam)
    if c == gripper_id:
        if gtc is None or int(obs.event) not in robot_T:
            return None
        return robot_T[int(obs.event)] @ gtc @ T_cam_cube
    if cams is None or c not in cams:
        return None
    return cams[c] @ T_cam_cube


def joint_cube_estimate(obs_list, cams, gtc, robot_T, K_map, D_map, gripper_id, init=None):
    """held-out 세트의 큐브 pose를 여러 카메라의 코너를 **한 번에** 써서 삼각측량
    (frozen 카메라, 큐브 6-DoF만 변수, canonical soft_l1 2px). 단일 이미지 PnP
    4개를 robust 평균내는 것보다 노이즈가 작은 추정기 -- 모든 방식에 똑같이
    적용되므로 비교 조건이 아니라 평가 정밀도 문제다."""
    from scipy.optimize import least_squares
    from calibration_pipeline.fk_factor import robustify_elementwise
    from calibration_pipeline.reprojection import retract, SE3Scaling
    views = []
    for o in obs_list:
        c = int(o.cam)
        if c == gripper_id:
            if gtc is None or int(o.event) not in robot_T:
                continue
            T_bc = robot_T[int(o.event)] @ gtc
        else:
            if cams is None or c not in cams:
                continue
            T_bc = cams[c]
        views.append((inv_T(T_bc), np.asarray(o.object_points, dtype=np.float64),
                      np.asarray(o.image_points, dtype=np.float64).reshape(-1, 2), K_map[c], D_map[c]))
    if not views:
        return None
    if init is None:
        cands = [camera_cube_estimate(o, cams, gtc, robot_T, K_map, D_map, gripper_id) for o in obs_list]
        cands = [c for c in cands if c is not None]
        if not cands:
            return None
        init = cands[0] if len(cands) == 1 else cp.robust_se3_average(cands, None)[0]
    scaling = SE3Scaling()

    def resid(x):
        T = retract(init, x, scaling)
        out = []
        for T_cb, obj, img, K, D in views:
            out.extend(robustify_elementwise((project_points(T_cb @ T, obj, K, D) - img).reshape(-1), "soft_l1", 2.0))
        return np.asarray(out)
    sol = least_squares(resid, np.zeros(6), method="trf", loss="linear", x_scale="jac", xtol=1e-10, ftol=1e-10, gtol=1e-10)
    return retract(init, sol.x, scaling)


def fit_frozen(method, data_fold, fk_mode, gtc_init, board_init):
    """(cams, gtc) 프리즈된 값 반환 -- 통합/진짜독립 공통 인터페이스.
    진짜독립은 no_fk(estimated)에서만 존재한다."""
    if method == "통합":
        state, _, _ = solve_unified(data_fold, fk_mode, gtc_init, board_init)
        return state.cams, state.gtc
    # method == "독립_true": 고정캠/그리퍼 그룹을 완전히 따로 (핸드오프 없음)
    state_fixed, _, _ = solve_parallel_fixed(data_fold)
    state_gripper, _, _ = solve_parallel_gripper(data_fold, gtc_init, board_init)
    return state_fixed.cams, state_gripper.gtc


def evaluate_heldout(method, data, fk_mode, gtc_init, board_init, robot_T_all, K_map, D_map, set_ids):
    """반환: (heldout px RMSE, 세트별 px, 세트별 mm/deg, heldout_cross)
    heldout_cross = 빠진 세트에 대해서만 잰 cross-view px / cam-common mm/deg를
    15개 fold에 걸쳐 pooled한 것 -- late_table1처럼 학습에 안 쓴 세트 위에서
    카메라 간 일치도를 재는 버전."""
    all_errs = []
    per_set = {}
    per_set_mm_deg = {}
    xview_sq, xview_pairs, xview_dirs = [], 0, 0
    camcom_mm, camcom_deg = [], []
    per_set_joint, per_set_frozen = {}, {}
    obs_all_s2 = data["obs_s2_fixed"] + data["obs_s2_gripper"]
    grasp_init = data["grasp_init"]
    for s in set_ids:
        if s not in data["items_by_index"]:
            continue
        data_fold = dict(data)
        data_fold["obs_s2_fixed"] = [o for o in data["obs_s2_fixed"] if o.set_idx is None or int(o.set_idx) != s]
        data_fold["obs_s2_gripper"] = [o for o in data["obs_s2_gripper"] if o.set_idx is None or int(o.set_idx) != s]
        # session3 큐브 관측은 마지막 placement 세트 변수를 공유하므로 그 세트가 held-out이면 같이 뺀다.
        for key in ("obs_s3_cube", "obs_s3_cube_fixed", "obs_s3_cube_gripper"):
            data_fold[key] = [o for o in data.get(key, []) if o.set_idx is None or int(o.set_idx) != s]
        data_fold["items_by_index"] = {k: v for k, v in data["items_by_index"].items() if k != s}
        try:
            cams, gtc = fit_frozen(method, data_fold, fk_mode, gtc_init, board_init)
        except Exception as exc:
            print(f"    [WARN] {method} fk={fk_mode} set={s}: fold fit 실패 ({exc}), 건너뜀")
            continue

        # 정답 = 카메라 전혀 안 쓰고, 그 세트에서 로봇이 실제로 명령받아 간
        # FK 위치 @ session1의 T_gripper_cube로 계산 (비전 무관, 완전 독립).
        T_gt = fk_anchor_cubes({s: data["items_by_index"][s]}, grasp_init)[s]

        heldout_obs = [o for o in obs_all_s2 if o.set_idx is not None and int(o.set_idx) == s]
        set_errs = []
        for o in heldout_obs:
            c = int(o.cam)
            if c == GRIPPER_LOCAL_ID:
                if int(o.event) not in robot_T_all:
                    continue
                T_base_cam = robot_T_all[int(o.event)] @ gtc
            else:
                if c not in cams:
                    continue
                T_base_cam = cams[c]
            pred = project_points(inv_T(T_base_cam) @ T_gt, o.object_points, K_map[c], D_map[c])
            e = np.linalg.norm(pred - o.image_points, axis=1)
            set_errs.extend(e.tolist())
        if not set_errs:
            continue
        all_errs.extend(set_errs)
        per_set[s] = rmse_px(set_errs)

        # 실제 mm/deg 오차: 이미지만으로(그 세트 자신의 사진 + frozen 카메라)
        # 삼각측량한 큐브 위치 vs 위의 FK 기반(비전 무관) 정답을 직접 비교.
        cands = [camera_cube_estimate(o, cams, gtc, robot_T_all, K_map, D_map, GRIPPER_LOCAL_ID)
                 for o in heldout_obs]
        cands = [c for c in cands if c is not None]
        if cands:
            T_vision = cands[0] if len(cands) == 1 else cp.robust_se3_average(cands, None)[0]
            d_mm, d_deg = pose_delta(T_vision, T_gt)
            per_set_mm_deg[s] = {"translation_mm": d_mm, "rotation_deg": d_deg}
            # 다중 뷰 공동 삼각측량 (노이즈 더 작은 추정기)
            T_joint = joint_cube_estimate(heldout_obs, cams, gtc, robot_T_all, K_map, D_map, GRIPPER_LOCAL_ID, init=T_vision)
            if T_joint is not None:
                j_mm, j_deg = pose_delta(T_joint, T_gt)
                per_set_joint[s] = {"translation_mm": j_mm, "rotation_deg": j_deg}
        per_set_frozen[s] = {"cams": {int(k): np.asarray(v).tolist() for k, v in cams.items()},
                             "gtc": None if gtc is None else np.asarray(gtc).tolist()}

        # held-out 세트에 대해서만 카메라 간 일치도 (이 fold의 frozen 카메라로)
        sq, n_p, n_d = _cross_view_squared(heldout_obs, cams, gtc, robot_T_all, K_map, D_map, GRIPPER_LOCAL_ID)
        xview_sq.extend(sq)
        xview_pairs += n_p
        xview_dirs += n_d
        t_mm, r_deg = _cross_camera_pairs(heldout_obs, cams, gtc, robot_T_all, K_map, D_map, GRIPPER_LOCAL_ID)
        camcom_mm.extend(t_mm)
        camcom_deg.extend(r_deg)

    heldout_cross = {
        "cross_view_cube_pixel_transfer_rmse_px": float(np.sqrt(np.mean(xview_sq))) if xview_sq else float("nan"),
        "cross_view_n_pairs": xview_pairs,
        "cross_view_n_directions": xview_dirs,
        "cam_common_translation_mm": float(np.mean(camcom_mm)) if camcom_mm else float("nan"),
        "cam_common_rotation_deg": float(np.mean(camcom_deg)) if camcom_deg else float("nan"),
        "cam_common_n_pairs": len(camcom_mm),
        "per_set_mm_deg_joint": per_set_joint,
        "heldout_joint_translation_mean_mm": float(np.mean([v["translation_mm"] for v in per_set_joint.values()])) if per_set_joint else float("nan"),
        "heldout_joint_rotation_mean_deg": float(np.mean([v["rotation_deg"] for v in per_set_joint.values()])) if per_set_joint else float("nan"),
        "per_set_frozen": per_set_frozen,
    }
    return rmse_px(all_errs), per_set, per_set_mm_deg, heldout_cross


def _cross_camera_pairs(obs_list, cams, gtc, robot_T_all, K_map, D_map, gripper_id):
    """세트별로 카메라 pairwise (mm, deg) 차이 리스트를 그대로 반환 (평균 안 냄)."""
    by_set = {}
    for o in obs_list:
        if o.set_idx is None:
            continue
        by_set.setdefault(int(o.set_idx), []).append(o)
    trans_mm, rot_deg = [], []
    for s, group in by_set.items():
        cam_pose = {}
        for o in group:
            est = camera_cube_estimate(o, cams, gtc, robot_T_all, K_map, D_map, gripper_id)
            if est is not None:
                cam_pose[int(o.cam)] = est  # 세트당 카메라 1관측 가정 (session2 구조상 맞음)
        cam_ids = sorted(cam_pose)
        for i in range(len(cam_ids)):
            for j in range(i + 1, len(cam_ids)):
                d_mm, d_deg = pose_delta(cam_pose[cam_ids[i]], cam_pose[cam_ids[j]])
                trans_mm.append(d_mm)
                rot_deg.append(d_deg)
    return trans_mm, rot_deg


def cross_camera_consistency(obs_all_s2, cams, gtc, robot_T_all, K_map, D_map, gripper_id):
    trans_mm, rot_deg = _cross_camera_pairs(obs_all_s2, cams, gtc, robot_T_all, K_map, D_map, gripper_id)
    return (float(np.mean(trans_mm)) if trans_mm else float("nan"),
            float(np.mean(rot_deg)) if rot_deg else float("nan"), len(trans_mm))


def _cross_view_squared(obs_list, cams, gtc, robot_T_all, K_map, D_map, gripper_id):
    """cross_view_pixel_transfer의 재료: 성분별 제곱오차 리스트, 쌍 수, 방향 수."""
    by_set = {}
    for o in obs_list:
        if o.set_idx is None:
            continue
        by_set.setdefault(int(o.set_idx), []).append(o)

    def base_cam_pose(o):
        c = int(o.cam)
        if c == gripper_id:
            if gtc is None or int(o.event) not in robot_T_all:
                return None
            return robot_T_all[int(o.event)] @ gtc
        return cams.get(c)

    sq_all, n_directions, n_pairs = [], 0, 0
    euclid_all = []   # 코너별 |e| (평균 지표용)
    for s, group in by_set.items():
        entries = []
        for o in group:
            T_base_cam = base_cam_pose(o)
            T_cam_cube = solve_observed_pose(o, K_map, D_map)
            if T_base_cam is None or T_cam_cube is None:
                continue
            entries.append((o, T_base_cam, T_base_cam @ T_cam_cube))
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                n_pairs += 1
                for (src, _, T_base_cube_src), (dst, T_base_cam_dst, _) in ((entries[i], entries[j]), (entries[j], entries[i])):
                    c = int(dst.cam)
                    pred = project_points(inv_T(T_base_cam_dst) @ T_base_cube_src, dst.object_points, K_map[c], D_map[c])
                    diff = pred - np.asarray(dst.image_points).reshape(-1, 2)
                    sq_all.extend(np.square(diff).reshape(-1).tolist())
                    euclid_all.extend(np.linalg.norm(diff, axis=1).tolist())
                    n_directions += 1
    _cross_view_squared.last_euclid = euclid_all
    return sq_all, n_pairs, n_directions


def cross_view_pixel_transfer_mean(obs_all_s2, cams, gtc, robot_T_all, K_map, D_map, gripper_id):
    """Cross-view의 '평균' 버전: 옮겨 재투영한 코너별 유클리드 오차 |e|의 평균 (px)."""
    _cross_view_squared(obs_all_s2, cams, gtc, robot_T_all, K_map, D_map, gripper_id)
    e = np.asarray(_cross_view_squared.last_euclid)
    return (float(np.mean(e)) if e.size else float("nan")), (float(np.percentile(e, 95)) if e.size else float("nan"))


def corner_consistency_3d(obs_all_s2, cams, gtc, robot_T_all, K_map, D_map, gripper_id, cube_model):
    """3D 코너 일관성 오차: 공통 시야 카메라 쌍(A, B)이 각자 추정한 큐브 pose로 큐브
    코너 24개를 base 좌표계에 놓고, 대응 코너끼리의 3D 거리(mm)를 평균. Cam-common이
    pose 차이(평행이동/회전 따로)라면 이건 둘을 코너 위치 하나로 합친 값."""
    corners = np.vstack([cube_model.marker_corners_in_rig(m) for m in sorted(cube_model.cfg.id_to_face)])
    corners_h = np.c_[corners, np.ones(len(corners))]
    by_set = {}
    for o in obs_all_s2:
        if o.set_idx is not None:
            by_set.setdefault(int(o.set_idx), []).append(o)
    d_all, d_ff, d_gf = [], [], []
    for s, group in by_set.items():
        est = {}
        for o in group:
            T = camera_cube_estimate(o, cams, gtc, robot_T_all, K_map, D_map, gripper_id)
            if T is not None:
                est[int(o.cam)] = (T @ corners_h.T).T[:, :3] * 1000.0
        ids = sorted(est)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                d = np.linalg.norm(est[ids[i]] - est[ids[j]], axis=1)
                d_all.extend(d.tolist())
                (d_gf if gripper_id in (ids[i], ids[j]) else d_ff).extend(d.tolist())
    f = lambda v: float(np.mean(v)) if v else float("nan")
    return {"mean_mm": f(d_all), "rms_mm": float(np.sqrt(np.mean(np.square(d_all)))) if d_all else float("nan"),
            "p95_mm": float(np.percentile(d_all, 95)) if d_all else float("nan"),
            "fixed_fixed_mean_mm": f(d_ff), "gripper_fixed_mean_mm": f(d_gf), "n_pairs": len(d_all) // max(1, len(corners))}


def cross_view_pixel_transfer(obs_all_s2, cams, gtc, robot_T_all, K_map, D_map, gripper_id):
    """late_table1의 "Cross-view Cube px"(cross_view_cube_pixel_transfer_rmse_px)와
    같은 정의: 같은 세트를 본 두 카메라 A, B에 대해, A 이미지 한 장만으로 PnP한
    큐브 pose를 캘리브레이션된 extrinsics로 B 카메라 좌표계로 옮겨서
    (inv(T_base_camB) @ T_base_camA @ T_camA_cube) B 이미지에 재투영하고, B가
    실제로 검출한 코너와의 픽셀 오차를 잰다. A->B, B->A 양방향, 고정캠-고정캠
    쌍과 그리퍼캠-고정캠 쌍을 한 지표에 같이 모아서(late_table1과 동일하게
    pooled) 성분별(dx,dy 펴서) RMSE. 그리퍼캠의 T_base_cam은 robot_T[event] @ gtc.

    Cam-common(mm/deg)이 "두 카메라의 3D pose 추정치 차이"라면, 이건 "한쪽
    pose를 다른 쪽 이미지로 가져갔을 때 픽셀이 얼마나 어긋나는가"라 카메라 간
    상대 extrinsics를 더 직접적으로 본다. 둘 다 카메라 공통 편향은 못 잡는다."""
    sq_all, n_pairs, n_directions = _cross_view_squared(
        obs_all_s2, cams, gtc, robot_T_all, K_map, D_map, gripper_id)
    return (float(np.sqrt(np.mean(sq_all))) if sq_all else float("nan")), n_pairs, n_directions


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
    ap.add_argument("--out", default=str(REPO_ROOT / "zeus_gello_calibration" / "heldout_and_consistency.json"))
    ap.add_argument("--cube-only", action="store_true",
                    help="보드 관측 전부 제외(session3 미사용 + session2 보드 제외); 통합 2조건만 (독립은 식별 불가)")
    ap.add_argument("--cube-config", default=None, help="큐브 마커 config JSON (GT 큐브 촬영이면 targets/gt_cube/cube_config.json)")
    ap.add_argument("--s3-gripper-only", action="store_true", help="session3는 그리퍼캠 관측만 사용")
    ap.add_argument("--no-s3-cube", action="store_true", help="session3에 찍힌(정지) 큐브 관측을 쓰지 않음")
    args = ap.parse_args()

    data = load_all_data(args)
    K_map, D_map = data["K_map"], data["D_map"]
    set_ids = sorted(data["items_by_index"])
    robot_T_all = {**data["robot_T_s1"], **data["robot_T_s2_gripper"], **data["robot_T_s3"]}
    obs_all_s2 = data["obs_s2_fixed"] + data["obs_s2_gripper"]

    if args.cube_only:
        from fit_calibration_methods import init_gtc_from_cubes, strip_board_data
        data = strip_board_data(data)
        gtc_init, _ = init_gtc_from_cubes(data)
        board_init = None
        conditions = (("통합", "no_fk", "통합_no-fk_cubeonly"), ("통합", "fixed_fk", "통합_raw-fk_cubeonly"))
        args.out = str(Path(args.out).with_name(Path(args.out).stem + "_cubeonly.json"))
    else:
        gtc_init, board_init, _eih_diag = estimate_board_handeye_initial(
            data["obs_s3"], data["robot_T_s3"], K_map, D_map, GRIPPER_LOCAL_ID)
        conditions = (("통합", "no_fk", "통합_no-fk"), ("통합", "fixed_fk", "통합_raw-fk"),
                      ("독립_true", "no_fk", "독립_no-fk"))

    results = {}
    for method, fk_mode, label in conditions:
        print(f"[{label}] leave-one-out held-out 계산 중 ({len(set_ids)}개 세트)...")
        heldout_rmse, per_set, per_set_mm_deg, heldout_cross = evaluate_heldout(
            method, data, fk_mode, gtc_init, board_init, robot_T_all, K_map, D_map, set_ids)
        heldout_mm = [v["translation_mm"] for v in per_set_mm_deg.values()]
        heldout_deg = [v["rotation_deg"] for v in per_set_mm_deg.values()]

        cams_full, gtc_full = fit_frozen(method, data, fk_mode, gtc_init, board_init)
        trans_mm, rot_deg, n_pairs = cross_camera_consistency(
            obs_all_s2, cams_full, gtc_full, robot_T_all, K_map, D_map, GRIPPER_LOCAL_ID)
        xview_px, xview_pairs, xview_dirs = cross_view_pixel_transfer(
            obs_all_s2, cams_full, gtc_full, robot_T_all, K_map, D_map, GRIPPER_LOCAL_ID)
        xview_mean_px, xview_p95_px = cross_view_pixel_transfer_mean(
            obs_all_s2, cams_full, gtc_full, robot_T_all, K_map, D_map, GRIPPER_LOCAL_ID)
        from calibration_pipeline.apriltag_cube import AprilTagCubeTarget
        from calibration_pipeline.config import get_default_cube_config
        corner3d = corner_consistency_3d(obs_all_s2, cams_full, gtc_full, robot_T_all, K_map, D_map,
                                         GRIPPER_LOCAL_ID, AprilTagCubeTarget(get_default_cube_config()).model)

        results[label] = {
            "cross_view_mean_px": xview_mean_px,
            "cross_view_p95_px": xview_p95_px,
            "corner_consistency_3d": corner3d,
            "heldout_cube_rmse_px": heldout_rmse,
            "n_heldout_sets": len(per_set),
            "heldout_translation_mean_mm": float(np.mean(heldout_mm)) if heldout_mm else float("nan"),
            "heldout_translation_max_mm": float(np.max(heldout_mm)) if heldout_mm else float("nan"),
            "heldout_rotation_mean_deg": float(np.mean(heldout_deg)) if heldout_deg else float("nan"),
            "heldout_rotation_max_deg": float(np.max(heldout_deg)) if heldout_deg else float("nan"),
            "cross_camera_translation_mm": trans_mm,
            "cross_camera_rotation_deg": rot_deg,
            "n_camera_pairs": n_pairs,
            "cross_view_cube_pixel_transfer_rmse_px": xview_px,
            "cross_view_n_pairs": xview_pairs,
            "cross_view_n_directions": xview_dirs,
            "heldout_cross": heldout_cross,
            "per_set_heldout_rmse_px": per_set,
            "per_set_heldout_mm_deg": per_set_mm_deg,
        }

    print(f"\n[train-pooled: 전체 데이터 fit에서 잰 카메라 간 일치도]")
    print(f"{'condition':>16} {'heldout_px':>11} {'heldout_mm':>11} {'heldout_deg':>12} "
          f"{'cross_view_px':>14} {'cross_cam_mm':>13} {'cross_cam_deg':>14}")
    for name in results:
        r = results[name]
        print(f"{name:>16} {r['heldout_cube_rmse_px']:>11.4f} "
              f"{r['heldout_translation_mean_mm']:>11.2f} {r['heldout_rotation_mean_deg']:>12.2f} "
              f"{r['cross_view_cube_pixel_transfer_rmse_px']:>14.4f} "
              f"{r['cross_camera_translation_mm']:>13.4f} {r['cross_camera_rotation_deg']:>14.4f}")
    print(f"\n[held-out: 빠진 세트에 대해서만 잰 카메라 간 일치도, 15 fold pooled / joint = 다중뷰 공동 삼각측량 held-out]")
    print(f"{'condition':>16} {'xview_px':>10} {'xview_pairs':>12} {'camcom_mm':>10} {'camcom_deg':>11} {'camcom_pairs':>13} {'joint_mm':>9} {'joint_deg':>10}")
    for name in results:
        h = results[name]["heldout_cross"]
        print(f"{name:>16} {h['cross_view_cube_pixel_transfer_rmse_px']:>10.4f} {h['cross_view_n_pairs']:>12d} "
              f"{h['cam_common_translation_mm']:>10.4f} {h['cam_common_rotation_deg']:>11.4f} {h['cam_common_n_pairs']:>13d} "
              f"{h['heldout_joint_translation_mean_mm']:>9.3f} {h['heldout_joint_rotation_mean_deg']:>10.3f}")

    Path(args.out).write_text(json.dumps({"results": results}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
