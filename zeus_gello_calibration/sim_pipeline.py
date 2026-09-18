#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zeus_gello_calibration/sim_pipeline.py -- Zeus 캘리브레이션 파이프라인 시뮬레이터.

목적: (1) 노이즈 0에서 알고리즘 무결성(정답 복원) 확인, (2) 어떤 노이즈에
어느 방식이 취약한지 한 요인씩 넣어보며 확인.

핵심 설계: 시뮬레이터는 `fit_calibration_methods.load_all_data()`가 돌려주는
것과 **똑같은 `data` 딕셔너리**를 합성한다. 그래서 통합/독립/raw-fk solver,
held-out/일치도 평가, corrected-FK 변형이 코드 변경 없이 그대로 돌아가고,
정답을 알기 때문에 카메라 extrinsics·T_gripper_cam·T_gripper_cube·큐브 pose의
**진짜 오차**를 직접 잰다.

장면(ground truth)은 실제 fit/촬영값에서 가져온다(현실적 기하 유지):
  고정캠 3대 extrinsics, T_gripper_cam, T_gripper_cube  <- fit_통합_no-fk.json
  보드 pose                                             <- 실제 보드 관측으로 초기화한 값
  로봇 궤적(session1 16자세, session2 place 15 + 촬영 자세, session3 15자세) <- 실제 기록
  K/D                                                   <- intrinsics 파일
  큐브/보드 모델                                        <- calibration_pipeline 설정

세션 재현:
  session1: FK_e @ T_gripper_cube 위치의 큐브를 고정캠 3대가 봄 (grasp+FK 모델)
  session2: FK_place_s @ T_gripper_cube (+ 릴리즈 슬립) 위치의 큐브를 고정캠 3대 +
            촬영 자세의 그리퍼캠이 봄; 같은 프레임에서 보드도 봄
  session3: 손목 자세 15개에서 그리퍼캠(+고정캠)이 보드를 봄

노이즈 (각각 계통 sys + 랜덤 rand; 단위 mm / deg / px):
  fk        로봇 FK 절대오차: solver에 주는 robot_T = exp(sys, base 프레임 상수) @ T_true @ exp(rand_e)
  place     릴리즈 슬립: 실제 놓인 큐브 = anchor_true @ exp(sys, 그리퍼 프레임) @ exp(rand_s)
  cam_ext   카메라 extrinsic 초기값 오차(cam_init에 sys+rand) + 세션 간 드리프트(cam_drift:
            session2/3의 진짜 고정캠 위치가 session1과 다름 -- solver는 고정이라 가정)
  cam_int   내부계수 오차: solver K = 진짜 K에 초점거리 배율(1+sys+rand), 주점 이동(px)
  pixel     코너 검출 노이즈(rand px) + outlier 비율(마커 단위 큰 오차)
  geom      큐브 마커 위치 오차(solver object_points = 진짜 + sys/rand mm) -- GT 큐브 config 문제 모사
  grasp     session1에서 구한 T_gripper_cube(=grasp_init)의 오차(sys+rand) -- raw-fk 앵커/held-out GT에 들어감

사용법:
  python sim_pipeline.py --build-scene            # 실제 데이터에서 sim_scene.json 생성 (1회)
  python sim_pipeline.py --run                    # 노이즈 0 무결성 검사
  python sim_pipeline.py --run --pixel-rand 0.3 --fk-sys 1.0 --place-rand 0.5
  python sim_pipeline.py --sweep --seeds 3        # 요인별 민감도 표
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from calibration_pipeline.apriltag_cube import AprilTagCubeTarget, inv_T  # noqa: E402
from calibration_pipeline.board_config import charuco_config_from_dict  # noqa: E402
from calibration_pipeline.charuco import CharucoTarget  # noqa: E402
from calibration_pipeline.config import get_default_cube_config  # noqa: E402
from calibration_pipeline.reprojection import PixelObs, pose_delta, project_points  # noqa: E402
from calibration_pipeline.table1 import estimate_board_handeye_initial  # noqa: E402
from robot.backends.zeus_client import T_to_pose6, pose6_to_T  # noqa: E402

import eval_heldout_and_consistency as ehc  # noqa: E402
from fit_calibration_methods import (  # noqa: E402
    GRIPPER_LOCAL_ID, SESSION1_DIR_DEFAULT, SESSION3_DIR_DEFAULT, fk_anchor_cubes, init_board_pose,
    load_all_data, solve_parallel_fixed, solve_parallel_gripper, solve_unified,
)
from fit_full_calibration import CHARUCO_BOARD_CONFIG, SESSION3_EVENT_OFFSET  # noqa: E402
from fit_placement_fk_ablation import SESSION2_EVENT_OFFSET  # noqa: E402
from session2_pick_and_place import SESSION2_DIR_DEFAULT  # noqa: E402

SCENE_JSON_DEFAULT = REPO_ROOT / "zeus_gello_calibration" / "results" / "sim" / "sim_scene.json"
IMAGE_W, IMAGE_H = 1280, 720
FIXED_MIN_CORNERS = 8
BOARD_MIN_CORNERS = 12


# ---------------------------------------------------------------- SE(3)
def se3_exp(rot_deg, trans_mm):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(np.deg2rad(np.asarray(rot_deg, float))).as_matrix()
    T[:3, 3] = np.asarray(trans_mm, float) / 1000.0
    return T


def rand_se3(rng, rot_deg_sigma, trans_mm_sigma):
    return se3_exp(rng.normal(0, rot_deg_sigma, 3) if rot_deg_sigma > 0 else np.zeros(3),
                   rng.normal(0, trans_mm_sigma, 3) if trans_mm_sigma > 0 else np.zeros(3))


def sys_se3(rng, rot_deg_mag, trans_mm_mag):
    """크기가 정확히 mag인 고정 방향(seed로 정해지는) 계통 오프셋."""
    if rot_deg_mag <= 0 and trans_mm_mag <= 0:
        return np.eye(4)
    r = rng.normal(0, 1, 3); r = r / np.linalg.norm(r) * rot_deg_mag
    t = rng.normal(0, 1, 3); t = t / np.linalg.norm(t) * trans_mm_mag
    return se3_exp(r, t)


# ---------------------------------------------------------------- 장면 구축
def build_scene(args):
    """실제 데이터/fit에서 정답 장면을 뽑아 JSON으로 저장."""
    data = load_all_data(args)
    K_map, D_map = data["K_map"], data["D_map"]
    fit = json.loads((REPO_ROOT / "zeus_gello_calibration" / "results" / "fits" / "fit_통합_no-fk.json").read_text())
    cams = {int(k.split("_", 1)[0]): np.asarray(v, float) for k, v in fit["T_base_cam"].items()}
    gtc = np.asarray(fit["T_gripper_cam"], float)
    grasp = np.asarray(fit["T_gripper_cube"], float)
    board = init_board_pose(data["obs_s2_board_fixed"] + data["obs_s3_fixed"], K_map, D_map, cams)
    scene = {
        "cams": {str(k): v.tolist() for k, v in cams.items()},
        "gtc": gtc.tolist(), "grasp": grasp.tolist(), "board": board.tolist(),
        "K": {str(k): np.asarray(v, float).tolist() for k, v in K_map.items()},
        "D": {str(k): np.asarray(v, float).reshape(-1).tolist() for k, v in D_map.items()},
        "s1_poses": {str(k): [float(x) for x in T_to_pose6(v)] for k, v in data["robot_T_s1"].items()},
        "s2_place_targets": {str(k): list(map(float, v["target"])) for k, v in data["items_by_index"].items()},
        "s2_photo_poses": {str(k): [float(x) for x in T_to_pose6(v)] for k, v in data["robot_T_s2_gripper"].items()},
        "s3_poses": {str(k): [float(x) for x in T_to_pose6(v)] for k, v in data["robot_T_s3"].items()},
    }
    Path(args.scene).write_text(json.dumps(scene, indent=1))
    print(f"scene saved -> {args.scene}  (fixed cams {sorted(cams)}, s1 {len(scene['s1_poses'])}, "
          f"s2 {len(scene['s2_place_targets'])}, s3 {len(scene['s3_poses'])})")


def load_scene(path):
    s = json.loads(Path(path).read_text())
    return dict(
        cams={int(k): np.asarray(v) for k, v in s["cams"].items()},
        gtc=np.asarray(s["gtc"]), grasp=np.asarray(s["grasp"]), board=np.asarray(s["board"]),
        K={int(k): np.asarray(v) for k, v in s["K"].items()},
        D={int(k): np.asarray(v).reshape(-1, 1) for k, v in s["D"].items()},
        s1={int(k): pose6_to_T(v) for k, v in s["s1_poses"].items()},
        s2_place={int(k): list(v) for k, v in s["s2_place_targets"].items()},
        s2_photo={int(k): pose6_to_T(v) for k, v in s["s2_photo_poses"].items()},
        s3={int(k): pose6_to_T(v) for k, v in s["s3_poses"].items()},
    )


# ---------------------------------------------------------------- 노이즈 설정
@dataclass
class NoiseConfig:
    fk_sys_mm: float = 0.0; fk_sys_deg: float = 0.0; fk_rand_mm: float = 0.0; fk_rand_deg: float = 0.0
    place_sys_mm: float = 0.0; place_sys_deg: float = 0.0; place_rand_mm: float = 0.0; place_rand_deg: float = 0.0
    cam_ext_sys_mm: float = 0.0; cam_ext_sys_deg: float = 0.0; cam_ext_rand_mm: float = 0.0; cam_ext_rand_deg: float = 0.0
    cam_drift_mm: float = 0.0; cam_drift_deg: float = 0.0
    cam_int_sys_focal: float = 0.0; cam_int_rand_focal: float = 0.0; cam_int_pp_px: float = 0.0
    pixel_rand_px: float = 0.0; pixel_outlier_frac: float = 0.0; pixel_outlier_px: float = 15.0
    geom_sys_mm: float = 0.0; geom_rand_mm: float = 0.0
    grasp_sys_mm: float = 0.0; grasp_sys_deg: float = 0.0; grasp_rand_mm: float = 0.0; grasp_rand_deg: float = 0.0
    gtc_init_mm: float = 5.0; gtc_init_deg: float = 2.0     # 그리퍼캠 초기값은 실제 파이프라인처럼 session3에서 추정 (fallback용)
    seed: int = 0


# ---------------------------------------------------------------- 관측 합성
class Synthesizer:
    def __init__(self, scene, noise: NoiseConfig):
        self.s, self.n = scene, noise
        self.rng = np.random.default_rng(noise.seed)
        self.cube = AprilTagCubeTarget(get_default_cube_config()).model
        self.board_target = CharucoTarget(charuco_config_from_dict(CHARUCO_BOARD_CONFIG))
        b = self.board_target.board
        self.board_pts = np.asarray(b.getChessboardCorners() if hasattr(b, "getChessboardCorners") else b.chessboardCorners,
                                    float).reshape(-1, 3)
        self.marker_ids = sorted(self.cube.cfg.id_to_face)
        # 큐브 마커 기하: 진짜(합성용) vs solver가 아는 값(geom 오차)
        self.marker_true = {m: self.cube.marker_corners_in_rig(m) for m in self.marker_ids}
        g_sys = self.rng.normal(0, 1, 3); g_sys = g_sys / np.linalg.norm(g_sys) * noise.geom_sys_mm / 1000.0
        self.marker_solver = {m: self.marker_true[m] + g_sys + (self.rng.normal(0, noise.geom_rand_mm / 1000.0, 3) if noise.geom_rand_mm > 0 else 0)
                              for m in self.marker_ids}
        # FK 계통 오프셋 (base 프레임), 릴리즈 계통 오프셋 (그리퍼 프레임), 드리프트
        self.fk_sys = sys_se3(self.rng, noise.fk_sys_deg, noise.fk_sys_mm)
        self.place_sys = sys_se3(self.rng, noise.place_sys_deg, noise.place_sys_mm)
        self.drift = {c: rand_se3(self.rng, noise.cam_drift_deg, noise.cam_drift_mm) for c in scene["cams"]}
        # solver용 K (내부계수 오차)
        self.K_solver = {}
        for c, K in scene["K"].items():
            f = 1.0 + noise.cam_int_sys_focal + (self.rng.normal(0, noise.cam_int_rand_focal) if noise.cam_int_rand_focal > 0 else 0)
            K2 = K.copy(); K2[0, 0] *= f; K2[1, 1] *= f
            if noise.cam_int_pp_px > 0:
                K2[0, 2] += self.rng.normal(0, noise.cam_int_pp_px); K2[1, 2] += self.rng.normal(0, noise.cam_int_pp_px)
            self.K_solver[c] = K2
        self.gt = {"cube_s2": {}, "fk_true": {}, "fk_solver": {}}

    # 진짜 pose에 FK 오차를 얹어 solver에 줄 pose
    def fk_solver(self, T_true):
        return self.fk_sys @ T_true @ rand_se3(self.rng, self.n.fk_rand_deg, self.n.fk_rand_mm)

    def _pixels(self, T_cam_target, pts_true, K, D):
        img = project_points(T_cam_target, pts_true, K, D)
        if self.n.pixel_rand_px > 0:
            img = img + self.rng.normal(0, self.n.pixel_rand_px, img.shape)
        return img

    def cube_obs(self, T_base_cam_true, T_base_cube_true, cam, event, set_idx, grasp_idx, min_corners):
        T_cc = inv_T(T_base_cam_true) @ T_base_cube_true
        K_true, D = self.s["K"][cam], self.s["D"][cam]
        obj_solver, img = [], []
        for m in self.marker_ids:
            vis, _ = self.cube.marker_visibility_score(m, T_cc)
            if not vis:
                continue
            px = self._pixels(T_cc, self.marker_true[m], K_true, D)
            if np.any(px < 5) or np.any(px[:, 0] > IMAGE_W - 5) or np.any(px[:, 1] > IMAGE_H - 5):
                continue
            side = np.linalg.norm(px[0] - px[1])
            if side < 12:
                continue
            if self.n.pixel_outlier_frac > 0 and self.rng.random() < self.n.pixel_outlier_frac:
                px = px + self.rng.normal(0, self.n.pixel_outlier_px, px.shape)
            obj_solver.append(self.marker_solver[m]); img.append(px)
        if not obj_solver or sum(len(o) for o in obj_solver) < min_corners:
            return None
        return PixelObs(marker="cube", cam=int(cam), event=int(event), set_idx=set_idx,
                        object_points=np.vstack(obj_solver), image_points=np.vstack(img), grasp_idx=grasp_idx)

    def board_obs(self, T_base_cam_true, cam, event):
        T_cb = inv_T(T_base_cam_true) @ self.s["board"]
        if (T_cb[:3, :3] @ np.array([0, 0, 1.0]))[2] > 0:  # 보드 앞면이 카메라를 향하지 않으면
            pass
        K_true, D = self.s["K"][cam], self.s["D"][cam]
        px = self._pixels(T_cb, self.board_pts, K_true, D)
        keep = (px[:, 0] > 5) & (px[:, 0] < IMAGE_W - 5) & (px[:, 1] > 5) & (px[:, 1] < IMAGE_H - 5)
        # 카메라 앞쪽(z>0)만
        z = (T_cb @ np.c_[self.board_pts, np.ones(len(self.board_pts))].T).T[:, 2]
        keep &= z > 0.05
        if keep.sum() < BOARD_MIN_CORNERS:
            return None
        return PixelObs(marker="board", cam=int(cam), event=int(event), set_idx=None,
                        object_points=self.board_pts[keep].copy(), image_points=px[keep], grasp_idx=None)

    def cams_true(self, session):
        """세션별 진짜 고정캠 extrinsics (session1 기준, session2/3는 드리프트)."""
        if session == 1:
            return dict(self.s["cams"])
        return {c: self.drift[c] @ T for c, T in self.s["cams"].items()}

    def synthesize(self):
        s, n, rng = self.s, self.n, self.rng
        fixed_ids = sorted(s["cams"])
        grasp_true = s["grasp"]
        grasp_init = sys_se3(rng, n.grasp_sys_deg, n.grasp_sys_mm) @ grasp_true @ rand_se3(rng, n.grasp_rand_deg, n.grasp_rand_mm)
        cam_init = {c: sys_se3(rng, n.cam_ext_sys_deg, n.cam_ext_sys_mm) @ T @ rand_se3(rng, n.cam_ext_rand_deg, n.cam_ext_rand_mm)
                    for c, T in s["cams"].items()}

        # session1: 쥔 큐브, 고정캠
        obs_s1, robot_T_s1 = [], {}
        cams1 = self.cams_true(1)
        for e, T_fl in s["s1"].items():
            T_cube = T_fl @ grasp_true
            robot_T_s1[e] = self.fk_solver(T_fl)
            for c in fixed_ids:
                o = self.cube_obs(cams1[c], T_cube, c, e, None, 0, FIXED_MIN_CORNERS)
                if o is not None:
                    obs_s1.append(o)

        # session2: 놓인 큐브(릴리즈 슬립) + 보드, 고정캠 + 그리퍼캠(촬영 자세)
        cams2 = self.cams_true(2)
        obs_s2_fixed, obs_s2_gripper, obs_s2_board, robot_T_s2_gripper, items = [], [], [], {}, {}
        for sidx, tgt in s["s2_place"].items():
            e = SESSION2_EVENT_OFFSET + sidx
            T_place_true = pose6_to_T(tgt)
            T_cube_true = T_place_true @ grasp_true @ self.place_sys @ rand_se3(rng, n.place_rand_deg, n.place_rand_mm)
            self.gt["cube_s2"][sidx] = T_cube_true
            # solver가 아는 place 명령 pose = FK 오차가 실린 값
            items[sidx] = {"target": T_to_pose6(self.fk_solver(T_place_true)), "index": sidx}
            for c in fixed_ids:
                o = self.cube_obs(cams2[c], T_cube_true, c, e, sidx, None, FIXED_MIN_CORNERS)
                if o is not None:
                    obs_s2_fixed.append(o)
                b = self.board_obs(cams2[c], c, e)
                if b is not None:
                    obs_s2_board.append(b)
            T_photo_true = s["s2_photo"].get(e)
            if T_photo_true is not None:
                T_cam_g = T_photo_true @ s["gtc"]
                robot_T_s2_gripper[e] = self.fk_solver(T_photo_true)
                o = self.cube_obs(T_cam_g, T_cube_true, GRIPPER_LOCAL_ID, e, sidx, None, 4)
                if o is not None:
                    obs_s2_gripper.append(o)
                b = self.board_obs(T_cam_g, GRIPPER_LOCAL_ID, e)
                if b is not None:
                    obs_s2_board.append(b)

        # session3: 보드, 그리퍼캠 + 고정캠
        cams3 = self.cams_true(3)
        obs_s3, robot_T_s3 = [], {}
        for e, T_fl in s["s3"].items():
            robot_T_s3[e] = self.fk_solver(T_fl)
            b = self.board_obs(T_fl @ s["gtc"], GRIPPER_LOCAL_ID, e)
            if b is not None:
                obs_s3.append(b)
            for c in fixed_ids:
                b = self.board_obs(cams3[c], c, e)
                if b is not None:
                    obs_s3.append(b)

        data = dict(
            cam_init=cam_init, grasp_init=grasp_init, K_map=self.K_solver, D_map=dict(s["D"]),
            obs_s1=obs_s1, robot_T_s1=robot_T_s1,
            obs_s2_fixed=obs_s2_fixed, obs_s2_gripper=obs_s2_gripper, robot_T_s2_gripper=robot_T_s2_gripper,
            obs_s2_board=obs_s2_board,
            obs_s2_board_fixed=[o for o in obs_s2_board if int(o.cam) != GRIPPER_LOCAL_ID],
            obs_s2_board_gripper=[o for o in obs_s2_board if int(o.cam) == GRIPPER_LOCAL_ID],
            obs_s3=obs_s3, obs_s3_fixed=[o for o in obs_s3 if int(o.cam) != GRIPPER_LOCAL_ID],
            obs_s3_gripper=[o for o in obs_s3 if int(o.cam) == GRIPPER_LOCAL_ID], robot_T_s3=robot_T_s3,
            items_by_index=items,
        )
        gt = dict(cams=dict(s["cams"]), cams_s2=cams2, gtc=s["gtc"], grasp=grasp_true, board=s["board"], cube_s2=self.gt["cube_s2"])
        return data, gt


# ---------------------------------------------------------------- 평가
def true_errors(state_cams, gtc, grasp, cubes, gt):
    out = {}
    if state_cams:
        errs = [pose_delta(state_cams[c], gt["cams_s2"][c]) for c in gt["cams"] if c in state_cams]
        out["cam_mm"] = float(np.mean([e[0] for e in errs])); out["cam_deg"] = float(np.mean([e[1] for e in errs]))
    if gtc is not None:
        out["gtc_mm"], out["gtc_deg"] = pose_delta(gtc, gt["gtc"])
    if grasp is not None:
        out["grasp_mm"], out["grasp_deg"] = pose_delta(grasp, gt["grasp"])
    if cubes:
        errs = [pose_delta(cubes[k], gt["cube_s2"][k]) for k in gt["cube_s2"] if k in cubes]
        if errs:
            out["cube_mm"] = float(np.mean([e[0] for e in errs])); out["cube_deg"] = float(np.mean([e[1] for e in errs]))
    return out


def run_once(scene, noise: NoiseConfig, with_consistency=True, verbose=False):
    syn = Synthesizer(scene, noise)
    data, gt = syn.synthesize()
    K_map, D_map = data["K_map"], data["D_map"]
    counts = {k: len(data[k]) for k in ("obs_s1", "obs_s2_fixed", "obs_s2_gripper", "obs_s2_board", "obs_s3")}
    gtc_init, board_init, _ = estimate_board_handeye_initial(data["obs_s3"], data["robot_T_s3"], K_map, D_map, GRIPPER_LOCAL_ID)
    robot_T_all = {**data["robot_T_s1"], **data["robot_T_s2_gripper"], **data["robot_T_s3"]}
    obs_all_s2 = data["obs_s2_fixed"] + data["obs_s2_gripper"]
    results = {"counts": counts}
    for fk_mode, label in (("no_fk", "통합_no-fk"), ("fixed_fk", "통합_raw-fk")):
        st, diag, _ = solve_unified(data, fk_mode, gtc_init, board_init)
        r = true_errors(st.cams, st.gtc, st.grasps[0], st.cubes if fk_mode == "no_fk" else None, gt)
        r["train_px"] = diag["train_reprojection_rmse_px"]; r["success"] = diag["success"]
        if with_consistency:
            r["cross_view_mean_px"], _ = ehc.cross_view_pixel_transfer_mean(obs_all_s2, st.cams, st.gtc, robot_T_all, K_map, D_map, GRIPPER_LOCAL_ID)
            r["cam_common_mm"], r["cam_common_deg"], _ = ehc.cross_camera_consistency(obs_all_s2, st.cams, st.gtc, robot_T_all, K_map, D_map, GRIPPER_LOCAL_ID)
        results[label] = r
    try:
        sf, df, _ = solve_parallel_fixed(data)
        sg, dg, _ = solve_parallel_gripper(data, gtc_init, board_init)
        r = true_errors(sf.cams, sg.gtc, sf.grasps[0], sf.cubes, gt)
        r["train_px"] = (df["train_reprojection_rmse_px"] + dg["train_reprojection_rmse_px"]) / 2
        r["success"] = bool(df["success"] and dg["success"])
        if with_consistency:
            r["cross_view_mean_px"], _ = ehc.cross_view_pixel_transfer_mean(obs_all_s2, sf.cams, sg.gtc, robot_T_all, K_map, D_map, GRIPPER_LOCAL_ID)
            r["cam_common_mm"], r["cam_common_deg"], _ = ehc.cross_camera_consistency(obs_all_s2, sf.cams, sg.gtc, robot_T_all, K_map, D_map, GRIPPER_LOCAL_ID)
        results["독립_no-fk"] = r
    except Exception as exc:
        results["독립_no-fk"] = {"error": str(exc)}
    if verbose:
        print(f"  obs: {counts}")
        for k, r in results.items():
            if k == "counts":
                continue
            if "error" in r:
                print(f"  {k:>12}: 실패 ({r['error']})"); continue
            print(f"  {k:>12}: cam {r.get('cam_mm', float('nan')):.3f}mm/{r.get('cam_deg', float('nan')):.3f}° "
                  f"gtc {r.get('gtc_mm', float('nan')):.3f}mm/{r.get('gtc_deg', float('nan')):.3f}° "
                  f"grasp {r.get('grasp_mm', float('nan')):.3f}mm cube {r.get('cube_mm', float('nan')):.3f}mm | "
                  f"train {r['train_px']:.3f}px xview {r.get('cross_view_mean_px', float('nan')):.2f}px camcom {r.get('cam_common_mm', float('nan')):.2f}mm")
    return results


# ---------------------------------------------------------------- 스윕
SWEEP = [
    ("pixel_rand_px", [0.2, 0.5, 1.0]),
    ("fk_sys_mm", [0.5, 1.0, 2.0]), ("fk_rand_mm", [0.1, 0.5, 1.0]),
    ("place_sys_mm", [0.5, 1.0, 2.0]), ("place_rand_mm", [0.3, 0.5, 1.0]),
    ("cam_ext_sys_mm", [2.0, 5.0, 10.0]), ("cam_drift_mm", [0.5, 1.0, 2.0]),
    ("cam_int_sys_focal", [0.002, 0.005, 0.01]), ("cam_int_pp_px", [2.0, 5.0, 10.0]),
    ("geom_sys_mm", [0.3, 0.5, 1.0]), ("geom_rand_mm", [0.2, 0.5, 1.0]),
    ("grasp_sys_mm", [0.5, 1.0, 2.0]),
    ("pixel_outlier_frac", [0.02, 0.05, 0.1]),
]
# 계통 오차에 각도도 같이 주는 짝 (mm 하나만 돌리면 회전 민감도를 못 보므로)
PAIRED_DEG = {"fk_sys_mm": "fk_sys_deg", "place_sys_mm": "place_sys_deg", "cam_ext_sys_mm": "cam_ext_sys_deg",
              "cam_drift_mm": "cam_drift_deg", "grasp_sys_mm": "grasp_sys_deg",
              "fk_rand_mm": "fk_rand_deg", "place_rand_mm": "place_rand_deg"}
MM_TO_DEG = 0.1  # 1mm 계통/랜덤 오차에 0.1° 회전을 짝지음 (표준 배율, 별도 실험 가능)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=str(SCENE_JSON_DEFAULT))
    ap.add_argument("--build-scene", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--base-pixel", type=float, default=0.3, help="스윕 시 기본으로 깔아두는 코너 검출 노이즈(px)")
    ap.add_argument("--out", default=str(REPO_ROOT / "zeus_gello_calibration" / "results" / "sim" / "sim_results.json"))
    for f in NoiseConfig.__dataclass_fields__:
        ap.add_argument(f"--{f.replace('_', '-')}", type=float, default=None)
    # build-scene용 (load_all_data 인자)
    ap.add_argument("--session1-dir", default=str(SESSION1_DIR_DEFAULT)); ap.add_argument("--session1-capture-subdir", default="capture_replayed")
    ap.add_argument("--session2-dir", default=str(SESSION2_DIR_DEFAULT)); ap.add_argument("--session2-capture-subdir", default="capture_placed")
    ap.add_argument("--session3-dir", default=str(SESSION3_DIR_DEFAULT)); ap.add_argument("--session3-capture-subdir", default="capture_replayed")
    ap.add_argument("--zeus-intrinsics-dir", default=str(REPO_ROOT / "intrinsics")); ap.add_argument("--ur3-intrinsics-dir", default=str(REPO_ROOT / "ur3_calibration" / "intrinsics"))
    ap.add_argument("--device-map", default=str(REPO_ROOT / "intrinsics" / "device_map.json"))
    ap.add_argument("--fit-json", default=str(REPO_ROOT / "zeus_gello_calibration" / "pass1_grasp_offset_replayed.json"))
    ap.add_argument("--fixed-min-corners", type=int, default=8); ap.add_argument("--cube-observation-policy", default="legacy")
    args = ap.parse_args()

    if args.build_scene:
        build_scene(args); return
    scene = load_scene(args.scene)

    if args.run:
        kw = {f: getattr(args, f) for f in NoiseConfig.__dataclass_fields__ if getattr(args, f) is not None}
        noise = NoiseConfig(**kw)
        print("noise:", {k: v for k, v in asdict(noise).items() if v})
        results = run_once(scene, noise, verbose=True)
        Path(args.out).write_text(json.dumps({"noise": asdict(noise), "results": results}, indent=2, default=str))
        print(f"wrote {args.out}"); return

    if args.sweep:
        rows = []
        def record(tag, level, noise):
            for seed in range(args.seeds):
                noise.seed = seed
                t0 = time.time(); res = run_once(scene, noise); dt = time.time() - t0
                for method, r in res.items():
                    if method == "counts" or "error" in r:
                        continue
                    rows.append({"factor": tag, "level": level, "seed": seed, "method": method, **r, "elapsed_s": dt})
                print(f"  {tag}={level} seed={seed} ({dt:.0f}s)")
        print("=== zero noise ===")
        record("zero", 0.0, NoiseConfig())
        print(f"=== pixel only (base {args.base_pixel}px) ===")
        record("pixel_base", args.base_pixel, NoiseConfig(pixel_rand_px=args.base_pixel))
        for factor, levels in SWEEP:
            print(f"=== {factor} ===")
            for lv in levels:
                kw = {"pixel_rand_px": args.base_pixel, factor: lv}
                if factor in PAIRED_DEG:
                    kw[PAIRED_DEG[factor]] = lv * MM_TO_DEG
                record(factor, lv, NoiseConfig(**kw))
        Path(args.out).write_text(json.dumps(rows, indent=1, default=str))
        # 요약 표: 요인/레벨/방식별 평균
        import collections
        agg = collections.defaultdict(list)
        for r in rows:
            agg[(r["factor"], r["level"], r["method"])].append(r)
        print(f"\n{'factor':>18} {'level':>6} {'method':>12} {'cam mm':>7} {'gtc mm':>7} {'grasp':>6} {'cube':>6} {'train':>6} {'xview':>6} {'camcom':>7}")
        for (f, lv, m), rs in agg.items():
            g = lambda k: np.mean([x.get(k, np.nan) for x in rs])
            print(f"{f:>18} {lv:>6} {m:>12} {g('cam_mm'):>7.3f} {g('gtc_mm'):>7.3f} {g('grasp_mm'):>6.3f} {g('cube_mm'):>6.3f} {g('train_px'):>6.3f} {g('cross_view_mean_px'):>6.2f} {g('cam_common_mm'):>7.3f}")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
